"""Build and persist dataset profiles from patient data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from data.loader import PatientDataLoader
from data.schema_registry import COLUMN_DEFINITIONS, get_dataset_schema
from schemas.data import ColumnProfile, DataProfile, DatasetSchema


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _mode_value(series: pd.Series) -> str | int | float | None:
    if series.empty:
        return None

    mode = series.mode(dropna=True)
    if mode.empty:
        return None

    value = mode.iloc[0]
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def build_column_profile(series: pd.Series, name: str) -> ColumnProfile:
    non_null = series.dropna()
    null_count = int(series.isna().sum())
    count = int(len(series))
    dtype = str(series.dtype)
    unique_count = int(non_null.nunique())

    profile = ColumnProfile(
        name=name,
        dtype=dtype,
        count=count,
        null_count=null_count,
        unique_count=unique_count,
    )

    if non_null.empty:
        return profile

    if _is_numeric(non_null):
        profile.min = float(non_null.min())
        profile.max = float(non_null.max())
        profile.mean = round(float(non_null.mean()), 4)
        profile.mode = _mode_value(non_null)
        return profile

    string_values = non_null.astype(str)
    profile.min = string_values.min()
    profile.max = string_values.max()
    profile.mode = _mode_value(string_values)

    if unique_count <= 20:
        profile.unique_values = sorted(string_values.unique().tolist())

    return profile


def build_data_profile(loader: PatientDataLoader) -> DataProfile:
    dataframe = loader.get_dataframe()
    columns = [
        build_column_profile(dataframe[column.name], column.name)
        for column in COLUMN_DEFINITIONS
        if column.name in dataframe.columns
    ]

    return DataProfile(
        row_count=loader.row_count,
        columns=columns,
    )


def save_data_profile(profile: DataProfile, path: Path | None = None) -> Path:
    output_path = path or settings.artifacts_dir / "data_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return output_path


def load_data_profile(path: Path | None = None) -> DataProfile:
    profile_path = path or settings.artifacts_dir / "data_profile.json"
    return DataProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))


def build_and_save_data_profile(
    csv_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[DataProfile, DatasetSchema, Path]:
    with PatientDataLoader(csv_path=csv_path) as loader:
        profile = build_data_profile(loader)
        schema = get_dataset_schema(loader.row_count)

    saved_path = save_data_profile(profile, output_path)
    return profile, schema, saved_path
