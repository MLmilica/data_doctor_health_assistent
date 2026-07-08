"""
Feature specification and preprocessing helpers for final ML training/inference.

This module implements the "contract" between:
- feature_mapper / prediction agent (raw user inputs)
- sklearn preprocessing pipelines (training + inference)

It is intentionally deterministic:
- required vs optional features
- missing values are imputed using `artifacts/data_profile.json`
- transparency via `defaults_used` and `missing_required`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder

from config import settings
from data.profile import load_data_profile
from schemas.data import ColumnProfile, DataProfile


# -----------------------------
# Feature sets (from notebook)
# -----------------------------

# COPD (classification) — Phase 3 reduced feature set (6 categorical features)
COPD_FEATURE_COLS: list[str] = [
    "diet_quality",
    "income_bracket",
    "urban",
    "diagnosis_code",
    "exercise_frequency",
    "smoker",
]

COPD_REQUIRED_COLS: list[str] = [
    "diet_quality",
    "exercise_frequency",
]

COPD_OPTIONAL_COLS: list[str] = [c for c in COPD_FEATURE_COLS if c not in COPD_REQUIRED_COLS]


# ALT (regression) — Phase 3 reduced feature set (6 features: 2 numeric + 4 categorical)
ALT_FEATURE_COLS: list[str] = [
    "bmi",
    "readmitted",
    "exercise_frequency",
    "albumin_globulin_ratio",
    "diagnosis_code",
    "diet_quality",
]

ALT_REQUIRED_COLS: list[str] = [
    "bmi",
]

ALT_OPTIONAL_COLS: list[str] = [c for c in ALT_FEATURE_COLS if c not in ALT_REQUIRED_COLS]

ALT_NUM_COLS: list[str] = ["bmi", "albumin_globulin_ratio"]
ALT_CAT_COLS: list[str] = [c for c in ALT_FEATURE_COLS if c not in ALT_NUM_COLS]


# -----------------------------
# Typing / return objects
# -----------------------------


@dataclass(frozen=True)
class MissingFeatures:
    missing_required: list[str]


@dataclass(frozen=True)
class FeatureImputationResult:
    used_features: dict[str, Any]
    defaults_used: dict[str, dict[str, Any]]
    missing_required: list[str]

    @property
    def can_predict(self) -> bool:
        return len(self.missing_required) == 0


# -----------------------------
# Helpers: imputation + parsing
# -----------------------------


def _parse_bool(x: Any) -> bool | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _parse_int01(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and x in (0, 1):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s in {"0", "1"}:
            return int(s)
    return None


def _col_profile_by_name(profile: DataProfile, name: str) -> ColumnProfile:
    for col in profile.columns:
        if col.name == name:
            return col
    raise KeyError(f"ColumnProfile for '{name}' not found in data_profile.json")


def _default_for_column(profile: DataProfile, column_name: str, *, is_numeric: bool) -> Any:
    col = _col_profile_by_name(profile, column_name)

    if is_numeric:
        # Prefer mean (more stable), fallback to mode if mean is missing.
        if col.mean is not None:
            return float(col.mean)
        if col.mode is not None:
            return float(col.mode)  # mode may be stored as float/int/string
        raise ValueError(f"No numeric default available for '{column_name}'")

    # categorical/default: mode
    if col.mode is not None:
        return col.mode
    # Extremely rare: dataset column could be empty
    raise ValueError(f"No categorical default available for '{column_name}'")


def _normalize_value_for_column(column_name: str, value: Any) -> Any:
    """
    Bring mapped/enriched values into shapes consistent with training dtypes:
    - readmitted/urban are ints 0/1 in the CSV
    - smoker is bool in the CSV
    - other categoricals are strings
    - numeric are floats
    """

    if column_name in {"readmitted", "urban"}:
        v = _parse_int01(value)
        if v is None:
            raise ValueError(f"Invalid value for '{column_name}': {value!r} (expected 0/1)")
        return v

    if column_name == "smoker":
        v = _parse_bool(value)
        if v is None:
            raise ValueError(f"Invalid value for '{column_name}': {value!r} (expected Yes/No or 0/1)")
        return v

    if column_name in {"bmi", "albumin_globulin_ratio"}:
        if value is None:
            raise ValueError(f"Invalid value for '{column_name}': None")
        return float(value)

    # Generic: exercise_frequency / diet_quality / income_bracket / diagnosis_code
    if value is None:
        raise ValueError(f"Invalid value for '{column_name}': None")
    return str(value)


def impute_features(
    raw_features: Mapping[str, Any],
    *,
    required_cols: list[str],
    optional_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    data_profile: DataProfile,
) -> FeatureImputationResult:
    missing_required = [c for c in required_cols if raw_features.get(c) is None]
    used_features: dict[str, Any] = {}
    defaults_used: dict[str, dict[str, Any]] = {}

    # Required columns
    for col in required_cols:
        if raw_features.get(col) is None:
            continue
        is_numeric = col in numeric_cols
        normalized = _normalize_value_for_column(col, raw_features.get(col))
        used_features[col] = normalized

    # Optional columns
    for col in optional_cols:
        if raw_features.get(col) is None:
            is_numeric = col in numeric_cols
            default_val = _default_for_column(data_profile, col, is_numeric=is_numeric)
            normalized_default = _normalize_value_for_column(col, default_val)
            used_features[col] = normalized_default
            defaults_used[col] = {"value": normalized_default, "source": "data_profile"}
        else:
            used_features[col] = _normalize_value_for_column(col, raw_features.get(col))

    return FeatureImputationResult(
        used_features=used_features,
        defaults_used=defaults_used,
        missing_required=missing_required,
    )


# -----------------------------
# Build sklearn preprocessors
# -----------------------------


def build_copd_preprocessor() -> ColumnTransformer:
    # COPD features are all categorical
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), COPD_FEATURE_COLS),
        ],
        remainder="drop",
    )


def build_alt_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALT_NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ALT_CAT_COLS),
        ],
        remainder="drop",
    )


# -----------------------------
# Train-time utilities
# -----------------------------


def fit_copd_label_encoder(y: pd.Series) -> LabelEncoder:
    """
    COPD labels in CSV are strings: A/B/C/D.
    XGBoost expects numeric class indices, so we fit a LabelEncoder and later inverse_transform outputs.
    """
    le = LabelEncoder()
    le.fit(y.astype(str).values)
    return le


def get_data_profile() -> DataProfile:
    return load_data_profile(path=settings.artifacts_dir / "data_profile.json")

