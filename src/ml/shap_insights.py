"""Compute and serialize SHAP summaries for Insight Tool artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ml.features import ALT_FEATURE_COLS, COPD_FEATURE_COLS

ALL_SOURCE_FEATURES = sorted(
    set(COPD_FEATURE_COLS + ALT_FEATURE_COLS),
    key=len,
    reverse=True,
)

DEFAULT_SHAP_SAMPLE_SIZE = 1000


def _pipeline_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["prep"]
    return list(preprocessor.get_feature_names_out())


def _source_column(encoded_name: str) -> str:
    if encoded_name.startswith("num__"):
        return encoded_name.removeprefix("num__")
    if encoded_name.startswith("cat__"):
        body = encoded_name.removeprefix("cat__")
        for column in ALL_SOURCE_FEATURES:
            if body == column or body.startswith(f"{column}_"):
                return column
    return encoded_name


def _aggregate_shap_to_source(
    feature_names: list[str],
    mean_abs_values: np.ndarray,
) -> list[dict[str, float | str]]:
    grouped: dict[str, float] = {}
    for name, value in zip(feature_names, mean_abs_values, strict=True):
        source = _source_column(name)
        grouped[source] = grouped.get(source, 0.0) + float(value)

    return [
        {"feature": feature, "mean_abs_shap": score}
        for feature, score in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    ]


def _top_encoded_features(
    feature_names: list[str],
    mean_abs_values: np.ndarray,
    *,
    limit: int = 15,
) -> list[dict[str, float | str]]:
    pairs = sorted(
        zip(feature_names, mean_abs_values, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        {"feature": name, "mean_abs_shap": float(score)}
        for name, score in pairs[:limit]
    ]


def _sample_rows(X: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if len(X) <= sample_size:
        return X
    return X.sample(n=sample_size, random_state=random_state)


def compute_copd_shap_summary(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    sample_size: int = DEFAULT_SHAP_SAMPLE_SIZE,
    random_state: int = 42,
) -> dict[str, Any]:
    """Mean |SHAP| per encoded and source feature for multiclass XGBoost."""
    sample = _sample_rows(X, sample_size, random_state)
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    feature_names = _pipeline_feature_names(pipeline)

    X_transformed = preprocessor.transform(sample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(values).mean(axis=0)

    return {
        "target": "copd",
        "model": "xgb",
        "method": "shap_tree",
        "sample_size": len(sample),
        "top_features": _aggregate_shap_to_source(feature_names, mean_abs),
        "encoded_top_features": _top_encoded_features(feature_names, mean_abs),
    }


def compute_alt_shap_summary(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    sample_size: int = DEFAULT_SHAP_SAMPLE_SIZE,
    random_state: int = 42,
) -> dict[str, Any]:
    """Mean |SHAP| per encoded and source feature for Ridge regression."""
    sample = _sample_rows(X, sample_size, random_state)
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    feature_names = _pipeline_feature_names(pipeline)

    X_transformed = preprocessor.transform(sample)
    explainer = shap.LinearExplainer(model, X_transformed)
    shap_values = np.asarray(explainer.shap_values(X_transformed), dtype=float)
    mean_abs = np.abs(shap_values).mean(axis=0)

    return {
        "target": "alt",
        "model": "ridge",
        "method": "shap_linear",
        "sample_size": len(sample),
        "top_features": _aggregate_shap_to_source(feature_names, mean_abs),
        "encoded_top_features": _top_encoded_features(feature_names, mean_abs),
    }
