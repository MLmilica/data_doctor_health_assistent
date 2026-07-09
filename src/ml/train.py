"""Train final COPD and ALT models and persist artifacts for inference."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import settings
from data.loader import PatientDataLoader
from ml.features import (
    ALT_FEATURE_COLS,
    COPD_FEATURE_COLS,
    build_alt_preprocessor,
    build_copd_preprocessor,
    fit_copd_label_encoder,
)
from ml.shap_insights import compute_alt_shap_summary, compute_copd_shap_summary

RANDOM_STATE = 42
TEST_SIZE = 0.2

COPD_TARGET = "chronic_obstructive_pulmonary_disease"
ALT_TARGET = "alanine_aminotransferase"

MODELS_DIR = settings.artifacts_dir / "models"
INSIGHTS_DIR = settings.artifacts_dir / "insights"

COPD_PIPELINE_PATH = MODELS_DIR / "copd_pipeline.joblib"
COPD_LABEL_ENCODER_PATH = MODELS_DIR / "copd_label_encoder.joblib"
ALT_PIPELINE_PATH = MODELS_DIR / "alt_pipeline.joblib"
METRICS_PATH = INSIGHTS_DIR / "ml_metrics.json"
COPD_IMPORTANCE_PATH = INSIGHTS_DIR / "copd_feature_importance.json"
ALT_IMPORTANCE_PATH = INSIGHTS_DIR / "alt_feature_importance.json"
COPD_SHAP_PATH = INSIGHTS_DIR / "copd_shap_summary.json"
ALT_SHAP_PATH = INSIGHTS_DIR / "alt_shap_summary.json"


def _ensure_artifact_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_copd_pipeline(label_encoder_classes: np.ndarray) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_copd_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    objective="multi:softmax",
                    num_class=len(label_encoder_classes),
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    verbosity=0,
                ),
            ),
        ]
    )


def _build_alt_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_alt_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _pipeline_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["prep"]
    return list(preprocessor.get_feature_names_out())


def _extract_importance(
    pipeline: Pipeline,
    *,
    target: str,
    model_name: str,
) -> dict[str, Any]:
    model = pipeline.named_steps["model"]
    feature_names = _pipeline_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        scores = np.abs(np.ravel(model.coef_))
    else:
        scores = np.zeros(len(feature_names), dtype=float)

    pairs = sorted(
        zip(feature_names, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    top_features = [
        {"feature": name, "score": float(score)}
        for name, score in pairs
    ]

    return {
        "target": target,
        "model": model_name,
        "top_features": top_features,
    }


def train_models(csv_path: Path | None = None) -> dict[str, Any]:
    """Train COPD + ALT models, evaluate on holdout, refit on full data, save artifacts."""
    _ensure_artifact_dirs()

    with PatientDataLoader(csv_path=csv_path or settings.patient_csv_path) as loader:
        df = loader.get_dataframe().copy()

    X_copd = cast(pd.DataFrame, df[COPD_FEATURE_COLS])
    y_copd = cast(pd.Series, df[COPD_TARGET])
    X_alt = cast(pd.DataFrame, df[ALT_FEATURE_COLS])
    y_alt = cast(pd.Series, df[ALT_TARGET])

    label_encoder = fit_copd_label_encoder(y_copd)

    X_copd_train, X_copd_test, y_copd_train, y_copd_test = train_test_split(
        X_copd,
        y_copd,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_copd,
    )
    X_alt_train, X_alt_test, y_alt_train, y_alt_test = train_test_split(
        X_alt,
        y_alt,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    y_copd_train_series = cast(pd.Series, y_copd_train)
    y_copd_train_enc = label_encoder.transform(y_copd_train_series.astype(str))
    copd_sample_weight = compute_sample_weight(class_weight="balanced", y=y_copd_train)

    label_classes = cast(np.ndarray, label_encoder.classes_)
    copd_pipeline = _build_copd_pipeline(label_classes)
    copd_pipeline.fit(X_copd_train, y_copd_train_enc, model__sample_weight=copd_sample_weight)
    copd_pred_test_enc = copd_pipeline.predict(X_copd_test)
    copd_pred_test = label_encoder.inverse_transform(copd_pred_test_enc.astype(int))
    copd_metrics = {
        "accuracy": float(accuracy_score(y_copd_test, copd_pred_test)),
        "macro_f1": float(f1_score(y_copd_test, copd_pred_test, average="macro")),
    }

    alt_pipeline = _build_alt_pipeline()
    alt_pipeline.fit(X_alt_train, y_alt_train)
    alt_pred_test = alt_pipeline.predict(X_alt_test)
    alt_metrics = {
        "mae": float(mean_absolute_error(y_alt_test, alt_pred_test)),
        "rmse": float(root_mean_squared_error(y_alt_test, alt_pred_test)),
        "r2": float(r2_score(y_alt_test, alt_pred_test)),
    }

    # Refit on full dataset for deployment artifacts
    y_copd_full_enc = label_encoder.transform(y_copd.astype(str))
    copd_sample_weight_full = compute_sample_weight(class_weight="balanced", y=y_copd)
    final_copd_pipeline = _build_copd_pipeline(label_classes)
    final_copd_pipeline.fit(
        X_copd,
        y_copd_full_enc,
        model__sample_weight=copd_sample_weight_full,
    )

    final_alt_pipeline = _build_alt_pipeline()
    final_alt_pipeline.fit(X_alt, y_alt)

    joblib.dump(final_copd_pipeline, COPD_PIPELINE_PATH)
    joblib.dump(label_encoder, COPD_LABEL_ENCODER_PATH)
    joblib.dump(final_alt_pipeline, ALT_PIPELINE_PATH)

    copd_importance = _extract_importance(
        final_copd_pipeline,
        target="copd",
        model_name="xgb",
    )
    alt_importance = _extract_importance(
        final_alt_pipeline,
        target="alt",
        model_name="ridge",
    )
    copd_shap = compute_copd_shap_summary(final_copd_pipeline, X_copd, random_state=RANDOM_STATE)
    alt_shap = compute_alt_shap_summary(final_alt_pipeline, X_alt, random_state=RANDOM_STATE)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "copd": {
            "model": "xgb",
            "features": COPD_FEATURE_COLS,
            "holdout_metrics": copd_metrics,
            "near_random_baseline": copd_metrics["accuracy"] < 0.28,
            "limitation_note": (
                "COPD model remains near random baseline (~0.25 for 4 balanced classes). "
                "Use predictions with caution in the POC."
                if copd_metrics["accuracy"] < 0.28
                else None
            ),
        },
        "alt": {
            "model": "ridge",
            "features": ALT_FEATURE_COLS,
            "holdout_metrics": alt_metrics,
            "note": "ALT prediction is largely BMI-driven in this dataset.",
        },
        "artifacts": {
            "copd_pipeline": str(COPD_PIPELINE_PATH),
            "copd_label_encoder": str(COPD_LABEL_ENCODER_PATH),
            "alt_pipeline": str(ALT_PIPELINE_PATH),
            "copd_shap_summary": str(COPD_SHAP_PATH),
            "alt_shap_summary": str(ALT_SHAP_PATH),
        },
    }

    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    COPD_IMPORTANCE_PATH.write_text(json.dumps(copd_importance, indent=2), encoding="utf-8")
    ALT_IMPORTANCE_PATH.write_text(json.dumps(alt_importance, indent=2), encoding="utf-8")
    COPD_SHAP_PATH.write_text(json.dumps(copd_shap, indent=2), encoding="utf-8")
    ALT_SHAP_PATH.write_text(json.dumps(alt_shap, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    summary = train_models()
    print("COPD holdout:", summary["copd"]["holdout_metrics"])
    print("ALT holdout:", summary["alt"]["holdout_metrics"])
    print("Saved models to:", MODELS_DIR)
    print("Saved insights to:", INSIGHTS_DIR)


if __name__ == "__main__":
    main()
