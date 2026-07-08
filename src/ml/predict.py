"""Load trained models and run COPD / ALT inference with transparency metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import settings
from ml.features import (
    ALT_CAT_COLS,
    ALT_FEATURE_COLS,
    ALT_NUM_COLS,
    ALT_OPTIONAL_COLS,
    ALT_REQUIRED_COLS,
    COPD_FEATURE_COLS,
    COPD_OPTIONAL_COLS,
    COPD_REQUIRED_COLS,
    FeatureImputationResult,
    get_data_profile,
    impute_features,
)

MODELS_DIR = settings.artifacts_dir / "models"

COPD_PIPELINE_PATH = MODELS_DIR / "copd_pipeline.joblib"
COPD_LABEL_ENCODER_PATH = MODELS_DIR / "copd_label_encoder.joblib"
ALT_PIPELINE_PATH = MODELS_DIR / "alt_pipeline.joblib"


@dataclass(frozen=True)
class PredictionResult:
    target: str
    prediction: str | float | None
    can_predict: bool
    used_features: dict[str, Any]
    defaults_used: dict[str, dict[str, Any]]
    missing_required: list[str]
    class_probabilities: dict[str, float] | None = None


class ModelRegistry:
    """Lazy-loaded singleton for trained model artifacts."""

    _copd_pipeline: Pipeline | None = None
    _copd_label_encoder: LabelEncoder | None = None
    _alt_pipeline: Pipeline | None = None

    @classmethod
    def load(cls, models_dir: Path | None = None) -> None:
        base = models_dir or MODELS_DIR
        copd_pipeline_path = base / "copd_pipeline.joblib"
        copd_label_encoder_path = base / "copd_label_encoder.joblib"
        alt_pipeline_path = base / "alt_pipeline.joblib"

        for path in (copd_pipeline_path, copd_label_encoder_path, alt_pipeline_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing model artifact: {path}. Run `uv run python -m ml.train` first."
                )

        cls._copd_pipeline = joblib.load(copd_pipeline_path)
        cls._copd_label_encoder = joblib.load(copd_label_encoder_path)
        cls._alt_pipeline = joblib.load(alt_pipeline_path)

    @classmethod
    def reset(cls) -> None:
        cls._copd_pipeline = None
        cls._copd_label_encoder = None
        cls._alt_pipeline = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._copd_pipeline is None or cls._copd_label_encoder is None or cls._alt_pipeline is None:
            cls.load()

    @classmethod
    def copd_pipeline(cls) -> Pipeline:
        cls._ensure_loaded()
        assert cls._copd_pipeline is not None
        return cls._copd_pipeline

    @classmethod
    def copd_label_encoder(cls) -> LabelEncoder:
        cls._ensure_loaded()
        assert cls._copd_label_encoder is not None
        return cls._copd_label_encoder

    @classmethod
    def alt_pipeline(cls) -> Pipeline:
        cls._ensure_loaded()
        assert cls._alt_pipeline is not None
        return cls._alt_pipeline


def _to_result(
    *,
    target: str,
    imputation: FeatureImputationResult,
    prediction: str | float | None,
    class_probabilities: dict[str, float] | None = None,
) -> PredictionResult:
    return PredictionResult(
        target=target,
        prediction=prediction,
        can_predict=imputation.can_predict,
        used_features=imputation.used_features,
        defaults_used=imputation.defaults_used,
        missing_required=imputation.missing_required,
        class_probabilities=class_probabilities,
    )


def _features_to_frame(used_features: Mapping[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{col: used_features[col] for col in feature_cols}])


def predict_copd(raw_features: Mapping[str, Any]) -> PredictionResult:
    """Predict COPD severity class (A/B/C/D) from raw feature values."""
    imputation = impute_features(
        raw_features,
        required_cols=COPD_REQUIRED_COLS,
        optional_cols=COPD_OPTIONAL_COLS,
        numeric_cols=[],
        categorical_cols=COPD_FEATURE_COLS,
        data_profile=get_data_profile(),
    )
    if not imputation.can_predict:
        return _to_result(target="copd", imputation=imputation, prediction=None)

    X = _features_to_frame(imputation.used_features, COPD_FEATURE_COLS)
    pipeline = ModelRegistry.copd_pipeline()
    label_encoder = ModelRegistry.copd_label_encoder()

    pred_enc = pipeline.predict(X)
    pred = str(label_encoder.inverse_transform(pred_enc.astype(int))[0])

    class_probabilities: dict[str, float] | None = None
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(pipeline.named_steps["prep"].transform(X))[0]
        class_probabilities = {
            str(label): float(score)
            for label, score in zip(label_encoder.classes_, proba, strict=True)
        }

    return _to_result(
        target="copd",
        imputation=imputation,
        prediction=pred,
        class_probabilities=class_probabilities,
    )


def predict_alt(raw_features: Mapping[str, Any]) -> PredictionResult:
    """Predict ALT (alanine aminotransferase) from raw feature values."""
    imputation = impute_features(
        raw_features,
        required_cols=ALT_REQUIRED_COLS,
        optional_cols=ALT_OPTIONAL_COLS,
        numeric_cols=ALT_NUM_COLS,
        categorical_cols=ALT_CAT_COLS,
        data_profile=get_data_profile(),
    )
    if not imputation.can_predict:
        return _to_result(target="alt", imputation=imputation, prediction=None)

    X = _features_to_frame(imputation.used_features, ALT_FEATURE_COLS)
    pipeline = ModelRegistry.alt_pipeline()
    pred = float(pipeline.predict(X)[0])

    return _to_result(target="alt", imputation=imputation, prediction=pred)


def predict_all(raw_features: Mapping[str, Any]) -> dict[str, PredictionResult]:
    """Run both COPD and ALT predictions for the same raw feature dict."""
    return {
        "copd": predict_copd(raw_features),
        "alt": predict_alt(raw_features),
    }


def main() -> None:
    sample = {
        "bmi": 28.5,
        "diet_quality": "Good",
        "exercise_frequency": "Moderate",
        "income_bracket": "Middle",
        "urban": 1,
        "diagnosis_code": "J44.9",
        "smoker": False,
    }
    results = predict_all(sample)
    for name, result in results.items():
        print(f"{name.upper()}: {result.prediction}")
        if result.defaults_used:
            print(f"  defaults_used: {result.defaults_used}")
        if result.class_probabilities:
            top = max(result.class_probabilities.items(), key=lambda item: item[1])
            print(f"  top class probability: {top[0]}={top[1]:.3f}")


if __name__ == "__main__":
    main()
