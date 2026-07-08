"""Integration tests for ML inference."""

import json

import pytest

from ml.predict import predict_all, predict_alt, predict_copd
from ml.train import (
    ALT_IMPORTANCE_PATH,
    ALT_SHAP_PATH,
    COPD_IMPORTANCE_PATH,
    COPD_SHAP_PATH,
    METRICS_PATH,
)

FULL_FEATURES = {
    "bmi": 28.5,
    "diet_quality": "Good",
    "exercise_frequency": "Moderate",
    "income_bracket": "Middle",
    "urban": 1,
    "diagnosis_code": "J44.9",
    "smoker": False,
    "readmitted": 0,
    "albumin_globulin_ratio": 0.5,
}


def test_predict_copd_returns_valid_class(ml_artifacts: None) -> None:
    result = predict_copd(FULL_FEATURES)

    assert result.can_predict
    assert result.prediction in {"A", "B", "C", "D"}
    assert result.missing_required == []
    assert result.class_probabilities is not None
    assert set(result.class_probabilities) == {"A", "B", "C", "D"}
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_predict_alt_returns_float_near_bmi(ml_artifacts: None) -> None:
    """Ridge ALT model is largely BMI-driven in this dataset."""
    result = predict_alt({"bmi": 30.0})

    assert result.can_predict
    assert isinstance(result.prediction, float)
    assert result.prediction == pytest.approx(30.0, rel=0.02)


def test_predict_alt_missing_bmi(ml_artifacts: None) -> None:
    result = predict_alt({"diet_quality": "Good"})

    assert not result.can_predict
    assert result.prediction is None
    assert result.missing_required == ["bmi"]


def test_predict_all_returns_both_targets(ml_artifacts: None) -> None:
    results = predict_all(FULL_FEATURES)

    assert set(results) == {"copd", "alt"}
    assert results["copd"].can_predict
    assert results["alt"].can_predict


def test_training_insight_artifacts_have_expected_shape(ml_artifacts: None) -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    copd_importance = json.loads(COPD_IMPORTANCE_PATH.read_text(encoding="utf-8"))
    alt_importance = json.loads(ALT_IMPORTANCE_PATH.read_text(encoding="utf-8"))

    assert "copd" in metrics and "alt" in metrics
    assert "holdout_metrics" in metrics["copd"]
    assert "holdout_metrics" in metrics["alt"]
    assert metrics["copd"]["holdout_metrics"]["accuracy"] <= 1.0

    assert copd_importance["target"] == "copd"
    assert alt_importance["target"] == "alt"
    assert len(copd_importance["top_features"]) > 0
    assert len(alt_importance["top_features"]) > 0


def test_shap_summary_artifacts_have_expected_shape(ml_artifacts: None) -> None:
    copd_shap = json.loads(COPD_SHAP_PATH.read_text(encoding="utf-8"))
    alt_shap = json.loads(ALT_SHAP_PATH.read_text(encoding="utf-8"))

    assert copd_shap["method"] == "shap_tree"
    assert alt_shap["method"] == "shap_linear"
    assert copd_shap["sample_size"] > 0
    assert len(copd_shap["top_features"]) > 0
    assert len(alt_shap["top_features"]) > 0
    assert "mean_abs_shap" in copd_shap["top_features"][0]
