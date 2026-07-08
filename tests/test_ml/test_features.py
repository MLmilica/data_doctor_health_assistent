"""Unit tests for ML feature contract and imputation."""

from data.profile import build_and_save_data_profile
from ml.features import (
    ALT_CAT_COLS,
    ALT_NUM_COLS,
    ALT_OPTIONAL_COLS,
    ALT_REQUIRED_COLS,
    COPD_FEATURE_COLS,
    COPD_OPTIONAL_COLS,
    COPD_REQUIRED_COLS,
    impute_features,
)


def test_impute_copd_blocks_when_required_missing() -> None:
    profile, _, _ = build_and_save_data_profile()
    result = impute_features(
        {"diet_quality": "Good"},
        required_cols=COPD_REQUIRED_COLS,
        optional_cols=COPD_OPTIONAL_COLS,
        numeric_cols=[],
        categorical_cols=COPD_FEATURE_COLS,
        data_profile=profile,
    )

    assert not result.can_predict
    assert result.missing_required == ["exercise_frequency"]


def test_impute_alt_blocks_when_bmi_missing() -> None:
    profile, _, _ = build_and_save_data_profile()
    result = impute_features(
        {"diet_quality": "Good"},
        required_cols=ALT_REQUIRED_COLS,
        optional_cols=ALT_OPTIONAL_COLS,
        numeric_cols=ALT_NUM_COLS,
        categorical_cols=ALT_CAT_COLS,
        data_profile=profile,
    )

    assert not result.can_predict
    assert result.missing_required == ["bmi"]


def test_impute_copd_fills_optional_from_profile() -> None:
    profile, _, _ = build_and_save_data_profile()
    result = impute_features(
        {
            "diet_quality": "Good",
            "exercise_frequency": "Moderate",
        },
        required_cols=COPD_REQUIRED_COLS,
        optional_cols=COPD_OPTIONAL_COLS,
        numeric_cols=[],
        categorical_cols=COPD_FEATURE_COLS,
        data_profile=profile,
    )

    assert result.can_predict
    assert set(result.used_features) == set(COPD_FEATURE_COLS)
    assert len(result.defaults_used) == len(COPD_OPTIONAL_COLS)
    for col in COPD_OPTIONAL_COLS:
        assert col in result.defaults_used
        assert result.defaults_used[col]["source"] == "data_profile"


def test_impute_normalizes_smoker_and_urban() -> None:
    profile, _, _ = build_and_save_data_profile()
    result = impute_features(
        {
            "diet_quality": "Good",
            "exercise_frequency": "Moderate",
            "income_bracket": "Middle",
            "urban": "1",
            "diagnosis_code": "J44.9",
            "smoker": "no",
        },
        required_cols=COPD_REQUIRED_COLS,
        optional_cols=COPD_OPTIONAL_COLS,
        numeric_cols=[],
        categorical_cols=COPD_FEATURE_COLS,
        data_profile=profile,
    )

    assert result.used_features["urban"] == 1
    assert result.used_features["smoker"] is False
    assert result.defaults_used == {}
