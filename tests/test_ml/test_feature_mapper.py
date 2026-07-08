"""Tests for feature mapping and prediction request flow."""

from ml.feature_mapper import map_patient_features, run_prediction
from schemas.prediction import PatientFeatures, PredictionRequest, PredictionResponse, PredictionTarget


def test_map_exercise_athlete_to_high() -> None:
    mapped, notes = map_patient_features({"exercise_frequency": "athlete"})

    assert mapped["exercise_frequency"] == "High"
    assert len(notes) == 1
    assert notes[0].field == "exercise_frequency"


def test_map_diet_poor_synonym() -> None:
    mapped, notes = map_patient_features({"diet_quality": "poor"})

    assert mapped["diet_quality"] == "Poor"
    assert len(notes) == 1


def test_map_urban_center_of_city() -> None:
    mapped, notes = map_patient_features({"urban": "center of city"})

    assert mapped["urban"] == 1
    assert notes[0].mapped == 1


def test_map_smoker_non_smoker() -> None:
    mapped, _ = map_patient_features({"smoker": "non-smoker"})

    assert mapped["smoker"] is False


def test_run_prediction_copd_with_mapped_features(ml_artifacts: None) -> None:
    request = PredictionRequest(
        target=PredictionTarget.COPD,
        features=PatientFeatures(
            diet_quality="good",
            exercise_frequency="sportista",
            urban="city",
            smoker="no",
        ),
        raw_query="Predict COPD for active urban patient with good diet",
    )
    response = run_prediction(request)

    assert isinstance(response, PredictionResponse)
    assert response.can_predict
    assert response.prediction in {"A", "B", "C", "D"}
    assert response.used_features["diet_quality"] == "Good"
    assert response.used_features["exercise_frequency"] == "High"
    assert response.used_features["urban"] == 1
    assert any(note.field == "exercise_frequency" for note in response.mapping_notes)


def test_run_prediction_alt_requires_bmi(ml_artifacts: None) -> None:
    request = PredictionRequest(
        target=PredictionTarget.ALT,
        features=PatientFeatures(diet_quality="Average"),
    )
    response = run_prediction(request)

    assert isinstance(response, PredictionResponse)
    assert not response.can_predict
    assert response.missing_required == ["bmi"]


def test_run_prediction_both_targets(ml_artifacts: None) -> None:
    request = PredictionRequest(
        target=PredictionTarget.BOTH,
        features=PatientFeatures(
            bmi=27.0,
            diet_quality="Good",
            exercise_frequency="Moderate",
        ),
    )
    results = run_prediction(request)

    assert isinstance(results, dict)
    assert results["copd"].can_predict
    assert results["alt"].can_predict
    assert isinstance(results["alt"].prediction, float)
