"""Tests for chat API schemas."""

from schemas.chat import ChatRequest, ChatResponse, ChatPredictionDetails
from schemas.prediction import PredictionResponse


def test_chat_request_defaults() -> None:
    req = ChatRequest(message="Predict ALT for BMI 30")
    assert req.user_id == "default-user"
    assert len(req.session_id) > 0


def test_chat_prediction_details_from_prediction_response() -> None:
    details = ChatPredictionDetails.from_prediction_response(
        PredictionResponse(
            target="alt",
            prediction=30.1,
            can_predict=True,
            used_features={"bmi": 30.0},
            defaults_used={},
        ),
        top_global_factors=["bmi", "diet_quality"],
    )
    assert details.prediction == 30.1
    assert details.top_global_factors == ["bmi", "diet_quality"]


def test_chat_response_single_prediction() -> None:
    response = ChatResponse.from_prediction_results(
        text="Predicted ALT is 30.1.",
        session_id="sess-1",
        result=PredictionResponse(
            target="alt",
            prediction=30.1,
            can_predict=True,
            used_features={"bmi": 30.0},
            defaults_used={},
        ),
        llm_model="gpt-4o-mini",
        latency_ms=120.5,
    )
    assert response.prediction is not None
    assert response.predictions is None
    assert response.metadata.extraction_method == "llm"
    assert response.metadata.llm_model == "gpt-4o-mini"


def test_chat_response_both_predictions() -> None:
    response = ChatResponse.from_prediction_results(
        text="COPD and ALT results below.",
        session_id="sess-2",
        result={
            "copd": PredictionResponse(
                target="copd",
                prediction="C",
                can_predict=True,
                used_features={"diet_quality": "Good", "exercise_frequency": "Moderate"},
                defaults_used={},
            ),
            "alt": PredictionResponse(
                target="alt",
                prediction=28.0,
                can_predict=True,
                used_features={"bmi": 28.0},
                defaults_used={},
            ),
        },
    )
    assert response.prediction is None
    assert response.predictions is not None
    assert set(response.predictions) == {"copd", "alt"}
