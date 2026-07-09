"""Tests for LangGraph AgentState helpers."""

from schemas.chat import ChatRequest
from schemas.prediction import LLMPredictionExtraction, PatientFeatures, PredictionResponse, PredictionTarget

from agents.state import (
    chat_response_from_state,
    get_extraction,
    get_prediction_result,
    initial_state_from_chat_request,
    set_extraction,
    set_prediction_result,
)


def test_initial_state_from_chat_request() -> None:
    request = ChatRequest(message="Predict ALT", session_id="s1", user_id="u1")
    state = initial_state_from_chat_request(request)

    assert state.get("user_message") == "Predict ALT"
    assert state.get("session_id") == "s1"
    assert state.get("user_id") == "u1"
    assert "extraction" not in state


def test_extraction_round_trip() -> None:
    state = initial_state_from_chat_request(ChatRequest(message="Predict COPD"))
    extraction = LLMPredictionExtraction(
        is_prediction_request=True,
        target=PredictionTarget.COPD,
        features=PatientFeatures(diet_quality="Good", exercise_frequency="Moderate"),
    )
    updated = set_extraction(state, extraction)

    parsed = get_extraction(updated)
    assert parsed is not None
    assert parsed.target == PredictionTarget.COPD
    assert parsed.features.diet_quality == "Good"


def test_prediction_result_single_and_both() -> None:
    state = initial_state_from_chat_request(ChatRequest(message="x"))

    single = PredictionResponse(
        target="alt",
        prediction=30.0,
        can_predict=True,
        used_features={"bmi": 30.0},
        defaults_used={},
    )
    with_single = set_prediction_result(state, single)
    parsed_single = get_prediction_result(with_single)
    assert isinstance(parsed_single, PredictionResponse)
    assert parsed_single.prediction == 30.0

    both = {
        "copd": PredictionResponse(
            target="copd",
            prediction="B",
            can_predict=True,
            used_features={"diet_quality": "Good", "exercise_frequency": "Low"},
            defaults_used={},
        ),
        "alt": single,
    }
    with_both = set_prediction_result(state, both)
    parsed_both = get_prediction_result(with_both)
    assert isinstance(parsed_both, dict)
    assert set(parsed_both) == {"copd", "alt"}


def test_chat_response_from_state() -> None:
    state = initial_state_from_chat_request(
        ChatRequest(message="Predict ALT", session_id="sess-99"),
    )
    state = set_prediction_result(
        state,
        PredictionResponse(
            target="alt",
            prediction=29.5,
            can_predict=True,
            used_features={"bmi": 29.5},
            defaults_used={},
        ),
    )
    state["response_text"] = "ALT prediction complete."
    state["llm_model"] = "gpt-4o-mini"
    state["latency_ms"] = 50.0

    response = chat_response_from_state(state)
    assert response.text == "ALT prediction complete."
    assert response.session_id == "sess-99"
    assert response.prediction is not None
    assert response.prediction.prediction == 29.5
    assert response.metadata.llm_model == "gpt-4o-mini"
