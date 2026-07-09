"""Tests for LangGraph prediction slice."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agents.graph import build_graph, run_chat_graph
from agents.state import initial_state_from_chat_request
from schemas.chat import ChatRequest


def test_build_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None
    assert callable(graph.invoke)


@patch("agents.graph.run_prediction_agent")
def test_graph_invoke_routes_through_prediction_node(mock_predict: Any) -> None:
    mock_predict.return_value = {
        "user_message": "Predict ALT for BMI 30",
        "session_id": "sess-1",
        "response_text": "ALT prediction complete.",
        "prediction_result": {
            "target": "alt",
            "prediction": 30.0,
            "can_predict": True,
            "used_features": {"bmi": 30.0},
            "defaults_used": {},
            "missing_required": [],
        },
        "llm_model": "gpt-4o-mini+gpt-4o",
        "latency_ms": 12.5,
    }

    state = initial_state_from_chat_request(
        ChatRequest(message="Predict ALT for BMI 30", session_id="sess-1"),
    )
    final_state = build_graph().invoke(state)

    mock_predict.assert_called_once()
    assert final_state.get("response_text") == "ALT prediction complete."
    assert final_state.get("prediction_result") is not None


@patch("agents.graph.run_prediction_agent")
def test_run_chat_graph_returns_chat_response(mock_predict: Any) -> None:
    mock_predict.return_value = {
        "user_message": "Predict ALT for BMI 30",
        "session_id": "sess-2",
        "response_text": "Estimated ALT is about 30.",
        "prediction_result": {
            "target": "alt",
            "prediction": 30.0,
            "can_predict": True,
            "used_features": {"bmi": 30.0},
            "defaults_used": {},
            "missing_required": [],
        },
        "llm_model": "gpt-4o-mini+gpt-4o",
        "latency_ms": 20.0,
    }

    response = run_chat_graph(
        ChatRequest(message="Predict ALT for BMI 30", session_id="sess-2"),
    )

    assert response.text == "Estimated ALT is about 30."
    assert response.session_id == "sess-2"
    assert response.prediction is not None
    assert response.prediction.prediction == 30.0


@patch("agents.graph.run_prediction_agent")
def test_run_chat_graph_non_prediction_message(mock_predict: Any) -> None:
    mock_predict.return_value = {
        "user_message": "Show SQL for readmissions",
        "session_id": "sess-3",
        "response_text": "I only handle COPD/ALT predictions.",
        "llm_model": "gpt-4o-mini+gpt-4o",
        "latency_ms": 5.0,
    }

    response = run_chat_graph(
        ChatRequest(message="Show SQL for readmissions", session_id="sess-3"),
    )

    assert "COPD/ALT" in response.text
    assert response.prediction is None
    assert response.predictions is None
