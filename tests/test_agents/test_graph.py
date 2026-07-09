"""Tests for LangGraph orchestrated multi-agent graph."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agents.graph import build_graph, run_chat_graph
from agents.state import initial_state_from_chat_request
from schemas.chat import ChatRequest
from schemas.routing import AgentRoute


def test_build_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None
    assert callable(graph.invoke)


@patch("agents.graph.run_prediction_agent")
def test_graph_routes_prediction_message(mock_predict: Any) -> None:
    mock_predict.return_value = {
        "user_message": "Predict ALT for BMI 30",
        "session_id": "sess-1",
        "route": AgentRoute.PREDICTION.value,
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
        "agent_steps": ["orchestrator:prediction", "prediction"],
    }

    state = initial_state_from_chat_request(
        ChatRequest(message="Predict ALT for BMI 30", session_id="sess-1"),
    )
    final_state = build_graph().invoke(state)

    mock_predict.assert_called_once()
    assert final_state.get("route") == AgentRoute.PREDICTION.value
    assert final_state.get("response_text") == "ALT prediction complete."


@patch("agents.graph.run_prediction_agent")
def test_run_chat_graph_returns_chat_response(mock_predict: Any) -> None:
    mock_predict.return_value = {
        "user_message": "Predict ALT for BMI 30",
        "session_id": "sess-2",
        "route": AgentRoute.PREDICTION.value,
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
        "agent_steps": ["orchestrator:prediction", "prediction"],
    }

    response = run_chat_graph(
        ChatRequest(message="Predict ALT for BMI 30", session_id="sess-2"),
    )

    assert response.text == "Estimated ALT is about 30."
    assert response.session_id == "sess-2"
    assert response.prediction is not None
    assert response.metadata.routed_to == AgentRoute.PREDICTION.value


def test_graph_routes_sql_message_to_data_stub() -> None:
    response = run_chat_graph(
        ChatRequest(message="Show me a SQL query for readmissions by month", session_id="sess-3"),
    )

    assert response.metadata.routed_to == AgentRoute.DATA.value
    assert response.prediction is None
    assert "data agent" in response.text.lower()


def test_graph_guardrail_block_uses_fallback() -> None:
    response = run_chat_graph(
        ChatRequest(message="What medication should the patient take for COPD?", session_id="sess-4"),
    )

    assert response.metadata.routed_to == AgentRoute.FALLBACK.value
    assert response.metadata.guardrail_blocked is True
    assert response.prediction is None
