"""Integration tests for multi-step graph orchestration."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agents.graph import build_graph
from agents.state import initial_state_from_chat_request
from schemas.chat import ChatRequest
from schemas.prediction import LLMPredictionExtraction, PatientFeatures, PredictionTarget
from schemas.routing import AgentRoute
from schemas.sql import LLMSQLExtraction


@patch("agents.subagents.synthesize_agent.synthesize_multi_step_response")
@patch("agents.subagents.prediction_agent.synthesize_response_text")
@patch("agents.subagents.prediction_agent.extract_with_llm")
@patch("agents.subagents.data_agent.synthesize_data_response_text")
@patch("agents.subagents.data_agent.extract_sql_with_llm")
def test_graph_multi_step_data_then_prediction(
    mock_extract_sql: Any,
    mock_synthesize_data: Any,
    mock_extract_prediction: Any,
    mock_synthesize_prediction: Any,
    mock_synthesize_multi: Any,
    ml_artifacts: None,
) -> None:
    mock_extract_sql.return_value = LLMSQLExtraction(
        sql="SELECT AVG(bmi) AS avg_bmi FROM patients",
        explanation="Average BMI in dataset.",
    )
    mock_synthesize_data.return_value = "Average BMI is about 29."
    mock_extract_prediction.return_value = LLMPredictionExtraction(
        is_prediction_request=True,
        target=PredictionTarget.ALT,
        features=PatientFeatures(bmi=30.0),
    )
    mock_synthesize_prediction.return_value = "Estimated ALT is about 30."
    mock_synthesize_multi.return_value = (
        "Average BMI in the dataset is about 29, while ALT for BMI 30 is about 30."
    )

    state = initial_state_from_chat_request(
        ChatRequest(
            message="Compare average BMI in the dataset with ALT prediction for BMI 30",
            session_id="multi-1",
        ),
    )
    final_state = build_graph().invoke(state)

    assert final_state.get("orchestrator_action") == "finish"
    assert len(final_state.get("step_records") or []) >= 2
    assert "Average BMI" in (final_state.get("response_text") or "")
    assert final_state.get("prediction_result") is not None
    assert final_state.get("data_result") is not None
    assert "orchestrator:data" in (final_state.get("agent_steps") or [])
    assert "orchestrator:prediction" in (final_state.get("agent_steps") or [])
    assert "synthesis" in (final_state.get("agent_steps") or [])


@patch("agents.graph.run_prediction_agent")
def test_graph_single_agent_finishes_without_synthesize(mock_predict: Any) -> None:
    mock_predict.return_value = {
        "user_message": "Predict ALT for BMI 30",
        "session_id": "sess-1",
        "route": AgentRoute.PREDICTION.value,
        "orchestrator_action": "finish",
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
    assert final_state.get("response_text") == "ALT prediction complete."
