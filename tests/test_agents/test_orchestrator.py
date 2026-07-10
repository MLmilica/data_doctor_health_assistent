"""Tests for orchestrator routing."""

from __future__ import annotations

from unittest.mock import patch

from agents.orchestrator import decide_route, route_with_rules, run_orchestrator
from agents.state import initial_state_from_chat_request
from schemas.chat import ChatRequest
from schemas.routing import AgentRoute


def test_route_with_rules_prediction() -> None:
    decision = route_with_rules("Predict ALT for a patient with BMI 30")
    assert decision is not None
    assert decision.route == AgentRoute.PREDICTION
    assert decision.confidence >= 0.6


def test_route_with_rules_data() -> None:
    decision = route_with_rules("Show me a SQL query for readmissions by month")
    assert decision is not None
    assert decision.route == AgentRoute.DATA


def test_route_with_rules_rag() -> None:
    decision = route_with_rules("What does the COPD guideline say about exercise?")
    assert decision is not None
    assert decision.route == AgentRoute.RAG


def test_route_with_rules_rag_treatment_plan_exercise() -> None:
    decision = route_with_rules(
        "What low-impact exercise is recommended in treatment plans?",
    )
    assert decision is not None
    assert decision.route == AgentRoute.RAG
    assert decision.source == "rules"


def test_route_with_rules_prediction_follow_up() -> None:
    decision = route_with_rules(
        "What if BMI is 35?",
        last_route=AgentRoute.PREDICTION.value,
    )
    assert decision is not None
    assert decision.route == AgentRoute.PREDICTION


def test_route_with_rules_ambiguous_returns_none() -> None:
    assert route_with_rules("hello there") is None


@patch("agents.orchestrator.route_with_llm")
def test_decide_route_uses_rules_before_llm(mock_llm) -> None:
    decision = decide_route("Predict COPD for smoker with poor diet", allow_llm=True)
    assert decision.route == AgentRoute.PREDICTION
    mock_llm.assert_not_called()


@patch("agents.orchestrator.route_with_llm")
def test_decide_route_falls_back_without_llm(mock_llm) -> None:
    decision = decide_route("hello there", allow_llm=False)
    assert decision.route == AgentRoute.FALLBACK
    assert decision.requires_clarification is True
    mock_llm.assert_not_called()


def test_run_orchestrator_guardrail_block() -> None:
    state = initial_state_from_chat_request(
        ChatRequest(message="What medication should the patient take?", session_id="s1"),
    )
    updated = run_orchestrator(state)
    assert updated.get("route") == AgentRoute.FALLBACK.value
    assert updated.get("guardrail_blocked") is True
    assert "orchestrator:fallback" in updated.get("agent_steps", [])


def test_run_orchestrator_routes_sql_to_data() -> None:
    state = initial_state_from_chat_request(
        ChatRequest(message="How many readmissions by month?", session_id="s2"),
    )
    updated = run_orchestrator(state)
    assert updated.get("route") == AgentRoute.DATA.value
    assert updated.get("guardrail_blocked") is False


def test_route_with_rules_copd_analytics_goes_to_data() -> None:
    decision = route_with_rules(
        "For each exercise frequency level, show the count of patients in each COPD severity class.",
    )
    assert decision is not None
    assert decision.route == AgentRoute.DATA
    assert decision.source == "rules"
