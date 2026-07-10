"""Tests for multi-step orchestration planning."""

from __future__ import annotations

from agents.multi_step import (
    completed_specialist_agents,
    detect_required_agents,
    is_multi_part_request,
    plan_next_step,
)
from agents.state import AgentState
from schemas.routing import AgentRoute, OrchestratorAction


def test_detect_required_agents_copd_risk_factors_data_only() -> None:
    required = detect_required_agents("What are the main risk factors for COPD?")
    assert required == {AgentRoute.DATA}


def test_plan_next_step_routes_copd_risk_factors_to_data() -> None:
    state = AgentState(user_message="What are the main risk factors for COPD?")
    plan = plan_next_step(state, allow_llm=False)
    assert plan.action == OrchestratorAction.ROUTE
    assert plan.route == AgentRoute.DATA


def test_detect_required_agents_combo_message() -> None:
    message = (
        "Compare average BMI in the dataset with ALT prediction for BMI 30 "
        "and what documents say about exercise"
    )
    required = detect_required_agents(message)
    assert AgentRoute.DATA in required
    assert AgentRoute.PREDICTION in required
    assert AgentRoute.RAG in required


def test_plan_next_step_starts_with_data_for_combo() -> None:
    state = AgentState(
        user_message=(
            "Compare average BMI in the dataset with ALT prediction for BMI 30"
        ),
    )
    plan = plan_next_step(state, allow_llm=False)
    assert plan.action == OrchestratorAction.ROUTE
    assert plan.route == AgentRoute.DATA


def test_plan_next_step_finishes_after_single_agent() -> None:
    state = AgentState(
        user_message="How many smokers?",
        agent_steps=["orchestrator:data", "data"],
        step_records=[
            {
                "agent": "data",
                "status": "ok",
                "artifact": {"row_count": 1},
                "assistant_summary": "One smoker group.",
            }
        ],
    )
    plan = plan_next_step(state, allow_llm=False)
    assert plan.action == OrchestratorAction.FINISH


def test_plan_next_step_synthesizes_after_two_agents() -> None:
    state = AgentState(
        user_message=(
            "Compare average BMI in the dataset with ALT prediction for BMI 30"
        ),
        step_records=[
            {"agent": "data", "status": "ok", "artifact": {}, "assistant_summary": "avg bmi"},
            {
                "agent": "prediction",
                "status": "ok",
                "artifact": {},
                "assistant_summary": "alt prediction",
            },
        ],
    )
    plan = plan_next_step(state, allow_llm=False)
    assert plan.action == OrchestratorAction.SYNTHESIZE


def test_completed_specialist_agents_falls_back_to_agent_steps() -> None:
    state = AgentState(agent_steps=["orchestrator:prediction", "prediction"])
    completed = completed_specialist_agents(state)
    assert completed == {AgentRoute.PREDICTION}


def test_is_multi_part_request_detects_data_and_rag() -> None:
    message = (
        "What is the average BMI in the dataset and what do documents "
        "recommend for low-impact exercise?"
    )
    assert is_multi_part_request(message) is True
    assert is_multi_part_request("How many smokers?") is False


def test_plan_next_step_finishes_when_agent_needs_clarification() -> None:
    state = AgentState(
        user_message=(
            "What is the average BMI in the dataset and what do documents "
            "recommend for low-impact exercise?"
        ),
        requires_clarification=True,
        clarification_prompt="Which grouping should I use?",
        step_records=[
            {
                "agent": "data",
                "status": "clarification",
                "artifact": {},
                "assistant_summary": "Which grouping should I use?",
            }
        ],
    )
    plan = plan_next_step(state, allow_llm=False)
    assert plan.action == OrchestratorAction.FINISH
