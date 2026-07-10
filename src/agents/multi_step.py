"""Multi-step orchestration planning — required agents and next actions."""

from __future__ import annotations

from typing import Any

from agents.orchestrator import (
    _analytics_signal_count,
    _clinical_knowledge_signal_count,
    _document_search_signal_count,
    _explicit_prediction_intent,
    _looks_like_prediction_follow_up,
    _model_insight_signal_count,
    decide_route,
)
from agents.state import AgentState, get_conversation_history, get_prior_steps, get_session_facts
from config import settings
from memory.context import format_prior_steps_for_prompt
from schemas.routing import (
    SPECIALIST_AGENT_ROUTES,
    AgentRoute,
    LLMMultiStepPlan,
    OrchestratorAction,
    RoutingDecision,
)

_AGENT_PRIORITY: tuple[AgentRoute, ...] = (
    AgentRoute.DATA,
    AgentRoute.PREDICTION,
    AgentRoute.RAG,
)


def count_specialist_steps(state: AgentState) -> int:
    records = state.get("step_records") or []
    return sum(1 for record in records if record.get("agent") in {route.value for route in SPECIALIST_AGENT_ROUTES})


def completed_specialist_agents(state: AgentState) -> set[AgentRoute]:
    completed: set[AgentRoute] = set()
    for record in state.get("step_records") or []:
        agent = record.get("agent")
        try:
            route = AgentRoute(agent)
        except ValueError:
            continue
        if route in SPECIALIST_AGENT_ROUTES:
            completed.add(route)
    if not completed:
        for step in state.get("agent_steps") or []:
            if step in {route.value for route in SPECIALIST_AGENT_ROUTES}:
                completed.add(AgentRoute(step))
    return completed


def detect_required_agents(message: str) -> set[AgentRoute]:
    """Rule-based estimate of which specialist agents the message needs."""
    lowered = message.lower()
    required: set[AgentRoute] = set()

    if _analytics_signal_count(message) > 0:
        required.add(AgentRoute.DATA)
    insight_hits = _model_insight_signal_count(message)
    if insight_hits > 0:
        required.add(AgentRoute.DATA)
    if _explicit_prediction_intent(message) or _looks_like_prediction_follow_up(message):
        required.add(AgentRoute.PREDICTION)
    elif "bmi" in lowered and ("predict" in lowered or "alt" in lowered or "copd" in lowered):
        required.add(AgentRoute.PREDICTION)
    if _document_search_signal_count(message) > 0:
        required.add(AgentRoute.RAG)
    elif _clinical_knowledge_signal_count(message) > 0 and insight_hits == 0:
        required.add(AgentRoute.RAG)

    return required


def _next_agent_in_priority(remaining: set[AgentRoute]) -> AgentRoute:
    for route in _AGENT_PRIORITY:
        if route in remaining:
            return route
    return next(iter(remaining))


def is_multi_part_request(message: str) -> bool:
    """True when the message needs more than one specialist agent."""
    return len(detect_required_agents(message)) > 1


def agent_requested_clarification(state: AgentState) -> bool:
    """True when a specialist agent paused the run for a user follow-up."""
    return bool(state.get("requires_clarification"))


def plan_next_step(state: AgentState, *, allow_llm: bool = True) -> LLMMultiStepPlan:
    """Decide whether to route, synthesize, or finish in the orchestrator loop."""
    message = state.get("user_message", "")
    completed = completed_specialist_agents(state)
    specialist_count = len(completed)

    if agent_requested_clarification(state):
        return LLMMultiStepPlan(
            action=OrchestratorAction.FINISH,
            reasoning="Specialist agent requires user clarification before continuing.",
        )

    if specialist_count >= settings.orchestrator_max_agent_steps:
        if specialist_count >= 2:
            return LLMMultiStepPlan(
                action=OrchestratorAction.SYNTHESIZE,
                reasoning=f"Reached max specialist steps ({settings.orchestrator_max_agent_steps}).",
            )
        return LLMMultiStepPlan(
            action=OrchestratorAction.FINISH,
            reasoning=f"Reached max specialist steps ({settings.orchestrator_max_agent_steps}) with one agent.",
        )

    required = detect_required_agents(message)
    if specialist_count == 0 and not required:
        decision = decide_route(
            message,
            allow_llm=allow_llm,
            conversation_history=get_conversation_history(state),
            prior_steps=get_prior_steps(state),
            last_route=get_session_facts(state).last_route,
        )
        if decision.route == AgentRoute.FALLBACK:
            return LLMMultiStepPlan(
                action=OrchestratorAction.ROUTE,
                route=AgentRoute.FALLBACK,
                reasoning=decision.reasoning,
                confidence=decision.confidence,
            )
        return LLMMultiStepPlan(
            action=OrchestratorAction.ROUTE,
            route=decision.route,
            reasoning=decision.reasoning,
            confidence=decision.confidence,
        )

    if specialist_count == 0 and required:
        first_route = _next_agent_in_priority(required)
        return LLMMultiStepPlan(
            action=OrchestratorAction.ROUTE,
            route=first_route,
            reasoning=f"Multi-part request; starting with {first_route.value}.",
            confidence=0.85,
        )

    remaining = required - completed
    if not remaining:
        if specialist_count >= 2:
            return LLMMultiStepPlan(
                action=OrchestratorAction.SYNTHESIZE,
                reasoning="All required specialist agents have completed.",
            )
        return LLMMultiStepPlan(
            action=OrchestratorAction.FINISH,
            reasoning="Single specialist agent completed the request.",
        )

    if allow_llm:
        llm_plan = _plan_with_llm(state, remaining=remaining)
        if llm_plan is not None:
            return llm_plan

    next_route = _next_agent_in_priority(remaining)
    return LLMMultiStepPlan(
        action=OrchestratorAction.ROUTE,
        route=next_route,
        reasoning=f"Continuing multi-step request with {next_route.value}.",
        confidence=0.8,
    )


def _plan_with_llm(state: AgentState, *, remaining: set[AgentRoute]) -> LLMMultiStepPlan | None:
    try:
        from agents.subagents.prediction_agent import configure_llm_environment, require_llm_api_key, routing_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        require_llm_api_key()
        configure_llm_environment()
    except ValueError:
        return None

    completed = completed_specialist_agents(state)
    step_lines = format_prior_steps_for_prompt(state.get("step_records") or [])
    remaining_text = ", ".join(sorted(route.value for route in remaining))
    completed_text = ", ".join(sorted(route.value for route in completed)) or "none"

    prompt = (
        f"User message:\n{state.get('user_message', '').strip()}\n\n"
        f"Completed specialist agents this turn: {completed_text}\n"
        f"Remaining required agents: {remaining_text}\n\n"
        f"Step summaries this turn:\n{step_lines or '(none)'}"
    )
    system = """You plan the next step in a multi-agent clinical analytics workflow.

Actions:
- route: call one specialist agent (prediction, data, or rag)
- synthesize: all required work is done and at least two agents already ran
- finish: exactly one agent ran and that is sufficient

Rules:
- Choose route only for an agent listed in Remaining required agents.
- Use synthesize when multiple parts of the question are answered.
- Use finish only when a single-agent answer is enough.
- Never choose fallback here.
"""

    llm = routing_llm().with_structured_output(LLMMultiStepPlan)
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    except Exception:
        return None

    if isinstance(result, LLMMultiStepPlan):
        plan = result
    else:
        plan = LLMMultiStepPlan.model_validate(result)

    if plan.action == OrchestratorAction.ROUTE:
        if plan.route is None or plan.route not in remaining:
            return None
    return plan


def plan_to_state_updates(plan: LLMMultiStepPlan) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "orchestrator_action": plan.action.value,
        "route_reasoning": plan.reasoning,
        "route_confidence": plan.confidence,
        "route_source": "multi_step",
    }
    if plan.action == OrchestratorAction.ROUTE and plan.route is not None:
        updates["route"] = plan.route.value
    return updates


def routing_decision_from_plan(plan: LLMMultiStepPlan) -> RoutingDecision | None:
    if plan.action != OrchestratorAction.ROUTE or plan.route is None:
        return None
    return RoutingDecision(
        route=plan.route,
        confidence=plan.confidence,
        reasoning=plan.reasoning,
        source="rules",
    )
