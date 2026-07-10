"""Orchestrator — guardrails + hybrid routing to specialist agents."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.guardrails import check_input_guardrails
from agents.state import AgentState, _merge_state, append_agent_step, get_conversation_history, get_prior_steps, get_session_facts
from memory.context import build_routing_prompt
from agents.subagents.prediction_agent import configure_llm_environment, require_llm_api_key, routing_llm
from config import settings
from schemas.routing import AgentRoute, LLMRoutingExtraction, OrchestratorAction, RoutingDecision

ORCHESTRATOR_SYSTEM_PROMPT = """You route analyst chat messages to the correct specialist agent.

Routes:
- prediction: COPD/ALT ML predictions from patient attributes (BMI, diet, exercise, smoker, etc.)
- data: SQL/analytics questions over the patient CSV dataset (counts, averages, trends, readmissions)
- rag: search clinical documents, guidelines, policies, treatment plans; citation-style questions; what documents recommend or say
- fallback: greetings, chit-chat, unrelated topics, or requests that do not fit the above

Rules:
- Choose exactly one route.
- Dataset analytics (counts, averages, groupings, distributions) -> data even if COPD/ALT column names appear.
- prediction is only for inferring a class/value for a specific patient scenario (usually with the word predict).
- Questions about what clinical documents, guidelines, policies, or treatment plans say/recommend -> rag (not fallback).
- Use confidence >= 0.8 only when intent is clear.
- Set requires_clarification=true when the message is ambiguous and cannot be executed safely.
- Provide a short clarification_prompt when requires_clarification is true.
- Do NOT route clinical diagnosis or treatment advice to prediction; use fallback instead.
"""

_RULE_KEYWORDS: dict[AgentRoute, tuple[str, ...]] = {
    AgentRoute.PREDICTION: (
        "predict",
        "prediction",
        "copd",
        " alt",
        "alt ",
        "bmi",
        "severity class",
        "alanine",
    ),
    AgentRoute.DATA: (
        "sql",
        "query",
        "select ",
        "count ",
        "show the count",
        "for each",
        "how many",
        "average",
        "mean ",
        "sum ",
        "group by",
        "most common",
        "compare ",
        "among ",
        "readmission",
        "dataset",
        "table",
        "duckdb",
        "by month",
        "by diet",
    ),
    AgentRoute.RAG: (
        "document",
        "documents",
        "guideline",
        "guidelines",
        "policy",
        "treatment plan",
        "clinical note",
        "search doc",
        "according to",
        "what does the",
        "what does our",
        "recommended in",
        "mentioned in",
        "summarize",
        "cite",
        "citation",
    ),
}


_ANALYTICS_SIGNALS: tuple[str, ...] = (
    "how many",
    "count ",
    "show the count",
    "for each",
    "average",
    "mean ",
    "sum ",
    "group by",
    "most common",
    "compare ",
    "among ",
    "distribution",
    " per ",
    "breakdown",
)

_PREDICTION_INTENT_SIGNALS: tuple[str, ...] = (
    "predict",
    "prediction",
    "estimate ",
    "forecast",
)

_DOCUMENT_SEARCH_SIGNALS: tuple[str, ...] = (
    "treatment plan",
    "clinical document",
    "clinical documents",
    "document",
    "documents",
    "what documents",
    "in documents",
    "from documents",
    "from the document",
    "guideline",
    "guidelines",
    "policy",
    "according to the",
    "according to our",
    "what does the",
    "what does our",
    "search our",
    "search the",
    "recommended in",
    "mentioned in",
    "summarize",
    "summarise",
    "cite",
    "citation",
)


def _keyword_score(message: str, keywords: tuple[str, ...]) -> int:
    lowered = message.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _explicit_prediction_intent(message: str) -> bool:
    lowered = message.lower()
    return any(signal in lowered for signal in _PREDICTION_INTENT_SIGNALS)


def _analytics_signal_count(message: str) -> int:
    lowered = message.lower()
    return sum(1 for signal in _ANALYTICS_SIGNALS if signal in lowered)


def _document_search_signal_count(message: str) -> int:
    lowered = message.lower()
    return sum(1 for signal in _DOCUMENT_SEARCH_SIGNALS if signal in lowered)


def _looks_like_prediction_follow_up(message: str) -> bool:
    lowered = message.lower()
    if _explicit_prediction_intent(message):
        return True
    follow_up_signals = ("what if", "and if", "instead", "change ", "also ", " too")
    feature_signals = ("bmi", "diet", "exercise", "smoker", "alt", "copd", "readmitted")
    return any(signal in lowered for signal in follow_up_signals) and any(
        signal in lowered for signal in feature_signals
    )


def route_with_rules(
    message: str,
    *,
    last_route: str | None = None,
) -> RoutingDecision | None:
    """Fast deterministic routing for clear keyword matches."""
    if last_route == AgentRoute.PREDICTION.value and _looks_like_prediction_follow_up(message):
        return RoutingDecision(
            route=AgentRoute.PREDICTION,
            confidence=0.78,
            reasoning="Follow-up message continues an in-session prediction conversation.",
            source="rules",
        )

    analytics_hits = _analytics_signal_count(message)
    if analytics_hits > 0 and not _explicit_prediction_intent(message):
        confidence = min(0.95, 0.60 + analytics_hits * 0.10)
        return RoutingDecision(
            route=AgentRoute.DATA,
            confidence=confidence,
            reasoning=(
                f"Detected {analytics_hits} dataset analytics signal(s) "
                "without explicit prediction intent."
            ),
            source="rules",
        )

    document_hits = _document_search_signal_count(message)
    if document_hits > 0 and not _explicit_prediction_intent(message) and analytics_hits == 0:
        confidence = min(0.95, 0.60 + document_hits * 0.10)
        return RoutingDecision(
            route=AgentRoute.RAG,
            confidence=confidence,
            reasoning=(
                f"Detected {document_hits} clinical document search signal(s) "
                "without dataset analytics or prediction intent."
            ),
            source="rules",
        )

    scores = {route: _keyword_score(message, keywords) for route, keywords in _RULE_KEYWORDS.items()}
    best_route = max(scores, key=lambda route: scores[route])
    best_score = scores[best_route]
    if best_score == 0:
        return None

    tied_routes = [route for route, score in scores.items() if score == best_score]
    if len(tied_routes) > 1:
        return None

    confidence = min(0.95, 0.55 + best_score * 0.15)
    return RoutingDecision(
        route=best_route,
        confidence=confidence,
        reasoning=f"Matched {best_score} keyword(s) for {best_route.value}.",
        source="rules",
    )


def route_with_llm(
    message: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    prior_steps: list[dict[str, Any]] | None = None,
) -> RoutingDecision:
    """LLM routing for ambiguous messages."""
    configure_llm_environment()
    llm = routing_llm().with_structured_output(LLMRoutingExtraction)
    prompt = build_routing_prompt(
        message,
        conversation_history=conversation_history,
        prior_steps=prior_steps,
    )
    result = llm.invoke(
        [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    if isinstance(result, LLMRoutingExtraction):
        extraction = result
    else:
        extraction = LLMRoutingExtraction.model_validate(result)

    return RoutingDecision(
        route=extraction.route,
        confidence=extraction.confidence,
        reasoning=extraction.reasoning,
        requires_clarification=extraction.requires_clarification,
        clarification_prompt=extraction.clarification_prompt,
        source="llm",
    )


def decide_route(
    message: str,
    *,
    allow_llm: bool = True,
    conversation_history: list[dict[str, Any]] | None = None,
    prior_steps: list[dict[str, Any]] | None = None,
    last_route: str | None = None,
) -> RoutingDecision:
    """Pick a route using rules first, then optional LLM fallback."""
    ruled = route_with_rules(message, last_route=last_route)
    if ruled is not None and ruled.confidence >= settings.routing_confidence_threshold:
        return ruled

    if allow_llm:
        try:
            require_llm_api_key()
            return route_with_llm(
                message,
                conversation_history=conversation_history,
                prior_steps=prior_steps,
            )
        except ValueError:
            pass

    if ruled is not None:
        return ruled

    return RoutingDecision(
        route=AgentRoute.FALLBACK,
        confidence=0.4,
        reasoning="No clear keyword match and LLM routing unavailable.",
        requires_clarification=True,
        clarification_prompt=(
            "I am not sure what you need. Do you want a COPD/ALT prediction, "
            "a SQL analytics query over the patient dataset, or a document search?"
        ),
        source="rules",
    )


def _apply_low_confidence(decision: RoutingDecision) -> RoutingDecision:
    if decision.confidence >= settings.routing_confidence_threshold:
        return decision
    if decision.requires_clarification:
        return RoutingDecision(
            route=AgentRoute.FALLBACK,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            requires_clarification=True,
            clarification_prompt=decision.clarification_prompt
            or "Could you clarify whether you need a prediction, a dataset query, or a document search?",
            source=decision.source,
        )
    return RoutingDecision(
        route=AgentRoute.FALLBACK,
        confidence=decision.confidence,
        reasoning=f"Low routing confidence ({decision.confidence:.2f}): {decision.reasoning}",
        requires_clarification=True,
        clarification_prompt=(
            "I am not confident about the best agent for this request. "
            "Please rephrase as a prediction, SQL/data question, or document search."
        ),
        source=decision.source,
    )


def routing_to_state_updates(decision: RoutingDecision) -> dict[str, Any]:
    """Map a routing decision onto AgentState fields."""
    return {
        "route": decision.route.value,
        "route_confidence": decision.confidence,
        "route_reasoning": decision.reasoning,
        "route_source": decision.source,
        "requires_clarification": decision.requires_clarification,
        "clarification_prompt": decision.clarification_prompt,
    }


def run_orchestrator(state: AgentState) -> AgentState:
    """
    LangGraph node: guardrails → multi-step plan → write routing fields to state.

    Does not produce user-facing response text; specialist/fallback/synthesize nodes do that.
    """
    from agents.multi_step import count_specialist_steps, plan_next_step, plan_to_state_updates

    started = time.perf_counter()
    user_message = state.get("user_message", "")
    guardrail = check_input_guardrails(user_message)

    if not guardrail.allowed:
        decision = RoutingDecision(
            route=AgentRoute.FALLBACK,
            confidence=1.0,
            reasoning="Input blocked by guardrails.",
            source="guardrail",
        )
        updated = _merge_state(
            state,
            user_message=guardrail.sanitized_message,
            guardrail_blocked=True,
            guardrail_reason=guardrail.blocked_reason,
            orchestrator_action=OrchestratorAction.ROUTE.value,
            **routing_to_state_updates(decision),
        )
        return append_agent_step(updated, f"orchestrator:{decision.route.value}")

    plan = plan_next_step(state)
    updates = plan_to_state_updates(plan)

    if count_specialist_steps(state) == 0 and plan.action == OrchestratorAction.ROUTE and plan.route is not None:
        decision = RoutingDecision(
            route=plan.route,
            confidence=plan.confidence,
            reasoning=plan.reasoning,
            source="rules" if plan.route != AgentRoute.FALLBACK else "multi_step",
        )
        decision = _apply_low_confidence(decision)
        updates.update(routing_to_state_updates(decision))
        if decision.route == AgentRoute.FALLBACK:
            plan = plan.model_copy(
                update={
                    "route": AgentRoute.FALLBACK,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                },
            )
            updates["route"] = AgentRoute.FALLBACK.value
            updates["orchestrator_action"] = OrchestratorAction.ROUTE.value

    updated = _merge_state(
        state,
        user_message=guardrail.sanitized_message,
        guardrail_blocked=False,
        guardrail_reason=None,
        **updates,
    )
    step_label = f"orchestrator:{plan.action.value}"
    if plan.action == OrchestratorAction.ROUTE and plan.route is not None:
        step_label = f"orchestrator:{plan.route.value}"
    updated = append_agent_step(updated, step_label)
    latency = state.get("latency_ms")
    orchestrator_ms = round((time.perf_counter() - started) * 1000, 2)
    if latency is None:
        updated["latency_ms"] = orchestrator_ms
    return updated
