"""Persist chat turns and step records after each /chat invocation."""

from __future__ import annotations

from typing import Any

from agents.state import (
    AgentState,
    append_step_record,
    get_chart_result,
    get_data_result,
    get_insight_result,
    get_prediction_result,
    get_rag_result,
)
from config import settings
from memory.context import steps_to_dicts, turns_to_dicts, window_steps, window_turns
from memory.session_store import InMemorySessionStore, get_session_store
from schemas.chat import ChatRequest, ChatResponse
from schemas.memory import ChatSession, ChatTurn, SessionFacts, StepRecord, utc_now
from schemas.prediction import PredictionResponse


def enrich_state_from_session(state: AgentState, session: ChatSession) -> AgentState:
    """Load windowed session context into AgentState before graph.invoke."""
    merged = dict(state)
    merged["conversation_history"] = turns_to_dicts(
        window_turns(session.turns, limit=settings.memory_max_turns),
    )
    merged["session_facts"] = session.facts.model_dump()
    merged["prior_steps"] = steps_to_dicts(
        window_steps(session.steps, limit=settings.memory_max_prior_steps),
    )
    merged["step_records"] = []
    return merged  # type: ignore[return-value]


def _sample_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows[:limit]


def _prediction_artifact(result: PredictionResponse | dict[str, PredictionResponse]) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            "target": "both",
            "predictions": {
                key: {
                    "prediction": value.prediction,
                    "can_predict": value.can_predict,
                    "used_features": value.used_features,
                    "missing_required": value.missing_required,
                    "defaults_used": value.defaults_used,
                }
                for key, value in result.items()
            },
        }
    return {
        "target": result.target,
        "prediction": result.prediction,
        "can_predict": result.can_predict,
        "used_features": result.used_features,
        "missing_required": result.missing_required,
        "defaults_used": result.defaults_used,
    }


def build_step_record(state: AgentState) -> StepRecord:
    """Build a compact step record from the terminal agent state."""
    route = state.get("route") or "fallback"
    response_text = state.get("response_text", "")
    latency_ms = state.get("latency_ms")
    error = state.get("error")
    status: str = "error" if error else "ok"

    artifact: dict[str, Any] = {}
    tool: str | None = None

    if route == "prediction":
        result = get_prediction_result(state)
        if result is not None:
            artifact = _prediction_artifact(result)
            if isinstance(result, PredictionResponse) and result.missing_required:
                status = "clarification"
        elif state.get("requires_clarification"):
            status = "clarification"
    elif route == "data":
        chart_result = get_chart_result(state)
        insight_result = get_insight_result(state)
        if chart_result is not None:
            tool = "chart"
            artifact = {
                "chart_type": chart_result.spec.chart_type,
                "title": chart_result.spec.title,
                "x_column": chart_result.spec.x_column,
                "y_column": chart_result.spec.y_column,
                "sql": chart_result.spec.sql,
                "row_count": chart_result.row_count,
            }
        elif insight_result is not None:
            tool = "insight"
            artifact = {
                "target": insight_result.target,
                "source": insight_result.source,
                "top_features": insight_result.top_features[:5],
            }
        else:
            tool = "sql"
            data_result = get_data_result(state)
            if data_result is not None:
                artifact = {
                    "sql": data_result.sql,
                    "columns": data_result.columns,
                    "row_count": data_result.row_count,
                    "truncated": data_result.truncated,
                    "sample_rows": _sample_rows(
                        data_result.rows,
                        settings.memory_sql_sample_rows,
                    ),
                }
            elif state.get("requires_clarification"):
                status = "clarification"
    elif route == "rag":
        rag_result = get_rag_result(state)
        if rag_result is not None:
            artifact = {
                "retrieved_count": rag_result.retrieved_count,
                "relevant_count": rag_result.relevant_count,
                "grounded": rag_result.grounded,
                "citations": [
                    {
                        "source_file": citation.source_file,
                        "section_name": citation.section_name,
                        "snippet": citation.snippet[:200],
                    }
                    for citation in rag_result.citations
                ],
            }
            if rag_result.relevant_count == 0:
                status = "clarification"
    elif route == "fallback":
        if state.get("guardrail_blocked") or state.get("requires_clarification"):
            status = "clarification"

    summary = response_text.strip()
    if len(summary) > 300:
        summary = summary[:297] + "..."

    return StepRecord(
        agent=route,
        tool=tool,
        status=status,  # type: ignore[arg-type]
        artifact=artifact,
        assistant_summary=summary,
        latency_ms=latency_ms,
    )


def append_run_step_record(state: AgentState) -> AgentState:
    """Append a step ledger record for the current specialist route in this run."""
    route = state.get("route")
    if not route or route in {"fallback", "multi", "synthesis"}:
        return state
    return append_step_record(state, build_step_record(state))


def update_session_facts(state: AgentState, facts: SessionFacts) -> SessionFacts:
    """Update session facts from the terminal graph state."""
    route = state.get("route")
    if route:
        facts.last_route = route

    prediction_result = get_prediction_result(state)
    if isinstance(prediction_result, PredictionResponse):
        facts.last_target = prediction_result.target
        facts.last_features = dict(prediction_result.used_features)
        facts.missing_required = list(prediction_result.missing_required)
    elif isinstance(prediction_result, dict):
        facts.last_target = "both"
        merged_features: dict[str, Any] = {}
        merged_missing: list[str] = []
        for single in prediction_result.values():
            merged_features.update(single.used_features)
            merged_missing.extend(single.missing_required)
        facts.last_features = merged_features
        facts.missing_required = sorted(set(merged_missing))

    data_result = get_data_result(state)
    if data_result is not None:
        facts.last_sql = data_result.sql

    return facts


def persist_chat_turn(
    session: ChatSession,
    *,
    request: ChatRequest,
    final_state: AgentState,
    response: ChatResponse,
) -> ChatSession:
    """Append transcript + step ledger and update session facts."""
    session.turns.append(
        ChatTurn(role="user", content=request.message),
    )
    session.turns.append(
        ChatTurn(
            role="assistant",
            content=response.text,
            routed_to=response.metadata.routed_to,
        ),
    )
    for record_dict in final_state.get("step_records") or []:
        session.steps.append(StepRecord.model_validate(record_dict))
    if not final_state.get("step_records"):
        session.steps.append(build_step_record(final_state))
    session.facts = update_session_facts(final_state, session.facts)
    session.updated_at = utc_now()
    return session


def run_chat_with_memory(
    graph: Any,
    request: ChatRequest,
    *,
    store: InMemorySessionStore | None = None,
) -> tuple[ChatResponse, AgentState]:
    """Load session, invoke graph, persist turn — used by invoke_chat_graph."""
    from agents.state import chat_response_from_state, initial_state_from_chat_request

    session_store = store or get_session_store()
    session = session_store.get_or_create(request.session_id, request.user_id)

    initial_state = initial_state_from_chat_request(request)
    initial_state = enrich_state_from_session(initial_state, session)

    final_state = graph.invoke(initial_state)
    response = chat_response_from_state(final_state)

    session = persist_chat_turn(
        session,
        request=request,
        final_state=final_state,
        response=response,
    )
    session_store.save(session)
    return response, final_state
