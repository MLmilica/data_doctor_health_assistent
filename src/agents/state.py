"""LangGraph agent state — shared clipboard for one chat request."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from schemas.chat import ChatRequest, ChatResponse
from schemas.prediction import LLMPredictionExtraction, PredictionResponse


class AgentState(TypedDict, total=False):
    """
    Shared state merged between LangGraph nodes for a single /chat invocation.

    Nodes read what they need and return partial updates (dict merge).
    Pydantic models are stored as plain dicts for LangGraph compatibility.
    """

    # --- Input (FastAPI → graph) ---
    user_message: str
    session_id: str
    user_id: str

    # --- Orchestrator routing ---
    route: str
    route_confidence: float
    route_reasoning: str
    route_source: str
    requires_clarification: bool
    clarification_prompt: str | None
    guardrail_blocked: bool
    guardrail_reason: str | None
    agent_steps: list[str]

    # --- LLM extraction (prediction node) ---
    extraction: dict[str, Any]

    # --- Inference (prediction node) ---
    # Single target: PredictionResponse.model_dump()
    # Both targets: {"copd": {...}, "alt": {...}}
    prediction_result: dict[str, Any]

    # --- Output (prediction node → FastAPI) ---
    response_text: str

    # --- Observability (UI metadata) ---
    llm_model: str | None
    latency_ms: float | None
    top_global_factors: dict[str, list[str]]

    # --- Failure path ---
    error: str


def _merge_state(state: AgentState, **updates: Any) -> AgentState:
    merged = dict(state)
    merged.update(updates)
    return cast(AgentState, merged)


def append_agent_step(state: AgentState, step: str) -> AgentState:
    """Append a completed graph step for observability and future multi-step routing."""
    steps = list(state.get("agent_steps") or [])
    steps.append(step)
    return _merge_state(state, agent_steps=steps)


def initial_state_from_chat_request(request: ChatRequest) -> AgentState:
    """Build the graph input state from an API ChatRequest."""
    return AgentState(
        user_message=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
    )


def get_extraction(state: AgentState) -> LLMPredictionExtraction | None:
    raw = state.get("extraction")
    if raw is None:
        return None
    return LLMPredictionExtraction.model_validate(raw)


def set_extraction(state: AgentState, extraction: LLMPredictionExtraction) -> AgentState:
    return _merge_state(state, extraction=extraction.model_dump())


def get_prediction_result(
    state: AgentState,
) -> PredictionResponse | dict[str, PredictionResponse] | None:
    raw = state.get("prediction_result")
    if raw is None:
        return None
    if "target" in raw:
        return PredictionResponse.model_validate(raw)
    return {key: PredictionResponse.model_validate(value) for key, value in raw.items()}


def set_prediction_result(
    state: AgentState,
    result: PredictionResponse | dict[str, PredictionResponse],
) -> AgentState:
    if isinstance(result, dict):
        payload = {key: value.model_dump() for key, value in result.items()}
    else:
        payload = result.model_dump()
    return _merge_state(state, prediction_result=payload)


def chat_response_from_state(state: AgentState) -> ChatResponse:
    """Map terminal AgentState to the public ChatResponse DTO."""
    session_id = state.get("session_id", "")
    response_text = state.get("response_text", "")
    error = state.get("error")
    if error:
        return ChatResponse(
            text=response_text or error,
            session_id=session_id,
            metadata=_metadata_from_state(state),
        )

    prediction_result = get_prediction_result(state)
    if prediction_result is None:
        return ChatResponse(
            text=response_text,
            session_id=session_id,
            metadata=_metadata_from_state(state),
        )

    response = ChatResponse.from_prediction_results(
        text=response_text,
        session_id=session_id,
        result=prediction_result,
        llm_model=state.get("llm_model"),
        latency_ms=state.get("latency_ms"),
        top_global_factors=state.get("top_global_factors"),
    )
    return response.model_copy(update={"metadata": _metadata_from_state(state)})


def _metadata_from_state(state: AgentState):
    from schemas.chat import ChatAgentMetadata

    routed_to = state.get("route")
    return ChatAgentMetadata(
        agent=routed_to or "orchestrator",
        routed_to=routed_to,
        route_confidence=state.get("route_confidence"),
        route_source=state.get("route_source"),
        guardrail_blocked=bool(state.get("guardrail_blocked")),
        llm_model=state.get("llm_model"),
        latency_ms=state.get("latency_ms"),
    )
