"""RAG Agent — clinical document search (stub until vectorstore ships)."""

from __future__ import annotations

import time

from agents.state import AgentState, _merge_state, append_agent_step

RAG_AGENT_STUB_MESSAGE = """The **RAG agent** (search over clinical documents in `data/documents/`) is being wired up next.

Your message was routed here correctly. Soon you will be able to ask questions like:
- "What does the COPD guideline say about exercise?"
- "Search our policy on readmission follow-up"
- "Summarize the ALT monitoring document with citations"

For now, try a **prediction** or **data/SQL** question, or use the **Form** tab for direct ML inference."""


def run_rag_agent(state: AgentState) -> AgentState:
    """LangGraph node: placeholder until Chroma indexing is implemented."""
    started = time.perf_counter()
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    prior_latency = state.get("latency_ms") or 0.0
    updated = _merge_state(
        state,
        response_text=RAG_AGENT_STUB_MESSAGE,
        latency_ms=round(prior_latency + latency_ms, 2),
    )
    return append_agent_step(updated, "rag")
