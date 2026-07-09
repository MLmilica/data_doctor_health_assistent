"""Data Agent — SQL analytics over patient_data.csv (stub until DuckDB layer ships)."""

from __future__ import annotations

import time

from agents.state import AgentState, _merge_state, append_agent_step

DATA_AGENT_STUB_MESSAGE = """The **data agent** (SQL analytics over `patient_data.csv`) is being wired up next.

Your message was routed here correctly. Soon you will be able to ask questions like:
- "How many patients are in each income bracket?"
- "What is the average BMI by diet quality?"
- "Show readmissions grouped by month"

For now, use the **Form** tab for direct ML predictions, or ask a **prediction** question in chat."""


def run_data_agent(state: AgentState) -> AgentState:
    """LangGraph node: placeholder until the shared SQL layer is implemented."""
    started = time.perf_counter()
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    prior_latency = state.get("latency_ms") or 0.0
    updated = _merge_state(
        state,
        response_text=DATA_AGENT_STUB_MESSAGE,
        latency_ms=round(prior_latency + latency_ms, 2),
    )
    return append_agent_step(updated, "data")
