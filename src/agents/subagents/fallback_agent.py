"""Fallback agent — guardrail blocks, low-confidence routing, and general help."""

from __future__ import annotations

import time

from agents.state import AgentState, _merge_state, append_agent_step

GENERAL_HELP_MESSAGE = """I can help with:
- **Predictions** — COPD severity class and ALT lab values from patient attributes
  (e.g. "Predict ALT for a patient with BMI 30")
- **Data analytics** — SQL-style questions over the patient dataset
  (e.g. "How many readmissions by month?")
- **Document search** — clinical documents, treatment plans, guidelines
  (e.g. "What low-impact exercise is recommended in treatment plans?")

This is an internal analytics prototype — not clinical advice."""


def run_fallback_agent(state: AgentState) -> AgentState:
    """LangGraph node: user-facing response when routing cannot proceed safely."""
    started = time.perf_counter()

    if state.get("guardrail_blocked"):
        response_text = state.get("guardrail_reason") or (
            "Your message could not be processed. Please rephrase your analytics request."
        )
    elif state.get("requires_clarification"):
        response_text = state.get("clarification_prompt") or GENERAL_HELP_MESSAGE
    else:
        response_text = GENERAL_HELP_MESSAGE

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    prior_latency = state.get("latency_ms") or 0.0
    updated = _merge_state(
        state,
        response_text=response_text,
        latency_ms=round(prior_latency + latency_ms, 2),
    )
    return append_agent_step(updated, "fallback")
