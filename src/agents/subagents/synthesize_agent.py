"""Synthesize agent — combine multi-step specialist results into one answer."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState, _merge_state, append_agent_step, append_step_record
from agents.subagents.prediction_agent import (
    configure_llm_environment,
    require_llm_api_key,
    synthesis_llm,
)
from schemas.memory import StepRecord
from schemas.rag import RAG_DISCLAIMER
from schemas.sql import DATA_QUERY_DISCLAIMER

SYNTHESIS_SYSTEM_PROMPT = """You synthesize clinical analytics prototype answers from multiple completed agent steps.

Rules:
- Use ONLY facts in the provided JSON step summaries and artifacts.
- Do not invent numbers, predictions, SQL results, or document claims.
- Combine the findings into one coherent answer for the user's original question.
- Mention each data source naturally (dataset query, prediction, documents).
- If a step failed or returned no data, say so briefly.
- End with the disclaimer field verbatim from the JSON.
- Be concise (3-6 short paragraphs or bullets).
"""


def _build_synthesis_facts(state: AgentState) -> dict[str, Any]:
    step_records = state.get("step_records") or []
    return {
        "user_message": state.get("user_message", ""),
        "steps": step_records,
        "disclaimer": (
            "Internal prototype output — combined from specialist agent steps. Not clinical advice."
        ),
    }


def synthesize_multi_step_response(facts: dict[str, Any]) -> str:
    configure_llm_environment()
    llm = synthesis_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(facts, indent=2)),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _fallback_synthesis_text(state: AgentState) -> str:
    lines = ["Combined results from this request:"]
    for record in state.get("step_records") or []:
        agent = record.get("agent", "unknown")
        summary = record.get("assistant_summary") or ""
        lines.append(f"- {agent}: {summary}")
    lines.append(DATA_QUERY_DISCLAIMER)
    lines.append(RAG_DISCLAIMER)
    return "\n".join(lines)


def run_synthesize_agent(state: AgentState) -> AgentState:
    """LangGraph node: merge step_records into one user-facing response."""
    started = time.perf_counter()
    facts = _build_synthesis_facts(state)

    try:
        require_llm_api_key()
        response_text = synthesize_multi_step_response(facts)
    except Exception:
        response_text = _fallback_synthesis_text(state)

    record = StepRecord(
        agent="synthesis",
        status="ok",
        artifact={"step_count": len(state.get("step_records") or [])},
        assistant_summary=response_text[:300],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    updated = _merge_state(
        state,
        response_text=response_text,
        orchestrator_action="finish",
        route="multi",
        route_reasoning="Synthesized multi-step specialist results.",
        route_source="multi_step",
        llm_model=state.get("llm_model"),
        latency_ms=round((state.get("latency_ms") or 0.0) + (time.perf_counter() - started) * 1000, 2),
    )
    updated = append_step_record(updated, record)
    return append_agent_step(updated, "synthesis")
