"""Memory helpers — history windowing, feature merge, prompt formatting."""

from __future__ import annotations

import json
from typing import Any

from schemas.memory import ChatTurn, SessionFacts, StepRecord
from schemas.prediction import PatientFeatures


def window_turns(turns: list[ChatTurn], *, limit: int) -> list[ChatTurn]:
    if limit <= 0:
        return []
    return turns[-limit:]


def window_steps(steps: list[StepRecord], *, limit: int) -> list[StepRecord]:
    if limit <= 0:
        return []
    return steps[-limit:]


def turns_to_dicts(turns: list[ChatTurn]) -> list[dict[str, Any]]:
    return [turn.model_dump(mode="json") for turn in turns]


def steps_to_dicts(steps: list[StepRecord]) -> list[dict[str, Any]]:
    return [step.model_dump(mode="json") for step in steps]


def format_conversation_for_prompt(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return ""
    lines: list[str] = []
    for turn in turns:
        role = turn.get("role", "user")
        label = "User" if role == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


def format_prior_steps_for_prompt(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    lines: list[str] = []
    for step in steps:
        agent = step.get("agent", "unknown")
        tool = step.get("tool")
        summary = step.get("assistant_summary") or ""
        label = f"{agent}/{tool}" if tool else agent
        if summary:
            lines.append(f"- {label}: {summary}")
    return "\n".join(lines)


def merge_patient_features(
    base: dict[str, Any] | None,
    delta: PatientFeatures,
) -> PatientFeatures:
    """Merge session features with newly extracted values (delta overrides base)."""
    merged: dict[str, Any] = dict(base or {})
    for key, value in delta.model_dump().items():
        if value is not None:
            merged[key] = value
    return PatientFeatures.model_validate(merged)


def build_prediction_extraction_prompt(
    user_message: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    session_facts: dict[str, Any] | None = None,
) -> str:
    """Build LLM input for prediction extraction with session context."""
    sections: list[str] = []
    history_text = format_conversation_for_prompt(conversation_history or [])
    if history_text:
        sections.append(f"Previous conversation:\n{history_text}")

    facts = SessionFacts.model_validate(session_facts or {})
    if facts.last_features or facts.last_target or facts.missing_required:
        sections.append(
            "Session context (carry forward unless the user changes these):\n"
            + json.dumps(
                {
                    "last_target": facts.last_target,
                    "last_features": facts.last_features,
                    "missing_required": facts.missing_required,
                },
                indent=2,
            )
        )

    sections.append(f"Current user message:\n{user_message.strip()}")
    return "\n\n".join(sections)


def build_routing_prompt(
    user_message: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    prior_steps: list[dict[str, Any]] | None = None,
) -> str:
    """Build LLM input for orchestrator routing with session context."""
    sections: list[str] = []
    history_text = format_conversation_for_prompt(conversation_history or [])
    if history_text:
        sections.append(f"Previous conversation:\n{history_text}")

    steps_text = format_prior_steps_for_prompt(prior_steps or [])
    if steps_text:
        sections.append(f"Recent backend steps in this session:\n{steps_text}")

    sections.append(f"Current user message:\n{user_message.strip()}")
    return "\n\n".join(sections)
