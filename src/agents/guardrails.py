"""Input guardrails applied before orchestrator routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import settings

_CLINICAL_ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwhat (medication|medicine|drug|treatment) should (i|we|the patient)",
        r"\b(prescribe|prescription|dosage|dose)\b",
        r"\b(is it safe for (me|the patient) to)\b",
        r"\b(diagnose|diagnosis for me|tell me if i have)\b",
    )
)


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of pre-routing input checks."""

    allowed: bool
    sanitized_message: str
    blocked_reason: str | None = None


def sanitize_message(message: str) -> str:
    """Normalize whitespace for downstream routing and agents."""
    return " ".join(message.strip().split())


def check_input_guardrails(message: str) -> GuardrailResult:
    """
    Deterministic input filter run before routing.

    Blocks clinical-advice style requests that are out of scope for an analytics prototype.
    """
    sanitized = sanitize_message(message)
    if not sanitized:
        return GuardrailResult(
            allowed=False,
            sanitized_message="",
            blocked_reason="Message is empty after trimming whitespace.",
        )

    if len(sanitized) > settings.chat_max_message_chars:
        return GuardrailResult(
            allowed=False,
            sanitized_message=sanitized,
            blocked_reason=(
                f"Message exceeds the maximum length of {settings.chat_max_message_chars} characters."
            ),
        )

    for pattern in _CLINICAL_ADVICE_PATTERNS:
        if pattern.search(sanitized):
            return GuardrailResult(
                allowed=False,
                sanitized_message=sanitized,
                blocked_reason=(
                    "Clinical diagnosis or treatment advice is out of scope. "
                    "Ask for analytics, predictions, SQL over the dataset, or document search."
                ),
            )

    return GuardrailResult(allowed=True, sanitized_message=sanitized)
