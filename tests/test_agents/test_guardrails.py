"""Tests for input guardrails."""

from agents.guardrails import check_input_guardrails, sanitize_message


def test_sanitize_message_collapses_whitespace() -> None:
    assert sanitize_message("  Predict   ALT  ") == "Predict ALT"


def test_guardrails_allow_prediction_message() -> None:
    result = check_input_guardrails("Predict ALT for a patient with BMI 30")
    assert result.allowed is True
    assert result.sanitized_message == "Predict ALT for a patient with BMI 30"


def test_guardrails_block_clinical_advice() -> None:
    result = check_input_guardrails("What medication should the patient take for COPD?")
    assert result.allowed is False
    assert result.blocked_reason is not None
    assert "out of scope" in result.blocked_reason.lower()


def test_guardrails_block_overlong_message(monkeypatch) -> None:
    monkeypatch.setattr("agents.guardrails.settings.chat_max_message_chars", 20)
    result = check_input_guardrails("x" * 25)
    assert result.allowed is False
    assert "maximum length" in (result.blocked_reason or "").lower()
