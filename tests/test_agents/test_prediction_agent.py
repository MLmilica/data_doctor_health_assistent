"""Tests for prediction agent (LLM mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agents.state import chat_response_from_state, initial_state_from_chat_request
from agents.subagents.prediction_agent import (
    extract_with_llm,
    format_fallback_response,
    load_top_global_factors,
    run_prediction_agent,
    synthesize_response_text,
)
from config import settings
from schemas.chat import ChatRequest
from schemas.prediction import (
    LLMPredictionExtraction,
    PatientFeatures,
    PredictionResponse,
    PredictionTarget,
)


def test_load_top_global_factors_reads_shap_json(ml_artifacts: None) -> None:
    factors = load_top_global_factors("copd", limit=3)
    assert len(factors) <= 3
    assert all(isinstance(f, str) and f for f in factors)


def test_format_fallback_missing_required() -> None:
    text = format_fallback_response(
        PredictionResponse(
            target="alt",
            prediction=None,
            can_predict=False,
            used_features={},
            defaults_used={},
            missing_required=["bmi"],
        )
    )
    assert "bmi" in text
    assert "Disclaimer" in text or "prototype" in text.lower()


@patch("agents.subagents.prediction_agent.synthesize_response_text")
@patch("agents.subagents.prediction_agent.extract_with_llm")
def test_run_prediction_agent_alt_success(
    mock_extract: Any,
    mock_synthesize: Any,
    ml_artifacts: None,
) -> None:
    mock_extract.return_value = LLMPredictionExtraction(
        is_prediction_request=True,
        target=PredictionTarget.ALT,
        features=PatientFeatures(bmi=30.0),
    )
    mock_synthesize.return_value = "The estimated ALT level is 30.0 U/L."

    state = initial_state_from_chat_request(
        ChatRequest(message="Predict ALT for BMI 30", session_id="s1"),
    )
    result_state = run_prediction_agent(state)

    assert result_state.get("error") is None
    assert result_state.get("response_text") == "The estimated ALT level is 30.0 U/L."
    assert result_state.get("prediction_result") is not None

    chat = chat_response_from_state(result_state)
    assert chat.prediction is not None
    assert chat.prediction.can_predict
    assert isinstance(chat.prediction.prediction, float)


@patch("agents.subagents.prediction_agent.extract_with_llm")
def test_run_prediction_agent_missing_target(mock_extract: Any) -> None:
    mock_extract.return_value = LLMPredictionExtraction(
        is_prediction_request=True,
        target=None,
        assistant_message="Do you want COPD, ALT, or both?",
    )

    state = initial_state_from_chat_request(ChatRequest(message="Predict something for this patient"))
    result_state = run_prediction_agent(state)

    assert "prediction_result" not in result_state
    assert result_state.get("response_text") == "Do you want COPD, ALT, or both?"
    assert "prediction" in (result_state.get("agent_steps") or [])


@patch("agents.subagents.prediction_agent.extract_with_llm")
def test_run_prediction_agent_missing_api_key(
    mock_extract: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = initial_state_from_chat_request(ChatRequest(message="Predict ALT for BMI 30"))
    result_state = run_prediction_agent(state)

    assert result_state.get("error")
    assert "OPENAI_API_KEY" in str(result_state.get("response_text"))
    mock_extract.assert_not_called()


def test_extract_with_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        extract_with_llm("Predict ALT")


def test_synthesize_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        synthesize_response_text({"prediction": 1})
