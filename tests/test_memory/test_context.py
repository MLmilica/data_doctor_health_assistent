"""Tests for memory context helpers."""

from memory.context import (
    build_prediction_extraction_prompt,
    format_conversation_for_prompt,
    merge_patient_features,
    window_turns,
)
from schemas.memory import ChatTurn
from schemas.prediction import PatientFeatures


def test_merge_patient_features_delta_overrides_base() -> None:
    base = {"bmi": 28.0, "diet_quality": "Good"}
    delta = PatientFeatures(bmi=35.0)
    merged = merge_patient_features(base, delta)
    assert merged.bmi == 35.0
    assert merged.diet_quality == "Good"


def test_format_conversation_for_prompt() -> None:
    turns = [
        {"role": "user", "content": "Predict COPD"},
        {"role": "assistant", "content": "Need exercise frequency."},
    ]
    text = format_conversation_for_prompt(turns)
    assert "User: Predict COPD" in text
    assert "Assistant: Need exercise frequency." in text


def test_build_prediction_extraction_prompt_includes_session_facts() -> None:
    prompt = build_prediction_extraction_prompt(
        "What if BMI is 35?",
        conversation_history=[{"role": "user", "content": "Predict ALT for BMI 28"}],
        session_facts={
            "last_target": "alt",
            "last_features": {"bmi": 28.0},
            "missing_required": [],
        },
    )
    assert "Previous conversation" in prompt
    assert "Session context" in prompt
    assert "What if BMI is 35?" in prompt


def test_window_turns_limits_history() -> None:
    turns = [ChatTurn(role="user", content=str(i)) for i in range(5)]
    windowed = window_turns(turns, limit=2)
    assert len(windowed) == 2
    assert windowed[0].content == "3"
