"""Tests for memory schemas."""

from schemas.memory import ChatSession, ChatTurn, SessionFacts, StepRecord


def test_chat_session_defaults() -> None:
    session = ChatSession(session_id="s1", user_id="u1")
    assert session.turns == []
    assert session.steps == []
    assert session.facts.last_features == {}


def test_step_record_compact_artifact() -> None:
    record = StepRecord(
        agent="prediction",
        artifact={"target": "alt", "prediction": 30.0},
        assistant_summary="ALT predicted.",
    )
    assert record.agent == "prediction"
    assert record.tool is None
