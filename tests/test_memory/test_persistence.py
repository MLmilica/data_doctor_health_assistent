"""Tests for session persistence across /chat invocations."""

from __future__ import annotations

from unittest.mock import patch

from agents.graph import build_graph
from agents.state import initial_state_from_chat_request
from memory.persistence import enrich_state_from_session, persist_chat_turn, run_chat_with_memory
from memory.session_store import InMemorySessionStore
from schemas.chat import ChatRequest, ChatResponse
from schemas.memory import ChatSession, SessionFacts
from schemas.prediction import LLMPredictionExtraction, PatientFeatures, PredictionTarget


def test_enrich_state_from_session_loads_windowed_context() -> None:
    session = ChatSession(session_id="s1", user_id="u1")
    session.facts = SessionFacts(last_target="alt", last_features={"bmi": 28.0})
    state = initial_state_from_chat_request(ChatRequest(message="follow up", session_id="s1"))
    enriched = enrich_state_from_session(state, session)

    assert enriched["session_facts"]["last_target"] == "alt"
    assert enriched["conversation_history"] == []
    assert enriched["step_records"] == []


@patch("agents.subagents.prediction_agent.synthesize_response_text")
@patch("agents.subagents.prediction_agent.extract_with_llm")
def test_run_chat_with_memory_persists_turns_and_facts(
    mock_extract,
    mock_synthesize,
    ml_artifacts: None,
) -> None:
    mock_extract.return_value = LLMPredictionExtraction(
        is_prediction_request=True,
        target=PredictionTarget.ALT,
        features=PatientFeatures(bmi=30.0),
    )
    mock_synthesize.return_value = "ALT is 30."

    store = InMemorySessionStore()
    graph = build_graph()
    request = ChatRequest(message="Predict ALT for BMI 30", session_id="mem-1", user_id="u1")

    response, _ = run_chat_with_memory(graph, request, store=store)
    assert response.prediction is not None

    session = store.get("mem-1", "u1")
    assert session is not None
    assert len(session.turns) == 2
    assert len(session.steps) == 1
    assert session.facts.last_target == "alt"
    assert session.facts.last_features.get("bmi") == 30.0


@patch("agents.subagents.prediction_agent.synthesize_response_text")
@patch("agents.subagents.prediction_agent.extract_with_llm")
def test_follow_up_uses_session_facts_in_state(
    mock_extract,
    mock_synthesize,
    ml_artifacts: None,
) -> None:
    mock_synthesize.return_value = "Updated prediction."

    store = InMemorySessionStore()
    graph = build_graph()

    first = ChatRequest(
        message="Predict ALT for BMI 28",
        session_id="mem-2",
        user_id="u1",
    )
    mock_extract.return_value = LLMPredictionExtraction(
        is_prediction_request=True,
        target=PredictionTarget.ALT,
        features=PatientFeatures(bmi=28.0),
    )
    run_chat_with_memory(graph, first, store=store)

    follow_up = ChatRequest(message="What if BMI is 35?", session_id="mem-2", user_id="u1")
    mock_extract.return_value = LLMPredictionExtraction(
        is_prediction_request=True,
        target=None,
        features=PatientFeatures(bmi=35.0),
    )
    run_chat_with_memory(graph, follow_up, store=store)

    session = store.get("mem-2", "u1")
    assert session is not None
    assert session.facts.last_features.get("bmi") == 35.0
    assert session.facts.last_target == "alt"
    assert len(session.turns) == 4


def test_persist_chat_turn_appends_transcript() -> None:
    session = ChatSession(session_id="s1", user_id="u1")
    state = initial_state_from_chat_request(ChatRequest(message="hi", session_id="s1"))
    state["route"] = "fallback"
    state["response_text"] = "Hello"

    response = ChatResponse(text="Hello", session_id="s1")
    updated = persist_chat_turn(
        session,
        request=ChatRequest(message="hi", session_id="s1"),
        final_state=state,
        response=response,
    )
    assert len(updated.turns) == 2
    assert updated.steps[0].agent == "fallback"
