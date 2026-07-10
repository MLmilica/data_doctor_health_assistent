"""Tests for the RAG agent (retrieval and LLM steps mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agents.state import chat_response_from_state, initial_state_from_chat_request
from agents.subagents.rag_agent import run_rag_agent
from agents.tools.rag_retrieval import reset_vectorstore
from data.vectorstore import DocumentVectorStore, build_test_embedding_function
from schemas.chat import ChatRequest
from schemas.citation import RetrievedChunk
from schemas.rag import LLMGroundingCheck


@pytest.fixture
def indexed_vectorstore(tmp_path, sample_documents_dir):
    reset_vectorstore()
    store = DocumentVectorStore(
        chroma_dir=tmp_path / "chroma",
        documents_dir=sample_documents_dir,
        collection_name="test_rag_agent",
        embedding_function=build_test_embedding_function(),
    )
    store.index_documents()
    with patch("agents.subagents.rag_agent.get_vectorstore", return_value=store):
        yield store
    reset_vectorstore()


@pytest.fixture
def sample_documents_dir(tmp_path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "copd_guideline.md").write_text(
        """# COPD Guideline

## Exercise
Patients with COPD should perform regular low-intensity exercise.
""",
        encoding="utf-8",
    )
    return docs_dir


def _exercise_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="copd_guideline.md::exercise",
        source_file="copd_guideline.md",
        section_name="Exercise",
        content="Patients with COPD should perform regular low-intensity exercise.",
        score=0.92,
        relevant=True,
        relevance_reason="Directly answers the exercise question.",
    )


@patch("agents.subagents.rag_agent.verify_grounding")
@patch("agents.subagents.rag_agent.synthesize_rag_answer")
@patch("agents.subagents.rag_agent.grade_chunks")
@patch("agents.subagents.rag_agent.retrieve_chunks_for_query")
def test_run_rag_agent_returns_grounded_answer(
    mock_retrieve: Any,
    mock_grade: Any,
    mock_synthesize: Any,
    mock_verify: Any,
    indexed_vectorstore: DocumentVectorStore,
) -> None:
    chunk = _exercise_chunk()
    mock_retrieve.return_value = [chunk]
    mock_grade.return_value = [chunk]
    mock_synthesize.return_value = (
        "The COPD guideline recommends regular low-intensity exercise. "
        "Internal prototype — answers from indexed clinical documents only. Not clinical advice."
    )
    mock_verify.return_value = LLMGroundingCheck(grounded=True, reasoning="Supported by chunk.")

    state = initial_state_from_chat_request(
        ChatRequest(message="What does the COPD guideline say about exercise?", session_id="r1"),
    )
    result_state = run_rag_agent(state)

    assert result_state.get("error") is None
    assert "low-intensity exercise" in result_state.get("response_text", "")
    assert result_state.get("rag_result", {}).get("grounded") is True
    assert result_state.get("rag_result", {}).get("relevant_count") == 1

    chat = chat_response_from_state({**state, **result_state})
    assert chat.rag is not None
    assert chat.rag.grounded is True
    assert chat.rag.citations[0].source_file == "copd_guideline.md"


@patch("agents.subagents.rag_agent.grade_chunks")
@patch("agents.subagents.rag_agent.retrieve_chunks_for_query")
def test_run_rag_agent_no_relevant_chunks(
    mock_retrieve: Any,
    mock_grade: Any,
    indexed_vectorstore: DocumentVectorStore,
) -> None:
    mock_retrieve.return_value = [
        RetrievedChunk(
            chunk_id="copd_guideline.md::exercise",
            source_file="copd_guideline.md",
            section_name="Exercise",
            content="Exercise guidance",
            score=0.5,
        )
    ]
    mock_grade.return_value = []

    state = initial_state_from_chat_request(
        ChatRequest(message="What medication dose should be used?", session_id="r2"),
    )
    result_state = run_rag_agent(state)

    assert "indexed clinical documents" in result_state.get("response_text", "").lower()
    assert result_state.get("rag_result", {}).get("relevant_count") == 0


def test_run_rag_agent_missing_index(tmp_path) -> None:
    reset_vectorstore()
    empty_store = DocumentVectorStore(
        chroma_dir=tmp_path / "empty-chroma",
        documents_dir=tmp_path / "documents",
        collection_name="empty_rag",
        embedding_function=build_test_embedding_function(),
    )
    (tmp_path / "documents").mkdir()

    with patch("agents.subagents.rag_agent.get_vectorstore", return_value=empty_store):
        state = initial_state_from_chat_request(ChatRequest(message="Any question?"))
        result_state = run_rag_agent(state)

    assert "index is empty" in result_state.get("response_text", "").lower()
    reset_vectorstore()
