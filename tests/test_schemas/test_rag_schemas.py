"""Tests for RAG and citation schemas."""

from __future__ import annotations

from schemas.citation import Citation, RetrievedChunk
from schemas.rag import (
    LLMChunkGrade,
    LLMChunkGradingResult,
    LLMGroundingCheck,
    RAGQueryResult,
)


def test_citation_from_chunk_truncates_long_snippet() -> None:
    chunk = RetrievedChunk(
        chunk_id="doc1:exercise",
        source_file="copd_guideline.md",
        section_name="Exercise",
        content="x" * 500,
        score=0.91,
    )
    citation = Citation.from_chunk(chunk, snippet_max_chars=100)
    assert citation.source_file == "copd_guideline.md"
    assert citation.section_name == "Exercise"
    assert citation.score == 0.91
    assert len(citation.snippet) == 100
    assert citation.snippet.endswith("...")


def test_rag_query_result_round_trip() -> None:
    chunk = RetrievedChunk(
        chunk_id="doc1:exercise",
        source_file="copd_guideline.md",
        section_name="Exercise",
        content="Patients should engage in regular low-intensity exercise.",
        score=0.88,
        relevant=True,
        relevance_reason="Directly answers exercise question.",
    )
    result = RAGQueryResult(
        user_message="What does the COPD guideline say about exercise?",
        retrieved_count=5,
        relevant_count=1,
        citations=[Citation.from_chunk(chunk)],
        chunks_used=[chunk],
        grounded=True,
    )
    restored = RAGQueryResult.model_validate(result.model_dump())
    assert restored.relevant_count == 1
    assert restored.citations[0].snippet.startswith("Patients should")
    assert restored.grounded is True


def test_llm_structured_outputs_defaults() -> None:
    grading = LLMChunkGradingResult(
        grades=[LLMChunkGrade(chunk_id="a", relevant=True, reason="on topic")],
    )
    grounding = LLMGroundingCheck(grounded=False, unsupported_claims=["dose not in sources"])
    assert len(grading.grades) == 1
    assert grounding.unsupported_claims == ["dose not in sources"]
