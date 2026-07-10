"""RAG agent schemas — retrieval results and LLM structured outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.citation import Citation, RetrievedChunk

RAG_DISCLAIMER = (
    "Internal prototype — answers from indexed clinical documents only. "
    "Not clinical advice."
)


class LLMChunkGrade(BaseModel):
    """Relevance grade for one retrieved chunk (corrective RAG)."""

    chunk_id: str
    relevant: bool
    reason: str = ""


class LLMChunkGradingResult(BaseModel):
    """Structured output when grading all retrieved chunks at once."""

    grades: list[LLMChunkGrade] = Field(default_factory=list)


class LLMGroundingCheck(BaseModel):
    """Structured output for self-RAG grounding verification."""

    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


class RAGQueryResult(BaseModel):
    """Full RAG pipeline result stored on AgentState and mapped to ChatResponse."""

    user_message: str
    retrieved_count: int
    relevant_count: int
    citations: list[Citation] = Field(default_factory=list)
    chunks_used: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Relevant chunks passed to synthesis.",
    )
    grounded: bool
    grounding_retry_count: int = 0
    disclaimer: str = RAG_DISCLAIMER
