"""Citation and retrieved chunk schemas for the RAG agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A source excerpt surfaced to the UI and API metadata."""

    source_file: str
    section_name: str
    snippet: str
    score: float | None = Field(
        default=None,
        description="Similarity score from vector retrieval, when available.",
    )

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk, *, snippet_max_chars: int = 400) -> Citation:
        snippet = chunk.content.strip()
        if len(snippet) > snippet_max_chars:
            snippet = snippet[: snippet_max_chars - 3].rstrip() + "..."
        return cls(
            source_file=chunk.source_file,
            section_name=chunk.section_name,
            snippet=snippet,
            score=chunk.score,
        )


class RetrievedChunk(BaseModel):
    """One indexed document section returned from vector search."""

    chunk_id: str
    source_file: str
    section_name: str
    content: str
    score: float
    relevant: bool | None = Field(
        default=None,
        description="Set after corrective grading.",
    )
    relevance_reason: str | None = Field(
        default=None,
        description="Short LLM rationale when graded.",
    )
