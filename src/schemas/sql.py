"""Schemas for SQL generation and query results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

DATA_QUERY_DISCLAIMER = (
    "Internal prototype output — analytics over a synthetic patient dataset. "
    "Not clinical advice."
)


class LLMSQLExtraction(BaseModel):
    """Structured SQL output from the data agent LLM."""

    sql: str = Field(description="A single DuckDB SELECT statement over the patients table.")
    explanation: str = Field(
        default="",
        description="Short plain-language summary of what the query computes.",
    )
    requires_clarification: bool = Field(
        default=False,
        description="True when the question is too vague to write SQL safely.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="Question to ask when clarification is required.",
    )


class DataQueryResult(BaseModel):
    """Validated SQL execution result."""

    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    explanation: str | None = None
