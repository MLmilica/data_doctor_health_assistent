"""Session memory schemas — transcript, step ledger, and session facts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

ChatRole = Literal["user", "assistant"]
StepStatus = Literal["ok", "clarification", "error"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatTurn(BaseModel):
    """One user or assistant message in a session transcript."""

    role: ChatRole
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    routed_to: str | None = None


class StepRecord(BaseModel):
    """Compact record of what the backend did in one graph step."""

    step_id: str = Field(default_factory=lambda: str(uuid4()))
    agent: str
    tool: str | None = None
    status: StepStatus = "ok"
    artifact: dict[str, Any] = Field(default_factory=dict)
    assistant_summary: str = ""
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


class SessionFacts(BaseModel):
    """Structured session context for deterministic follow-ups."""

    last_route: str | None = None
    last_target: str | None = None
    last_features: dict[str, Any] = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)
    last_sql: str | None = None


class ChatSession(BaseModel):
    """Server-side session persisted across /chat invocations."""

    session_id: str
    user_id: str
    turns: list[ChatTurn] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    facts: SessionFacts = Field(default_factory=SessionFacts)
    updated_at: datetime = Field(default_factory=utc_now)
