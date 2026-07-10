"""LangSmith tracing configuration for LangChain / LangGraph."""

from __future__ import annotations

import os
from typing import Any

from config import settings
from schemas.chat import ChatRequest

_CONFIGURED = False


def configure_langsmith() -> bool:
    """
    Sync LangSmith settings from `.env` into process environment.

    LangChain reads LANGCHAIN_* from os.environ at runtime. Pydantic settings
    alone does not export them, so we mirror values here on startup and before
    LLM calls.
    """
    global _CONFIGURED

    enabled = settings.langchain_tracing_v2 and bool(settings.langchain_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    elif "LANGCHAIN_API_KEY" in os.environ and not enabled:
        os.environ.pop("LANGCHAIN_API_KEY", None)

    if settings.langchain_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    elif "LANGCHAIN_ENDPOINT" in os.environ:
        os.environ.pop("LANGCHAIN_ENDPOINT", None)

    _CONFIGURED = True
    return enabled


def is_langsmith_enabled() -> bool:
    """Return whether tracing is active for this process."""
    if not _CONFIGURED:
        configure_langsmith()
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def build_graph_run_config(request: ChatRequest) -> dict[str, Any]:
    """LangGraph invoke config — groups traces by user/session in LangSmith."""
    thread_id = f"{request.user_id}:{request.session_id}"
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "session_id": request.session_id,
            "user_id": request.user_id,
            "project": settings.langchain_project,
        },
        "tags": ["data-doctor", "chat"],
        "run_name": "chat_graph",
    }
