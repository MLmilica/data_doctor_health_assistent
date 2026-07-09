"""FastAPI dependencies and application singletons."""

from __future__ import annotations

import os
from typing import Any

from agents.graph import build_graph, invoke_chat_graph
from config import settings
from ml.predict import ModelRegistry
from schemas.chat import ChatRequest, ChatResponse

_graph: Any | None = None
_ml_models_loaded: bool = False


def is_llm_configured() -> bool:
    if settings.llm_provider == "anthropic":
        return bool(settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    return bool(settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))


def are_ml_models_loaded() -> bool:
    return _ml_models_loaded


def get_graph():
    if _graph is None:
        raise RuntimeError("LangGraph is not initialized. API startup may have failed.")
    return _graph


def invoke_chat(request: ChatRequest) -> ChatResponse:
    return invoke_chat_graph(get_graph(), request)


def startup() -> None:
    global _graph, _ml_models_loaded

    _ml_models_loaded = False
    try:
        ModelRegistry.load()
        _ml_models_loaded = True
    except FileNotFoundError:
        pass

    _graph = build_graph()


def shutdown() -> None:
    global _graph, _ml_models_loaded

    _graph = None
    _ml_models_loaded = False
    ModelRegistry.reset()
