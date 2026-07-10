"""Tests for LangSmith configuration helpers."""

from __future__ import annotations

import os

import pytest

from observability.langsmith import build_graph_run_config, configure_langsmith, is_langsmith_enabled
from schemas.chat import ChatRequest


@pytest.fixture(autouse=True)
def reset_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_configure_langsmith_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("observability.langsmith.settings.langchain_tracing_v2", True)
    monkeypatch.setattr("observability.langsmith.settings.langchain_api_key", "")
    monkeypatch.setattr("observability.langsmith.settings.langchain_project", "data-doctor-test")

    assert configure_langsmith() is False
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGCHAIN_PROJECT"] == "data-doctor-test"


def test_configure_langsmith_enabled_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("observability.langsmith.settings.langchain_tracing_v2", True)
    monkeypatch.setattr("observability.langsmith.settings.langchain_api_key", "ls-test-key")
    monkeypatch.setattr("observability.langsmith.settings.langchain_project", "data-doctor-test")

    assert configure_langsmith() is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"
    assert is_langsmith_enabled() is True


def test_build_graph_run_config_groups_by_session() -> None:
    config = build_graph_run_config(
        ChatRequest(message="hello", session_id="sess-1", user_id="user-42"),
    )
    assert config["configurable"]["thread_id"] == "user-42:sess-1"
    assert config["metadata"]["session_id"] == "sess-1"
    assert config["run_name"] == "chat_graph"
