"""Tests for FastAPI chat and health endpoints."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from schemas.chat import ChatResponse, ChatPredictionDetails, ChatAgentMetadata


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok_when_llm_and_ml_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.routes.health.is_llm_configured", lambda: True)
    monkeypatch.setattr("api.routes.health.are_ml_models_loaded", lambda: True)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["api"] == "up"
    assert payload["llm_configured"] is True
    assert payload["ml_models_loaded"] is True


def test_health_degraded_when_llm_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.routes.health.is_llm_configured", lambda: False)
    monkeypatch.setattr("api.routes.health.are_ml_models_loaded", lambda: True)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["llm_configured"] is False
    assert "LLM API key" in payload["detail"]


@patch("api.routes.chat.invoke_chat")
def test_chat_success(mock_invoke: Any, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.routes.chat.is_llm_configured", lambda: True)
    mock_invoke.return_value = ChatResponse(
        text="Estimated ALT is about 30.",
        session_id="sess-1",
        prediction=ChatPredictionDetails(
            target="alt",
            prediction=30.0,
            can_predict=True,
            used_features={"bmi": 30.0},
        ),
        metadata=ChatAgentMetadata(
            agent="prediction",
            extraction_method="llm",
            llm_model="gpt-4o-mini+gpt-4o",
            latency_ms=10.0,
            routed_to="prediction",
        ),
    )

    response = client.post(
        "/chat",
        json={"message": "Predict ALT for BMI 30", "session_id": "sess-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Estimated ALT is about 30."
    assert payload["session_id"] == "sess-1"
    assert payload["prediction"]["prediction"] == 30.0
    mock_invoke.assert_called_once()


def test_chat_returns_503_when_llm_not_configured(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api.routes.chat.is_llm_configured", lambda: False)

    response = client.post(
        "/chat",
        json={"message": "Predict ALT for BMI 30"},
    )

    assert response.status_code == 503
    assert "LLM API key" in response.json()["detail"]
