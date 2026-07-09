"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from config import settings
from data.profile import build_and_save_data_profile
from ml.predict import ModelRegistry
from ml.train import (
    ALT_PIPELINE_PATH,
    ALT_SHAP_PATH,
    COPD_LABEL_ENCODER_PATH,
    COPD_PIPELINE_PATH,
    COPD_SHAP_PATH,
    train_models,
)


@pytest.fixture(scope="session")
def ml_artifacts() -> Generator[None, None, None]:
    """Ensure data profile and trained models exist (CI has no gitignored artifacts)."""
    profile_path = settings.artifacts_dir / "data_profile.json"
    if not profile_path.exists():
        build_and_save_data_profile()

    if not all(
        path.exists()
        for path in (
            COPD_PIPELINE_PATH,
            COPD_LABEL_ENCODER_PATH,
            ALT_PIPELINE_PATH,
            COPD_SHAP_PATH,
            ALT_SHAP_PATH,
        )
    ):
        train_models()

    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


@pytest.fixture(autouse=True)
def openai_api_key_for_agent_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent unit tests mock LLM calls but still pass the API key guard."""
    if not settings.openai_api_key:
        monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", settings.openai_api_key or "test-key")
