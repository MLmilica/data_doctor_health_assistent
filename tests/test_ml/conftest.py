"""Shared fixtures for ML tests."""

from __future__ import annotations

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
def ml_artifacts() -> None:
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
