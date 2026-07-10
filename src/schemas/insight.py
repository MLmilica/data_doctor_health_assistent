"""Schemas for the Insight Tool — offline ML artifacts + synthesis."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

INSIGHT_DISCLAIMER = (
    "Insights are derived from offline model analysis on a synthetic dataset. "
    "Not clinical advice."
)

InsightTarget = Literal["copd", "alt"]
InsightSource = Literal["precomputed"]


class InsightResult(BaseModel):
    """Pre-computed model insights loaded for LLM synthesis."""

    target: InsightTarget
    source: InsightSource = "precomputed"
    shap_summary: dict[str, Any] = Field(default_factory=dict)
    feature_importance: dict[str, Any] = Field(default_factory=dict)
    top_features: list[dict[str, Any]] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    disclaimer: str = INSIGHT_DISCLAIMER
