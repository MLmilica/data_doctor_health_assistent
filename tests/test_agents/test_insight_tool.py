"""Tests for insight tool artifact loading."""

from __future__ import annotations

import pytest

from agents.tools.insight_tool import (
    InsightToolError,
    format_insight_fallback,
    load_insight_result,
    resolve_insight_target,
    run_insight_tool,
)


def test_resolve_insight_target_copd() -> None:
    assert resolve_insight_target("What are the main risk factors for COPD?") == "copd"


def test_resolve_insight_target_requires_specificity() -> None:
    with pytest.raises(InsightToolError, match="specify whether"):
        resolve_insight_target("Compare COPD and ALT risk factors")


def test_load_insight_result_copd(ml_artifacts: None) -> None:
    result = load_insight_result("copd")
    assert result.target == "copd"
    assert len(result.top_features) > 0
    assert result.shap_summary["target"] == "copd"


def test_run_insight_tool_alt(ml_artifacts: None) -> None:
    result, target = run_insight_tool("What drives ALT predictions?")
    assert target == "alt"
    assert result.target == "alt"
    assert result.top_features


def test_format_insight_fallback_lists_features(ml_artifacts: None) -> None:
    result = load_insight_result("copd")
    text = format_insight_fallback(result)
    assert "COPD" in text
    assert result.top_features[0]["feature"] in text
