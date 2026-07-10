"""Insight Tool — load offline SHAP/importance artifacts and synthesize prose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.subagents.prediction_agent import configure_llm_environment, synthesis_llm
from agents.tools.data_task_classifier import detect_insight_target
from config import settings
from schemas.insight import INSIGHT_DISCLAIMER, InsightResult, InsightTarget

INSIGHT_SYNTHESIS_SYSTEM_PROMPT = """You explain model insights for clinical analysts.

Rules:
- Use ONLY the pre-computed feature rankings and scores in the provided JSON.
- Do not invent features, rankings, or numeric importance values.
- Explain what the top drivers mean in plain language (2-4 short paragraphs or bullets).
- Mention that insights come from offline model training on the synthetic dataset.
- End with the disclaimer field verbatim from the JSON.
"""


class InsightToolError(ValueError):
    """Raised when insight artifacts are missing or invalid."""


def _insight_paths(target: InsightTarget) -> tuple[Path, Path]:
    insights_dir = settings.artifacts_dir / "insights"
    return (
        insights_dir / f"{target}_shap_summary.json",
        insights_dir / f"{target}_feature_importance.json",
    )


def load_insight_result(target: InsightTarget) -> InsightResult:
    """Load pre-computed SHAP and feature-importance artifacts."""
    shap_path, importance_path = _insight_paths(target)

    if not shap_path.exists():
        raise InsightToolError(
            f"Missing insight artifact: {shap_path.name}. Run `uv run python -m ml.train` first."
        )

    shap_summary = json.loads(shap_path.read_text(encoding="utf-8"))
    feature_importance: dict[str, Any] = {}
    if importance_path.exists():
        feature_importance = json.loads(importance_path.read_text(encoding="utf-8"))

    top_features = list(shap_summary.get("top_features", []))
    if not top_features and feature_importance.get("top_features"):
        top_features = list(feature_importance["top_features"])

    return InsightResult(
        target=target,
        source="precomputed",
        shap_summary=shap_summary,
        feature_importance=feature_importance,
        top_features=top_features,
        artifact_paths={
            "shap_summary": str(shap_path),
            "feature_importance": str(importance_path) if importance_path.exists() else "",
        },
        disclaimer=INSIGHT_DISCLAIMER,
    )


def resolve_insight_target(user_message: str) -> InsightTarget:
    """Map a user question to COPD or ALT insight artifacts."""
    target = detect_insight_target(user_message)
    if target is None:
        raise InsightToolError(
            "Please specify whether you want insights for COPD or ALT predictions."
        )
    return target  # type: ignore[return-value]


def _insight_facts_payload(user_message: str, result: InsightResult) -> dict[str, Any]:
    return {
        "user_message": user_message,
        "target": result.target,
        "source": result.source,
        "top_features": result.top_features[:10],
        "shap_method": result.shap_summary.get("method"),
        "model": result.shap_summary.get("model") or result.feature_importance.get("model"),
        "sample_size": result.shap_summary.get("sample_size"),
        "artifact_paths": result.artifact_paths,
        "disclaimer": result.disclaimer,
    }


def synthesize_insight_response_text(facts: dict[str, Any]) -> str:
    """LLM: natural-language answer from read-only insight artifacts."""
    configure_llm_environment()
    llm = synthesis_llm()
    response = llm.invoke(
        [
            SystemMessage(content=INSIGHT_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(facts, indent=2)),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def format_insight_fallback(result: InsightResult) -> str:
    """Deterministic fallback when insight synthesis LLM is unavailable."""
    lines = [
        f"Top model drivers for {result.target.upper()} (offline SHAP/importance):",
        "",
    ]
    for index, item in enumerate(result.top_features[:5], start=1):
        feature = item.get("feature", "unknown")
        score = item.get("mean_abs_shap", item.get("score"))
        if score is not None:
            lines.append(f"{index}. {feature} (score={score})")
        else:
            lines.append(f"{index}. {feature}")
    lines.extend(["", result.disclaimer])
    return "\n".join(lines)


def run_insight_tool(user_message: str) -> tuple[InsightResult, InsightTarget]:
    """Resolve target, load artifacts, and return insight context."""
    target = resolve_insight_target(user_message)
    result = load_insight_result(target)
    return result, target
