"""Heuristic routing inside the data agent: sql vs chart vs insight."""

from __future__ import annotations

import re
from typing import Literal

DataTask = Literal["sql", "chart", "insight"]

_INSIGHT_SIGNALS: tuple[str, ...] = (
    "risk factor",
    "risk factors",
    "main driver",
    "main drivers",
    "feature importance",
    "what affects",
    "what drives",
    "what influences",
    "shap",
    "strongest predictor",
    "key predictor",
    "important feature",
    "important features",
)

_CHART_SIGNALS: tuple[str, ...] = (
    "chart",
    "plot",
    "graph",
    "visualize",
    "visualise",
    "visualization",
    "visualisation",
    "distribution",
    "histogram",
    "boxplot",
    "box plot",
    "scatter",
    "heatmap",
    "heat map",
    "relationship between",
    "show the",
    "compare",
    "comparison",
    " versus ",
    " vs ",
)


_SQL_PRIORITY_SIGNALS: tuple[str, ...] = (
    "how many",
    "count",
    "average",
    "avg ",
    "mean ",
    "total ",
    "number of",
    "patients per",
    "per income",
    "per bracket",
    "group by",
)


def classify_data_task(message: str) -> DataTask:
    """Pick the data-agent sub-path from the user message."""
    lower = message.lower()

    if any(signal in lower for signal in _INSIGHT_SIGNALS):
        return "insight"

    chart_like = any(signal in lower for signal in _CHART_SIGNALS)
    sql_aggregate = any(signal in lower for signal in _SQL_PRIORITY_SIGNALS)

    if chart_like and not sql_aggregate:
        return "chart"

    return "sql"


def detect_insight_target(message: str) -> str | None:
    """Resolve COPD vs ALT for insight questions."""
    lower = message.lower()

    alt_signals = (
        r"\balt\b",
        "alanine",
        "aminotransferase",
        "liver enzyme",
        "liver function",
    )
    copd_signals = (
        r"\bcopd\b",
        "pulmonary",
        "lung disease",
        "obstructive",
    )

    has_alt = any(
        re.search(pattern, lower) if pattern.startswith(r"\b") else pattern in lower
        for pattern in alt_signals
    )
    has_copd = any(
        re.search(pattern, lower) if pattern.startswith(r"\b") else pattern in lower
        for pattern in copd_signals
    )

    if has_alt and not has_copd:
        return "alt"
    if has_copd and not has_alt:
        return "copd"
    if has_alt and has_copd:
        return None
    if any(signal in lower for signal in _INSIGHT_SIGNALS):
        return "copd"
    return None
