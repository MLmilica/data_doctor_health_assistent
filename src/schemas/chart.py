"""Schemas for the Chart Tool — LLM chart spec and Plotly output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CHART_DISCLAIMER = (
    "Internal prototype chart — synthetic patient dataset. "
    "Not clinical advice."
)

ChartType = Literal["histogram", "bar", "boxplot", "scatter", "line", "heatmap"]


class LLMChartExtraction(BaseModel):
    """Structured chart specification from the data agent LLM."""

    chart_type: ChartType = Field(description="Predefined chart renderer to use.")
    x_column: str = Field(description="Primary column for the x-axis or histogram.")
    y_column: str | None = Field(
        default=None,
        description="Value column for boxplot, scatter, line, or bar charts.",
    )
    group_by: str | None = Field(
        default=None,
        description="Optional grouping column (e.g. readmitted).",
    )
    value_columns: list[str] = Field(
        default_factory=list,
        description="Extra numeric columns for heatmap correlation matrix.",
    )
    title: str = Field(description="Short chart title.")
    explanation: str = Field(
        default="",
        description="Plain-language description of what the chart shows.",
    )
    requires_clarification: bool = Field(
        default=False,
        description="True when the visualization request is too vague.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="Question to ask when clarification is required.",
    )


class ChartSpec(BaseModel):
    """Validated chart specification passed to render functions."""

    chart_type: ChartType
    x_column: str
    y_column: str | None = None
    group_by: str | None = None
    value_columns: list[str] = Field(default_factory=list)
    title: str
    explanation: str = ""
    sql: str = Field(description="Deterministic SELECT used to fetch chart data.")


class ChartResult(BaseModel):
    """Full chart pipeline output stored on AgentState."""

    spec: ChartSpec
    plotly_json: dict[str, Any]
    row_count: int
    disclaimer: str = CHART_DISCLAIMER
