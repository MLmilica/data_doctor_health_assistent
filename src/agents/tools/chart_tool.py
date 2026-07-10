"""Chart Tool — LLM ChartSpec → deterministic SQL → predefined Plotly renderers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, SystemMessage

from agents.subagents.prediction_agent import configure_llm_environment, routing_llm
from agents.tools.sql_layer import SqlLayer, get_sql_layer
from data.schema_registry import PATIENT_TABLE_NAME, get_column_names
from schemas.chart import CHART_DISCLAIMER, ChartResult, ChartSpec, ChartType, LLMChartExtraction

_ALLOWED_COLUMNS = set(get_column_names())


class ChartToolError(ValueError):
    """Raised when chart spec or rendering fails."""


CHART_EXTRACTION_SYSTEM_PROMPT = """You choose a predefined chart type and columns for clinical data visualization.

Rules:
- Output a chart specification only — never Plotly or matplotlib code.
- Use only column names from the provided schema.
- chart_type must be one of: histogram, bar, boxplot, scatter, line, heatmap.
- For comparing a numeric lab value across groups (e.g. readmitted vs not), use boxplot with x=group column and y=value column.
- For distribution of one numeric column, use histogram with x_column set.
- For relationship between two numeric columns, use scatter with x and y.
- For category counts, use bar with x_column as the category column.
- For correlation overview across numeric labs, use heatmap and list numeric columns in value_columns (at least 2).
- readmitted is 0/1; use it as a grouping column for comparisons.
- If the visualization request is ambiguous, set requires_clarification=true.
- Never set requires_clarification because the message also asks about documents or predictions.
"""


def _validate_column(name: str) -> str:
    if name not in _ALLOWED_COLUMNS:
        raise ChartToolError(f"Unknown column `{name}`.")
    return name


def _validate_columns(names: list[str]) -> list[str]:
    return [_validate_column(name) for name in names]


def chart_spec_from_extraction(extraction: LLMChartExtraction) -> ChartSpec:
    """Validate LLM extraction and attach deterministic SQL."""
    x_column = _validate_column(extraction.x_column)
    y_column = _validate_column(extraction.y_column) if extraction.y_column else None
    group_by = _validate_column(extraction.group_by) if extraction.group_by else None
    value_columns = _validate_columns(extraction.value_columns)

    if extraction.chart_type == "heatmap" and len(value_columns) < 2:
        if y_column is not None:
            value_columns = sorted({x_column, y_column, *value_columns})
        else:
            raise ChartToolError("Heatmap requires at least two numeric columns in value_columns.")

    sql = build_chart_sql(
        x_column=x_column,
        y_column=y_column,
        group_by=group_by,
        value_columns=value_columns,
    )
    return ChartSpec(
        chart_type=extraction.chart_type,
        x_column=x_column,
        y_column=y_column,
        group_by=group_by,
        value_columns=value_columns,
        title=extraction.title.strip() or "Patient dataset chart",
        explanation=extraction.explanation,
        sql=sql,
    )


def build_chart_sql(
    *,
    x_column: str,
    y_column: str | None = None,
    group_by: str | None = None,
    value_columns: list[str] | None = None,
) -> str:
    """Build a deterministic read-only SELECT for chart data."""
    columns: set[str] = {x_column}
    if y_column:
        columns.add(y_column)
    if group_by:
        columns.add(group_by)
    for column in value_columns or []:
        columns.add(column)
    cols_sql = ", ".join(sorted(columns))
    return f"SELECT {cols_sql} FROM {PATIENT_TABLE_NAME}"


def extract_chart_with_llm(
    user_message: str,
    *,
    schema_prompt: str,
    combo_context: bool = False,
) -> LLMChartExtraction:
    """LLM: natural language → chart specification."""
    configure_llm_environment()
    llm = routing_llm().with_structured_output(LLMChartExtraction)
    prompt = f"Schema:\n{schema_prompt}\n\nUser question:\n{user_message}"
    if combo_context:
        prompt += (
            "\n\nThis is a multi-part question. Choose chart columns only for the "
            "dataset visualization portion and ignore document or prediction parts."
        )
    result = llm.invoke(
        [
            SystemMessage(content=CHART_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    if isinstance(result, LLMChartExtraction):
        return result
    return LLMChartExtraction.model_validate(result)


def create_histogram(df: pd.DataFrame, x: str, *, title: str | None = None) -> dict[str, Any]:
    fig = px.histogram(df, x=x, title=title)
    return fig.to_plotly_json()


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str | None = None,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    if y and y in df.columns:
        fig = px.bar(df, x=x, y=y, title=title)
    else:
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, "count"]
        fig = px.bar(counts, x=x, y="count", title=title)
    return fig.to_plotly_json()


def create_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    fig = px.box(df, x=x, y=y, title=title)
    return fig.to_plotly_json()


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    fig = px.scatter(df, x=x, y=y, title=title)
    return fig.to_plotly_json()


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    ordered = df.sort_values(by=x)
    fig = px.line(ordered, x=x, y=y, title=title)
    return fig.to_plotly_json()


def create_heatmap(
    df: pd.DataFrame,
    columns: list[str],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    subset = df.loc[:, columns]
    numeric = subset.apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(title=title)
    return fig.to_plotly_json()


def render_chart(df: pd.DataFrame, spec: ChartSpec) -> dict[str, Any]:
    """Dispatch to a predefined Plotly renderer."""
    chart_type: ChartType = spec.chart_type
    title = spec.title

    if chart_type == "histogram":
        return create_histogram(df, spec.x_column, title=title)
    if chart_type == "bar":
        return create_bar_chart(df, spec.x_column, spec.y_column, title=title)
    if chart_type == "boxplot":
        if not spec.y_column:
            raise ChartToolError("Boxplot requires y_column.")
        x_col = spec.group_by or spec.x_column
        return create_boxplot(df, x_col, spec.y_column, title=title)
    if chart_type == "scatter":
        if not spec.y_column:
            raise ChartToolError("Scatter plot requires y_column.")
        return create_scatter_plot(df, spec.x_column, spec.y_column, title=title)
    if chart_type == "line":
        if not spec.y_column:
            raise ChartToolError("Line chart requires y_column.")
        return create_line_chart(df, spec.x_column, spec.y_column, title=title)
    if chart_type == "heatmap":
        columns = spec.value_columns or (
            [spec.x_column, spec.y_column] if spec.y_column else [spec.x_column]
        )
        if len(columns) < 2:
            raise ChartToolError("Heatmap requires at least two columns.")
        return create_heatmap(df, columns, title=title)

    raise ChartToolError(f"Unsupported chart type: {chart_type}")


def run_chart_tool(
    extraction: LLMChartExtraction,
    *,
    layer: SqlLayer | None = None,
) -> ChartResult:
    """Validate spec, fetch data via SQL layer, and render Plotly JSON."""
    spec = chart_spec_from_extraction(extraction)
    sql_layer = layer or get_sql_layer()
    query_result = sql_layer.execute(spec.sql, explanation=spec.explanation)
    if query_result.row_count == 0:
        raise ChartToolError("No rows returned for chart query.")

    frame = pd.DataFrame(query_result.rows)
    plotly_json = render_chart(frame, spec)
    return ChartResult(
        spec=spec,
        plotly_json=plotly_json,
        row_count=query_result.row_count,
        disclaimer=CHART_DISCLAIMER,
    )


def format_chart_fallback(result: ChartResult) -> str:
    """Deterministic fallback when chart synthesis LLM is unavailable."""
    spec = result.spec
    lines = [
        spec.explanation or spec.title,
        "",
        f"Chart type: {spec.chart_type}",
        f"Rows plotted: {result.row_count}",
        f"SQL: `{spec.sql}`",
        "",
        result.disclaimer,
    ]
    return "\n".join(lines)
