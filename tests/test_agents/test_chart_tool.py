"""Tests for chart tool renderers and SQL builder."""

from __future__ import annotations

import pandas as pd
import pytest

from agents.tools.chart_tool import (
    ChartToolError,
    build_chart_sql,
    chart_spec_from_extraction,
    create_boxplot,
    create_histogram,
    create_scatter_plot,
    render_chart,
    run_chart_tool,
)
from agents.tools.sql_layer import reset_sql_layer
from schemas.chart import LLMChartExtraction


def setup_function() -> None:
    reset_sql_layer()


def teardown_function() -> None:
    reset_sql_layer()


def test_build_chart_sql_is_deterministic() -> None:
    sql = build_chart_sql(x_column="readmitted", y_column="last_lab_glucose")
    assert sql == "SELECT last_lab_glucose, readmitted FROM patients"


def test_chart_spec_from_extraction_validates_columns() -> None:
    spec = chart_spec_from_extraction(
        LLMChartExtraction(
            chart_type="boxplot",
            x_column="readmitted",
            y_column="last_lab_glucose",
            title="Glucose by readmission",
            explanation="Compare glucose across readmitted groups.",
        )
    )
    assert spec.chart_type == "boxplot"
    assert "readmitted" in spec.sql
    assert "last_lab_glucose" in spec.sql


def test_chart_spec_rejects_unknown_column() -> None:
    with pytest.raises(ChartToolError, match="Unknown column"):
        chart_spec_from_extraction(
            LLMChartExtraction(
                chart_type="histogram",
                x_column="not_a_column",
                title="Bad chart",
            )
        )


def test_create_boxplot_returns_plotly_json() -> None:
    frame = pd.DataFrame(
        {
            "readmitted": [0, 0, 1, 1],
            "last_lab_glucose": [90.0, 95.0, 110.0, 120.0],
        }
    )
    payload = create_boxplot(frame, "readmitted", "last_lab_glucose", title="Glucose")
    assert "data" in payload
    assert "layout" in payload


def test_render_chart_histogram() -> None:
    frame = pd.DataFrame({"bmi": [22.0, 25.0, 30.0, 28.0]})
    spec = chart_spec_from_extraction(
        LLMChartExtraction(
            chart_type="histogram",
            x_column="bmi",
            title="BMI distribution",
        )
    )
    payload = render_chart(frame, spec)
    assert payload["data"]


def test_run_chart_tool_executes_against_dataset() -> None:
    result = run_chart_tool(
        LLMChartExtraction(
            chart_type="scatter",
            x_column="bmi",
            y_column="alanine_aminotransferase",
            title="BMI vs ALT",
            explanation="Relationship between BMI and ALT.",
        )
    )
    assert result.row_count > 0
    assert result.plotly_json["data"]
    assert "bmi" in result.spec.sql
