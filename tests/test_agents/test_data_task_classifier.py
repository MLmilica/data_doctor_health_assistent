"""Tests for data-agent sub-path classification."""

from __future__ import annotations

from agents.tools.data_task_classifier import classify_data_task, detect_insight_target


def test_classify_sql_for_count_question() -> None:
    assert classify_data_task("How many smokers are in the dataset?") == "sql"


def test_classify_chart_for_compare_question() -> None:
    message = "Compare glucose levels between readmitted and non-readmitted patients."
    assert classify_data_task(message) == "chart"


def test_classify_chart_for_distribution_question() -> None:
    assert classify_data_task("Show the distribution of BMI") == "chart"


def test_classify_insight_for_risk_factors() -> None:
    assert classify_data_task("What are the main risk factors for COPD?") == "insight"


def test_classify_sql_for_compare_average_question() -> None:
    message = "Compare average BMI in the dataset with ALT prediction for BMI 30"
    assert classify_data_task(message) == "sql"


def test_detect_insight_target_copd() -> None:
    assert detect_insight_target("What are the main risk factors for COPD?") == "copd"


def test_detect_insight_target_alt() -> None:
    assert detect_insight_target("What drives ALT predictions?") == "alt"


def test_detect_insight_target_ambiguous_returns_none() -> None:
    assert detect_insight_target("Compare COPD and ALT risk factors") is None
