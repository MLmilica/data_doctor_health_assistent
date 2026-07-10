"""Tests for the data agent (LLM mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agents.state import chat_response_from_state, initial_state_from_chat_request
from agents.subagents.data_agent import format_data_response, run_data_agent
from agents.tools.sql_layer import reset_sql_layer
from schemas.chat import ChatRequest
from schemas.sql import DATA_QUERY_DISCLAIMER, DataQueryResult, LLMSQLExtraction


def setup_function() -> None:
    reset_sql_layer()


def teardown_function() -> None:
    reset_sql_layer()


@patch("agents.subagents.data_agent.synthesize_data_response_text")
@patch("agents.subagents.data_agent.extract_sql_with_llm")
def test_run_data_agent_executes_query(mock_extract: Any, mock_synthesize: Any) -> None:
    mock_extract.return_value = LLMSQLExtraction(
        sql=(
            "SELECT income_bracket, COUNT(*) AS patient_count "
            "FROM patients GROUP BY income_bracket ORDER BY income_bracket"
        ),
        explanation="Count patients in each income bracket.",
    )
    mock_synthesize.return_value = (
        "There are patients across three income brackets. "
        f"{DATA_QUERY_DISCLAIMER}"
    )

    state = initial_state_from_chat_request(
        ChatRequest(message="How many patients per income bracket?", session_id="d1"),
    )
    result_state = run_data_agent(state)

    assert result_state.get("error") is None
    assert result_state.get("data_result") is not None
    assert "income brackets" in result_state.get("response_text", "")
    assert result_state.get("data_result", {}).get("row_count") == 3
    mock_synthesize.assert_called_once()

    chat = chat_response_from_state({**state, **result_state})
    assert chat.data_query is not None
    assert chat.data_query.row_count == 3


@patch("agents.subagents.data_agent.synthesize_data_response_text", side_effect=RuntimeError("llm down"))
@patch("agents.subagents.data_agent.extract_sql_with_llm")
def test_run_data_agent_falls_back_when_synthesis_fails(mock_extract: Any, _mock_synthesize: Any) -> None:
    mock_extract.return_value = LLMSQLExtraction(
        sql="SELECT COUNT(*) AS total FROM patients",
        explanation="Total patients",
    )

    state = initial_state_from_chat_request(ChatRequest(message="How many patients?"))
    result_state = run_data_agent(state)

    assert result_state.get("error") is None
    assert "SELECT COUNT(*) AS total FROM patients" in result_state.get("response_text", "")


@patch("agents.subagents.data_agent.extract_sql_with_llm")
def test_run_data_agent_clarification(mock_extract: Any) -> None:
    mock_extract.return_value = LLMSQLExtraction(
        sql="SELECT COUNT(*) FROM patients",
        requires_clarification=True,
        clarification_prompt="Which grouping should I use?",
    )

    state = initial_state_from_chat_request(ChatRequest(message="Show me counts"))
    result_state = run_data_agent(state)

    assert "data_result" not in result_state
    assert result_state.get("response_text") == "Which grouping should I use?"
    assert result_state.get("requires_clarification") is True


@patch("agents.subagents.data_agent.synthesize_data_response_text")
@patch("agents.subagents.data_agent.extract_sql_with_llm")
def test_run_data_agent_passes_combo_context_for_multi_part_question(
    mock_extract: Any,
    mock_synthesize: Any,
) -> None:
    mock_extract.return_value = LLMSQLExtraction(
        sql="SELECT AVG(bmi) AS average_bmi FROM patients",
        explanation="Average BMI in dataset.",
    )
    mock_synthesize.return_value = "Average BMI is about 27."

    message = (
        "What is the average BMI in the dataset and what do documents "
        "recommend for low-impact exercise?"
    )
    state = initial_state_from_chat_request(ChatRequest(message=message, session_id="combo-1"))
    run_data_agent(state)

    assert mock_extract.call_args.kwargs.get("combo_context") is True


def test_format_data_response_includes_sql_and_rows() -> None:
    text = format_data_response(
        DataQueryResult(
            sql="SELECT COUNT(*) AS total FROM patients",
            columns=["total"],
            rows=[{"total": 10000}],
            row_count=1,
            explanation="Total patients",
        )
    )
    assert "SELECT COUNT(*) AS total FROM patients" in text
    assert "10000" in text


@patch("agents.subagents.data_agent.synthesize_chart_response_text")
@patch("agents.subagents.data_agent.extract_chart_with_llm")
def test_run_data_agent_chart_path(mock_extract: Any, mock_synthesize: Any) -> None:
    from schemas.chart import LLMChartExtraction

    mock_extract.return_value = LLMChartExtraction(
        chart_type="boxplot",
        x_column="readmitted",
        y_column="last_lab_glucose",
        title="Glucose by readmission",
        explanation="Compare glucose across readmitted groups.",
    )
    mock_synthesize.return_value = "Here is a boxplot comparing glucose levels."

    state = initial_state_from_chat_request(
        ChatRequest(
            message="Compare glucose levels between readmitted and non-readmitted patients.",
            session_id="chart-1",
        ),
    )
    result_state = run_data_agent(state)

    assert result_state.get("error") is None
    assert result_state.get("chart_result") is not None
    assert result_state["chart_result"]["spec"]["chart_type"] == "boxplot"

    chat = chat_response_from_state({**state, **result_state})
    assert chat.chart is not None
    assert chat.chart.chart_type == "boxplot"
    assert chat.chart.plotly_json


@patch("agents.subagents.data_agent.synthesize_insight_response_text")
@patch("agents.subagents.data_agent.run_insight_tool")
def test_run_data_agent_insight_path(
    mock_run_insight: Any,
    mock_synthesize: Any,
    ml_artifacts: None,
) -> None:
    from agents.tools.insight_tool import run_insight_tool as real_run_insight

    insight_result, target = real_run_insight("What are the main risk factors for COPD?")
    mock_run_insight.return_value = (insight_result, target)
    mock_synthesize.return_value = "Exercise and diet are top COPD drivers."

    state = initial_state_from_chat_request(
        ChatRequest(message="What are the main risk factors for COPD?", session_id="insight-1"),
    )
    result_state = run_data_agent(state)

    assert result_state.get("error") is None
    assert result_state.get("insight_result") is not None

    chat = chat_response_from_state({**state, **result_state})
    assert chat.insight is not None
    assert chat.insight.target == "copd"
    assert chat.insight.top_features

