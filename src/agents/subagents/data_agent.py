"""Data Agent — natural language → SQL / chart / insight → synthesis."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import (
    AgentState,
    _merge_state,
    append_agent_step,
    set_chart_result,
    set_data_result,
    set_insight_result,
)
from agents.subagents.prediction_agent import (
    configure_llm_environment,
    require_llm_api_key,
    routing_llm,
    synthesis_llm,
)
from agents.multi_step import is_multi_part_request
from agents.tools.chart_tool import (
    ChartToolError,
    extract_chart_with_llm,
    format_chart_fallback,
    run_chart_tool,
)
from agents.tools.data_task_classifier import classify_data_task
from agents.tools.insight_tool import (
    InsightToolError,
    format_insight_fallback,
    run_insight_tool,
    synthesize_insight_response_text,
    _insight_facts_payload,
)
from agents.tools.sql_layer import SqlValidationError, get_sql_layer
from config import settings
from memory.persistence import append_run_step_record
from schemas.chart import CHART_DISCLAIMER
from schemas.sql import DATA_QUERY_DISCLAIMER, DataQueryResult, LLMSQLExtraction

SQL_EXTRACTION_SYSTEM_PROMPT = """You write DuckDB SQL for clinical analytics questions.

Rules:
- Output exactly one read-only SELECT (or WITH ... SELECT) against the `patients` table.
- The database has ONLY the `patients` table. Never use FROM/JOIN with any other table name.
- For dataset-wide metrics (average BMI, counts, etc.), use aggregates on columns in `patients`.
  Example: `SELECT AVG(bmi) AS average_bmi FROM patients`
- Do not invent tables such as `avg_bmi`, `bmi_stats`, or `patient_summary`.
- Use only columns from the provided schema.
- Prefer clear aliases for aggregates (e.g. patient_count, average_bmi).
- Do not use DDL/DML, file readers, or multiple statements.
- If the question is ambiguous about the **dataset metric or grouping**, set requires_clarification=true.
- Never set requires_clarification because the message also asks about documents, predictions, or guidelines.
- The dataset has no admission dates — do not invent month/time columns. For readmission analytics, use the `readmitted` flag.
- For combo questions (e.g. average BMI plus document search), write SQL only for the analytics part and ignore the rest.
"""

COMBO_SQL_EXTRA_PROMPT = """This is a multi-part question (dataset + prediction and/or documents).

Your job is ONLY the dataset/SQL portion:
- Write SQL for clear analytics requests (average BMI, counts, group by, etc.) even if the message also asks about documents or predictions.
- Do NOT set requires_clarification because of document or prediction content — other agents handle those parts.
- If average BMI or a similar aggregate is requested, produce the SQL now.
"""

DATA_SYNTHESIS_SYSTEM_PROMPT = """You answer clinical analytics questions for an analyst audience.

Rules:
- Use ONLY the numbers and labels in the user-provided JSON (columns, rows, row_count). Do not invent or change any values.
- Summarize the findings clearly in natural language (2-4 short paragraphs or bullets).
- Reference the SQL explanation when helpful, but do not paste the full SQL query.
- If rows is empty, say no rows matched the query.
- If truncated is true, note that results were capped at the row limit.
- End with the disclaimer field verbatim from the JSON.
"""

CHART_SYNTHESIS_SYSTEM_PROMPT = """You describe a chart for clinical analysts.

Rules:
- Use ONLY the chart metadata in the provided JSON (chart_type, columns, row_count, explanation).
- Do not invent statistics or claim specific numeric findings unless they are in the JSON.
- Write 2-3 sentences describing what the chart shows and how to read it.
- End with the disclaimer field verbatim from the JSON.
"""


def _parse_sql_extraction(result: Any) -> LLMSQLExtraction:
    if isinstance(result, LLMSQLExtraction):
        return result
    return LLMSQLExtraction.model_validate(result)


def extract_sql_with_llm(
    user_message: str,
    *,
    schema_prompt: str,
    correction_hint: str | None = None,
    combo_context: bool = False,
) -> LLMSQLExtraction:
    """LLM: natural language → SQL extraction schema."""
    configure_llm_environment()
    llm = routing_llm().with_structured_output(LLMSQLExtraction)
    prompt = (
        f"Schema:\n{schema_prompt}\n\n"
        f"User question:\n{user_message}"
    )
    if combo_context:
        prompt += f"\n\n{COMBO_SQL_EXTRA_PROMPT}"
    if correction_hint:
        prompt += (
            "\n\nYour previous SQL was rejected or failed. "
            f"Fix it using only the `patients` table:\n{correction_hint}"
        )
    result = llm.invoke(
        [
            SystemMessage(content=SQL_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return _parse_sql_extraction(result)


def format_data_response(result: DataQueryResult) -> str:
    """Deterministic fallback when LLM synthesis is unavailable."""
    lines = [
        result.explanation or "Query results from the patient dataset:",
        "",
        f"SQL: `{result.sql}`",
        f"Rows returned: {result.row_count}",
    ]
    if result.truncated:
        lines.append(f"(Truncated to {settings.sql_max_rows} rows.)")

    if not result.rows:
        lines.append("")
        lines.append("No rows matched the query.")
    else:
        lines.append("")
        header = " | ".join(result.columns)
        lines.append(header)
        lines.append(" | ".join(["---"] * len(result.columns)))
        for row in result.rows[:20]:
            lines.append(" | ".join(str(row.get(column, "")) for column in result.columns))
        if result.row_count > 20:
            lines.append(f"... ({result.row_count - 20} more row(s) not shown)")

    lines.append("")
    lines.append(DATA_QUERY_DISCLAIMER)
    return "\n".join(lines)


def _data_facts_payload(user_message: str, result: DataQueryResult) -> dict[str, Any]:
    return {
        "user_message": user_message,
        "sql": result.sql,
        "explanation": result.explanation,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "disclaimer": DATA_QUERY_DISCLAIMER,
    }


def _chart_facts_payload(user_message: str, chart_result: Any) -> dict[str, Any]:
    spec = chart_result.spec
    return {
        "user_message": user_message,
        "chart_type": spec.chart_type,
        "x_column": spec.x_column,
        "y_column": spec.y_column,
        "group_by": spec.group_by,
        "title": spec.title,
        "explanation": spec.explanation,
        "sql": spec.sql,
        "row_count": chart_result.row_count,
        "disclaimer": CHART_DISCLAIMER,
    }


def synthesize_data_response_text(facts: dict[str, Any]) -> str:
    """LLM: polish analyst prose using read-only query facts JSON."""
    configure_llm_environment()
    llm = synthesis_llm()
    response = llm.invoke(
        [
            SystemMessage(content=DATA_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(facts, indent=2)),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def synthesize_chart_response_text(facts: dict[str, Any]) -> str:
    """LLM: short prose for a rendered chart."""
    configure_llm_environment()
    llm = synthesis_llm()
    response = llm.invoke(
        [
            SystemMessage(content=CHART_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(facts, indent=2)),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _finish_data_agent(
    state: AgentState,
    *,
    response_text: str,
    started: float,
    llm_model: str,
    requires_clarification: bool = False,
    clarification_prompt: str | None = None,
    error: str | None = None,
) -> AgentState:
    prior_latency = state.get("latency_ms") or 0.0
    updated = _merge_state(
        state,
        response_text=response_text,
        requires_clarification=requires_clarification,
        clarification_prompt=clarification_prompt,
        llm_model=llm_model,
        latency_ms=round(prior_latency + (time.perf_counter() - started) * 1000, 2),
        error=error,
    )
    return append_agent_step(append_run_step_record(updated), "data")


def _run_sql_path(
    state: AgentState,
    *,
    user_message: str,
    started: float,
    llm_model: str,
) -> AgentState:
    extraction: LLMSQLExtraction | None = None
    layer = get_sql_layer()
    schema_prompt = layer.schema_prompt()
    correction_hint: str | None = None
    combo_context = is_multi_part_request(user_message)

    for attempt in range(2):
        extraction = extract_sql_with_llm(
            user_message,
            schema_prompt=schema_prompt,
            correction_hint=correction_hint,
            combo_context=combo_context,
        )

        if extraction.requires_clarification:
            response_text = (
                extraction.clarification_prompt
                or "Could you clarify which metric or grouping you need from the patient dataset?"
            )
            return _finish_data_agent(
                state,
                response_text=response_text,
                started=started,
                llm_model=llm_model,
                requires_clarification=True,
                clarification_prompt=response_text,
            )

        try:
            query_result = layer.execute(extraction.sql, explanation=extraction.explanation)
            break
        except (SqlValidationError, Exception) as exc:
            if attempt == 0:
                correction_hint = f"Error: {exc}\nRejected SQL:\n{extraction.sql}"
                continue
            raise

    state = set_data_result(state, query_result)
    facts = _data_facts_payload(user_message, query_result)
    try:
        response_text = synthesize_data_response_text(facts)
    except Exception:
        response_text = format_data_response(query_result)
    return _finish_data_agent(
        state,
        response_text=response_text,
        started=started,
        llm_model=llm_model,
    )


def _run_chart_path(
    state: AgentState,
    *,
    user_message: str,
    started: float,
    llm_model: str,
) -> AgentState:
    layer = get_sql_layer()
    schema_prompt = layer.schema_prompt()
    combo_context = is_multi_part_request(user_message)
    extraction = extract_chart_with_llm(
        user_message,
        schema_prompt=schema_prompt,
        combo_context=combo_context,
    )

    if extraction.requires_clarification:
        response_text = (
            extraction.clarification_prompt
            or "Could you clarify which columns or comparison you want visualized?"
        )
        return _finish_data_agent(
            state,
            response_text=response_text,
            started=started,
            llm_model=llm_model,
            requires_clarification=True,
            clarification_prompt=response_text,
        )

    chart_result = run_chart_tool(extraction, layer=layer)
    state = set_chart_result(state, chart_result)
    state = set_data_result(
        state,
        DataQueryResult(
            sql=chart_result.spec.sql,
            columns=[
                column
                for column in (
                    chart_result.spec.x_column,
                    chart_result.spec.y_column,
                    chart_result.spec.group_by,
                )
                if column
            ],
            rows=[],
            row_count=chart_result.row_count,
            explanation=chart_result.spec.explanation,
        ),
    )
    facts = _chart_facts_payload(user_message, chart_result)
    try:
        response_text = synthesize_chart_response_text(facts)
    except Exception:
        response_text = format_chart_fallback(chart_result)
    return _finish_data_agent(
        state,
        response_text=response_text,
        started=started,
        llm_model=llm_model,
    )


def _run_insight_path(
    state: AgentState,
    *,
    user_message: str,
    started: float,
    llm_model: str,
) -> AgentState:
    insight_result, _target = run_insight_tool(user_message)
    state = set_insight_result(state, insight_result)
    facts = _insight_facts_payload(user_message, insight_result)
    try:
        response_text = synthesize_insight_response_text(facts)
    except Exception:
        response_text = format_insight_fallback(insight_result)
    return _finish_data_agent(
        state,
        response_text=response_text,
        started=started,
        llm_model=llm_model,
    )


def run_data_agent(state: AgentState) -> AgentState:
    """LangGraph node: SQL, chart, or insight depending on the user question."""
    started = time.perf_counter()
    user_message = state.get("user_message", "")
    llm_model = f"{settings.llm_model_routing}+{settings.llm_model_synthesis}"
    task = classify_data_task(user_message)

    try:
        require_llm_api_key()
        if task == "chart":
            return _run_chart_path(
                state,
                user_message=user_message,
                started=started,
                llm_model=llm_model,
            )
        if task == "insight":
            return _run_insight_path(
                state,
                user_message=user_message,
                started=started,
                llm_model=llm_model,
            )
        return _run_sql_path(
            state,
            user_message=user_message,
            started=started,
            llm_model=llm_model,
        )

    except SqlValidationError as exc:
        return _finish_data_agent(
            state,
            response_text=f"Could not run that query safely: {exc}",
            started=started,
            llm_model=llm_model,
            error=str(exc),
        )
    except (ChartToolError, InsightToolError) as exc:
        return _finish_data_agent(
            state,
            response_text=str(exc),
            started=started,
            llm_model=llm_model,
            error=str(exc),
        )
    except FileNotFoundError as exc:
        return _finish_data_agent(
            state,
            response_text=f"Patient dataset is not available: {exc}",
            started=started,
            llm_model=llm_model,
            error=str(exc),
        )
    except Exception as exc:
        return _finish_data_agent(
            state,
            response_text=f"Data query failed: {exc}",
            started=started,
            llm_model=llm_model,
            error=str(exc),
        )
