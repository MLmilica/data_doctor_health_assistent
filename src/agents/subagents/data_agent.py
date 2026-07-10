"""Data Agent — natural language → validated SQL → DuckDB results → LLM synthesis."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState, _merge_state, append_agent_step, set_data_result
from agents.subagents.prediction_agent import (
    configure_llm_environment,
    require_llm_api_key,
    routing_llm,
    synthesis_llm,
)
from agents.multi_step import is_multi_part_request
from agents.tools.sql_layer import SqlValidationError, get_sql_layer
from config import settings
from memory.persistence import append_run_step_record
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


def run_data_agent(state: AgentState) -> AgentState:
    """LangGraph node: generate SQL, validate, execute, and format results."""
    started = time.perf_counter()
    user_message = state.get("user_message", "")
    llm_model = f"{settings.llm_model_routing}+{settings.llm_model_synthesis}"
    extraction: LLMSQLExtraction | None = None

    try:
        require_llm_api_key()
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
                updated = _merge_state(
                    state,
                    response_text=response_text,
                    requires_clarification=True,
                    clarification_prompt=response_text,
                    llm_model=llm_model,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                return append_agent_step(append_run_step_record(updated), "data")

            try:
                query_result = layer.execute(extraction.sql, explanation=extraction.explanation)
                break
            except (SqlValidationError, Exception) as exc:
                if attempt == 0:
                    correction_hint = (
                        f"Error: {exc}\nRejected SQL:\n{extraction.sql}"
                    )
                    continue
                raise
        state = set_data_result(state, query_result)
        facts = _data_facts_payload(user_message, query_result)
        try:
            response_text = synthesize_data_response_text(facts)
        except Exception:
            response_text = format_data_response(query_result)
        prior_latency = state.get("latency_ms") or 0.0
        updated = _merge_state(
            state,
            response_text=response_text,
            requires_clarification=False,
            clarification_prompt=None,
            llm_model=llm_model,
            latency_ms=round(prior_latency + (time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(append_run_step_record(updated), "data")

    except SqlValidationError as exc:
        sql_detail = f"\n\nGenerated SQL:\n{extraction.sql}" if extraction is not None else ""
        updated = _merge_state(
            state,
            response_text=f"Could not run that query safely: {exc}{sql_detail}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(append_run_step_record(updated), "data")
    except FileNotFoundError as exc:
        updated = _merge_state(
            state,
            response_text=f"Patient dataset is not available: {exc}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(append_run_step_record(updated), "data")
    except Exception as exc:
        updated = _merge_state(
            state,
            response_text=f"Data query failed: {exc}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(append_run_step_record(updated), "data")
