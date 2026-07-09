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
from agents.tools.sql_layer import SqlValidationError, get_sql_layer
from config import settings
from schemas.sql import DATA_QUERY_DISCLAIMER, DataQueryResult, LLMSQLExtraction

SQL_EXTRACTION_SYSTEM_PROMPT = """You write DuckDB SQL for clinical analytics questions.

Rules:
- Output exactly one read-only SELECT (or WITH ... SELECT) against the `patients` table.
- Use only columns from the provided schema.
- Prefer clear aliases for aggregates (e.g. patient_count, average_bmi).
- Do not use DDL/DML, file readers, or multiple statements.
- If the question is ambiguous, set requires_clarification=true and ask a focused follow-up.
- The dataset has no admission dates — do not invent month/time columns. For readmission analytics, use the `readmitted` flag.
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


def extract_sql_with_llm(user_message: str, *, schema_prompt: str) -> LLMSQLExtraction:
    """LLM: natural language → SQL extraction schema."""
    configure_llm_environment()
    llm = routing_llm().with_structured_output(LLMSQLExtraction)
    result = llm.invoke(
        [
            SystemMessage(content=SQL_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Schema:\n{schema_prompt}\n\n"
                    f"User question:\n{user_message}"
                )
            ),
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
        extraction = extract_sql_with_llm(user_message, schema_prompt=layer.schema_prompt())

        if extraction.requires_clarification:
            response_text = (
                extraction.clarification_prompt
                or "Could you clarify which metric or grouping you need from the patient dataset?"
            )
            updated = _merge_state(
                state,
                response_text=response_text,
                llm_model=llm_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return append_agent_step(updated, "data")

        query_result = layer.execute(extraction.sql, explanation=extraction.explanation)
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
            llm_model=llm_model,
            latency_ms=round(prior_latency + (time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(updated, "data")

    except SqlValidationError as exc:
        sql_detail = f"\n\nGenerated SQL:\n{extraction.sql}" if extraction is not None else ""
        updated = _merge_state(
            state,
            response_text=f"Could not run that query safely: {exc}{sql_detail}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(updated, "data")
    except FileNotFoundError as exc:
        updated = _merge_state(
            state,
            response_text=f"Patient dataset is not available: {exc}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(updated, "data")
    except Exception as exc:
        updated = _merge_state(
            state,
            response_text=f"Data query failed: {exc}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(updated, "data")
