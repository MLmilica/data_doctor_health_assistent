"""Shared read-only SQL access layer over the patient DuckDB table."""

from __future__ import annotations

import re
from typing import Any

from config import settings
from data.loader import PatientDataLoader
from data.schema_registry import PATIENT_TABLE_NAME, get_dataset_schema
from schemas.data import DatasetSchema
from schemas.sql import DataQueryResult

_FORBIDDEN_SQL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|pragma|set|call|execute|truncate)\b",
        r"\b(read_csv|read_parquet|glob|http|https|s3://)\b",
    )
)


def _extract_cte_names(sql: str) -> set[str]:
    """Return CTE aliases declared in WITH ... AS (...) clauses."""
    ctes: set[str] = set()
    if not re.match(r"\s*with\b", sql, re.IGNORECASE):
        return ctes
    for match in re.finditer(
        r"(?:\bwith|,)\s+([a-z_][a-z0-9_]*)\s+as\s*\(",
        sql,
        re.IGNORECASE,
    ):
        ctes.add(match.group(1).lower())
    return ctes


def _extract_table_references(sql: str) -> list[str]:
    """Return table names referenced after FROM or JOIN."""
    return [
        match.group(1).lower()
        for match in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", sql, re.IGNORECASE)
    ]


class SqlValidationError(ValueError):
    """Raised when generated SQL fails safety checks."""


class SqlLayer:
    """Validated, read-only SQL execution over the patient dataset."""

    def __init__(self, loader: PatientDataLoader | None = None) -> None:
        self._loader = loader or PatientDataLoader()
        self._schema = get_dataset_schema(self._loader.row_count)

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    def close(self) -> None:
        self._loader.close()

    def schema_prompt(self) -> str:
        """Compact schema description for LLM SQL generation."""
        lines = [
            f"IMPORTANT: The database has exactly ONE table: `{self._schema.table_name}`.",
            "There are no other tables (no avg_bmi, bmi_stats, etc.).",
            "For averages or counts, use SQL aggregates on columns in that table "
            "(e.g. `SELECT AVG(bmi) AS average_bmi FROM patients`).",
            f"Table `{self._schema.table_name}` ({self._schema.row_count} rows).",
            "Generate DuckDB SQL using only this table and the columns below.",
            "",
        ]
        for column in self._schema.columns:
            allowed = ""
            if column.allowed_values:
                allowed = f" allowed values: {', '.join(column.allowed_values)}"
            target = " [target]" if column.is_target else ""
            lines.append(
                f"- {column.name} ({column.dtype}): {column.description}{allowed}{target}"
            )
        return "\n".join(lines)

    def validate_sql(self, sql: str) -> str:
        """Normalize and validate a read-only SELECT statement."""
        cleaned = self._normalize_sql(sql)
        if not cleaned:
            raise SqlValidationError("SQL is empty.")

        if ";" in cleaned:
            raise SqlValidationError("Multiple SQL statements are not allowed.")

        for pattern in _FORBIDDEN_SQL_PATTERNS:
            if pattern.search(cleaned):
                raise SqlValidationError("Only a single read-only SELECT over patients is allowed.")

        if not re.match(r"^(with|select)\b", cleaned, re.IGNORECASE):
            raise SqlValidationError("Query must start with SELECT (or WITH ... SELECT).")

        if PATIENT_TABLE_NAME not in cleaned.lower():
            raise SqlValidationError(f"Query must read from `{PATIENT_TABLE_NAME}`.")

        self._validate_table_references(cleaned)

        return self._ensure_limit(cleaned)

    def _validate_table_references(self, sql: str) -> None:
        """Ensure every FROM/JOIN target is `patients` or a CTE defined in the same query."""
        allowed_ctes = _extract_cte_names(sql)
        allowed = {PATIENT_TABLE_NAME, *allowed_ctes}
        for table in _extract_table_references(sql):
            if table not in allowed:
                raise SqlValidationError(
                    f"Unknown table `{table}`. Only `{PATIENT_TABLE_NAME}` exists; "
                    "use column aggregates (e.g. AVG(bmi)), not invented table names."
                )

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Collapse whitespace and strip a single trailing semicolon from LLM output."""
        cleaned = " ".join(sql.strip().split())
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].rstrip()
        return cleaned

    def execute(
        self,
        sql: str,
        *,
        explanation: str | None = None,
    ) -> DataQueryResult:
        """Validate and run a SELECT query."""
        safe_sql = self.validate_sql(sql)
        frame = self._loader.query(safe_sql)
        rows = _normalize_rows(frame.to_dict(orient="records"))
        truncated = len(rows) >= settings.sql_max_rows
        return DataQueryResult(
            sql=safe_sql,
            columns=[str(column) for column in frame.columns],
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            explanation=explanation,
        )

    def _ensure_limit(self, sql: str) -> str:
        if re.search(r"\blimit\b", sql, re.IGNORECASE):
            return sql
        return f"{sql} LIMIT {settings.sql_max_rows}"


_sql_layer: SqlLayer | None = None


def get_sql_layer() -> SqlLayer:
    """Return a process-wide SQL layer singleton."""
    global _sql_layer
    if _sql_layer is None:
        _sql_layer = SqlLayer()
    return _sql_layer


def reset_sql_layer() -> None:
    """Close and clear the singleton (mainly for tests)."""
    global _sql_layer
    if _sql_layer is not None:
        _sql_layer.close()
    _sql_layer = None


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                key: (
                    value.item()
                    if hasattr(value, "item") and callable(value.item)
                    else value
                )
                for key, value in row.items()
            }
        )
    return normalized
