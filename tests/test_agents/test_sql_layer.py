"""Tests for the read-only SQL layer."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from agents.tools.sql_layer import SqlLayer, SqlValidationError, reset_sql_layer
from data.schema_registry import PATIENT_TABLE_NAME


@pytest.fixture(autouse=True)
def _reset_sql_layer() -> Generator[None, None, None]:
    yield
    reset_sql_layer()


def test_sql_layer_executes_group_by_query() -> None:
    layer = SqlLayer()
    result = layer.execute(
        f"""
        SELECT income_bracket, COUNT(*) AS patient_count
        FROM {PATIENT_TABLE_NAME}
        GROUP BY income_bracket
        ORDER BY income_bracket
        """
    )
    assert result.row_count == 3
    assert "income_bracket" in result.columns
    assert sum(int(row["patient_count"]) for row in result.rows) == 10000


def test_sql_layer_appends_limit_when_missing() -> None:
    layer = SqlLayer()
    safe_sql = layer.validate_sql(f"SELECT patient_id FROM {PATIENT_TABLE_NAME}")
    assert "LIMIT" in safe_sql.upper()


def test_sql_layer_blocks_delete() -> None:
    layer = SqlLayer()
    with pytest.raises(SqlValidationError, match="read-only"):
        layer.validate_sql(f"DELETE FROM {PATIENT_TABLE_NAME}")


def test_sql_layer_blocks_multiple_statements() -> None:
    layer = SqlLayer()
    with pytest.raises(SqlValidationError, match="Multiple SQL statements"):
        layer.validate_sql(
            f"SELECT COUNT(*) FROM {PATIENT_TABLE_NAME}; DROP TABLE {PATIENT_TABLE_NAME}"
        )


def test_sql_layer_allows_trailing_semicolon() -> None:
    layer = SqlLayer()
    safe_sql = layer.validate_sql(f"SELECT patient_id FROM {PATIENT_TABLE_NAME};")
    assert safe_sql.upper().startswith("SELECT")
    assert ";" not in safe_sql
    assert "LIMIT" in safe_sql.upper()


def test_sql_layer_executes_query_with_trailing_semicolon() -> None:
    layer = SqlLayer()
    result = layer.execute(
        f"SELECT COUNT(*) AS total FROM {PATIENT_TABLE_NAME};",
        explanation="Total patients",
    )
    assert result.row_count == 1
    assert int(result.rows[0]["total"]) == 10000


def test_sql_layer_requires_patients_table() -> None:
    layer = SqlLayer()
    with pytest.raises(SqlValidationError, match="patients"):
        layer.validate_sql("SELECT 1")


def test_sql_layer_blocks_hallucinated_table_names() -> None:
    layer = SqlLayer()
    with pytest.raises(SqlValidationError, match="Unknown table `avg_bmi`"):
        layer.validate_sql(
            "SELECT (SELECT AVG(bmi) FROM patients) AS average_bmi FROM avg_bmi"
        )


def test_sql_layer_allows_cte_over_patients() -> None:
    layer = SqlLayer()
    safe_sql = layer.validate_sql(
        f"""
        WITH avg_bmi AS (
            SELECT AVG(bmi) AS average_bmi FROM {PATIENT_TABLE_NAME}
        )
        SELECT average_bmi FROM avg_bmi
        """
    )
    assert "avg_bmi" in safe_sql.lower()
    result = layer.execute(safe_sql)
    assert result.row_count == 1
    assert "average_bmi" in result.columns
