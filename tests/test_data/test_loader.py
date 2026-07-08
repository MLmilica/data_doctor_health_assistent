"""Tests for DuckDB patient data loader."""

from data.loader import PatientDataLoader
from data.schema_registry import PATIENT_TABLE_NAME, get_column_names


def test_loader_row_count() -> None:
    with PatientDataLoader() as loader:
        assert loader.row_count == 10000


def test_loader_column_names_match_schema() -> None:
    with PatientDataLoader() as loader:
        assert loader.get_column_names() == get_column_names()


def test_loader_sql_smoker_count() -> None:
    with PatientDataLoader() as loader:
        result = loader.query(
            f"""
            SELECT COUNT(*) AS smoker_count
            FROM {PATIENT_TABLE_NAME}
            WHERE smoker = 'Yes'
            """
        )
        assert int(result.iloc[0]["smoker_count"]) == 2961
