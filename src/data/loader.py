"""Load patient CSV into DuckDB for SQL queries."""

from pathlib import Path

import duckdb
import pandas as pd

from config import settings
from data.schema_registry import PATIENT_TABLE_NAME


class PatientDataLoader:
    """Loads data/raw/patient_data.csv into an in-memory DuckDB table."""

    def __init__(
        self,
        csv_path: Path | None = None,
        database_path: str = ":memory:",
    ) -> None:
        self.csv_path = csv_path or settings.patient_csv_path
        self.connection = duckdb.connect(database_path)
        self._load_csv()

    def _load_csv(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Patient CSV not found: {self.csv_path}")

        self.connection.execute(
            f"""
            CREATE OR REPLACE TABLE {PATIENT_TABLE_NAME} AS
            SELECT * FROM read_csv_auto(?)
            """,
            [str(self.csv_path)],
        )

    @property
    def row_count(self) -> int:
        result = self.connection.execute(
            f"SELECT COUNT(*) FROM {PATIENT_TABLE_NAME}"
        ).fetchone()
        if result is None:
            return 0
        return int(result[0])

    def query(self, sql: str) -> pd.DataFrame:
        return self.connection.execute(sql).df()

    def query_to_records(self, sql: str) -> list[dict]:
        return self.query(sql).to_dict(orient="records")

    def get_dataframe(self) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {PATIENT_TABLE_NAME}")

    def get_column_names(self) -> list[str]:
        rows = self.connection.execute(
            f"DESCRIBE {PATIENT_TABLE_NAME}"
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PatientDataLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
