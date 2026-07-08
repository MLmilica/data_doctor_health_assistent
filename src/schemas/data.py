"""SQL, statistics, and dataset profile schemas."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    description: str
    allowed_values: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    is_target: bool = False


class DatasetSchema(BaseModel):
    table_name: str = "patients"
    row_count: int
    columns: list[ColumnSchema]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    count: int
    null_count: int
    min: float | int | str | None = None
    max: float | int | str | None = None
    mean: float | None = None
    mode: str | int | float | None = None
    unique_count: int | None = None
    unique_values: list[str] | None = None


class DataProfile(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    row_count: int
    columns: list[ColumnProfile]
