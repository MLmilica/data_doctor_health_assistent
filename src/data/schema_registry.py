"""Patient dataset column definitions for agents and SQL generation."""

from schemas.data import ColumnSchema, DatasetSchema

PATIENT_TABLE_NAME = "patients"

COLUMN_DEFINITIONS: list[ColumnSchema] = [
    ColumnSchema(
        name="patient_id",
        dtype="str",
        description="Unique anonymized patient identifier",
    ),
    ColumnSchema(
        name="age",
        dtype="int",
        description="Patient age in years",
        min_value=0,
        max_value=120,
    ),
    ColumnSchema(
        name="sex",
        dtype="category",
        description="Patient sex",
        allowed_values=["Male", "Female"],
    ),
    ColumnSchema(
        name="bmi",
        dtype="float",
        description="Body Mass Index",
    ),
    ColumnSchema(
        name="smoker",
        dtype="category",
        description="Whether the patient is a smoker",
        allowed_values=["Yes", "No"],
    ),
    ColumnSchema(
        name="diagnosis_code",
        dtype="category",
        description="Primary diagnosis code",
        allowed_values=["D1", "D2", "D3", "D4", "D5"],
    ),
    ColumnSchema(
        name="medication_count",
        dtype="int",
        description="Number of medications the patient takes",
        min_value=0,
    ),
    ColumnSchema(
        name="days_hospitalized",
        dtype="int",
        description="Number of days hospitalized",
        min_value=0,
    ),
    ColumnSchema(
        name="readmitted",
        dtype="int",
        description="Readmission flag (0 = no, 1 = yes)",
        allowed_values=["0", "1"],
    ),
    ColumnSchema(
        name="last_lab_glucose",
        dtype="float",
        description="Most recent lab glucose measurement",
    ),
    ColumnSchema(
        name="exercise_frequency",
        dtype="category",
        description="Exercise frequency level",
        allowed_values=["None", "Low", "Moderate", "High"],
    ),
    ColumnSchema(
        name="diet_quality",
        dtype="category",
        description="Self-reported diet quality",
        allowed_values=["Poor", "Average", "Good"],
    ),
    ColumnSchema(
        name="income_bracket",
        dtype="category",
        description="Income bracket",
        allowed_values=["Low", "Middle", "High"],
    ),
    ColumnSchema(
        name="education_level",
        dtype="category",
        description="Education level",
        allowed_values=["Primary", "Secondary", "Tertiary"],
    ),
    ColumnSchema(
        name="urban",
        dtype="int",
        description="Urban residence flag (0 = rural, 1 = urban)",
        allowed_values=["0", "1"],
    ),
    ColumnSchema(
        name="albumin_globulin_ratio",
        dtype="float",
        description="Albumin to globulin ratio",
    ),
    ColumnSchema(
        name="chronic_obstructive_pulmonary_disease",
        dtype="category",
        description="COPD severity class (prediction target)",
        allowed_values=["A", "B", "C", "D"],
        is_target=True,
    ),
    ColumnSchema(
        name="alanine_aminotransferase",
        dtype="float",
        description="ALT lab value (prediction target)",
        is_target=True,
    ),
]


def get_column_names() -> list[str]:
    return [column.name for column in COLUMN_DEFINITIONS]


def get_feature_columns() -> list[str]:
    return [column.name for column in COLUMN_DEFINITIONS if not column.is_target]


def get_target_columns() -> list[str]:
    return [column.name for column in COLUMN_DEFINITIONS if column.is_target]


def get_dataset_schema(row_count: int) -> DatasetSchema:
    return DatasetSchema(
        table_name=PATIENT_TABLE_NAME,
        row_count=row_count,
        columns=COLUMN_DEFINITIONS,
    )


def get_column_schema(name: str) -> ColumnSchema | None:
    for column in COLUMN_DEFINITIONS:
        if column.name == name:
            return column
    return None
