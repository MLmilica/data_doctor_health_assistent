"""Pydantic schemas for Prediction Agent requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PREDICTION_DISCLAIMER = (
    "Internal prototype output — not clinical advice. "
    "Predictions may use default values for missing optional fields."
)


class PredictionTarget(str, Enum):
    COPD = "copd"
    ALT = "alt"
    BOTH = "both"


class PatientFeatures(BaseModel):
    """Structured patient attributes extracted from natural language."""

    model_config = ConfigDict(extra="ignore")

    bmi: float | None = None
    diet_quality: str | None = None
    exercise_frequency: str | None = None
    income_bracket: str | None = None
    urban: str | int | bool | None = None
    diagnosis_code: str | None = None
    smoker: str | bool | int | None = None
    readmitted: str | int | bool | None = None
    albumin_globulin_ratio: float | None = None
    sex: str | None = None


class PredictionRequest(BaseModel):
    target: PredictionTarget
    features: PatientFeatures
    raw_query: str | None = None


class FeatureMappingNote(BaseModel):
    field: str
    original: Any
    mapped: Any


class PredictionResponse(BaseModel):
    target: str
    prediction: str | float | None
    can_predict: bool
    used_features: dict[str, Any]
    defaults_used: dict[str, dict[str, Any]]
    missing_required: list[str] = Field(default_factory=list)
    class_probabilities: dict[str, float] | None = None
    mapping_notes: list[FeatureMappingNote] = Field(default_factory=list)
    disclaimer: str = PREDICTION_DISCLAIMER
