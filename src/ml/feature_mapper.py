"""Map informal / LLM-extracted feature values to canonical training values."""

from __future__ import annotations

from typing import Any

from ml.features import ALT_FEATURE_COLS, COPD_FEATURE_COLS, _parse_bool, _parse_int01
from ml.predict import PredictionResult, predict_all, predict_alt, predict_copd
from schemas.prediction import (
    FeatureMappingNote,
    PatientFeatures,
    PredictionRequest,
    PredictionResponse,
    PredictionTarget,
)

# Normalized synonym key (lowercase, stripped) -> canonical enum value
EXERCISE_SYNONYMS: dict[str, str] = {
    "none": "None",
    "no exercise": "None",
    "doesn't exercise": "None",
    "doesnt exercise": "None",
    "never": "None",
    "sedentary": "None",
    "inactive": "None",
    "low": "Low",
    "little": "Low",
    "rarely": "Low",
    "moderate": "Moderate",
    "average": "Moderate",
    "sometimes": "Moderate",
    "medium": "Moderate",
    "high": "High",
    "athlete": "High",
    "sportista": "High",
    "very active": "High",
    "active": "High",
    "often": "High",
    "frequently": "High",
}

DIET_SYNONYMS: dict[str, str] = {
    "poor": "Poor",
    "bad": "Poor",
    "unhealthy": "Poor",
    "loše": "Poor",
    "average": "Average",
    "fair": "Average",
    "ok": "Average",
    "good": "Good",
    "healthy": "Good",
    "dobro": "Good",
}

INCOME_SYNONYMS: dict[str, str] = {
    "low": "Low",
    "poor": "Low",
    "middle": "Middle",
    "mid": "Middle",
    "medium": "Middle",
    "high": "High",
    "rich": "High",
}

SEX_SYNONYMS: dict[str, str] = {
    "male": "Male",
    "man": "Male",
    "m": "Male",
    "muškarac": "Male",
    "muskarac": "Male",
    "female": "Female",
    "woman": "Female",
    "f": "Female",
    "žena": "Female",
    "zena": "Female",
}

URBAN_SYNONYMS: dict[str, int] = {
    "urban": 1,
    "city": 1,
    "center": 1,
    "centre": 1,
    "center of city": 1,
    "centar grada": 1,
    "rural": 0,
    "countryside": 0,
    "village": 0,
}

SMOKER_SYNONYMS: dict[str, bool] = {
    "yes": True,
    "y": True,
    "true": True,
    "smoker": True,
    "pušač": True,
    "pusac": True,
    "no": False,
    "n": False,
    "false": False,
    "non-smoker": False,
    "nonsmoker": False,
    "non smoker": False,
}

READMITTED_SYNONYMS: dict[str, int] = {
    "yes": 1,
    "y": 1,
    "true": 1,
    "readmitted": 1,
    "no": 0,
    "n": 0,
    "false": 0,
    "not readmitted": 0,
}

CANONICAL_CATEGORICAL: dict[str, set[str]] = {
    "exercise_frequency": {"None", "Low", "Moderate", "High"},
    "diet_quality": {"Poor", "Average", "Good"},
    "income_bracket": {"Low", "Middle", "High"},
    "diagnosis_code": {"D1", "D2", "D3", "D4", "D5"},
    "sex": {"Male", "Female"},
}

SYNONYM_TABLES: dict[str, dict[str, Any]] = {
    "exercise_frequency": EXERCISE_SYNONYMS,
    "diet_quality": DIET_SYNONYMS,
    "income_bracket": INCOME_SYNONYMS,
    "sex": SEX_SYNONYMS,
    "urban": URBAN_SYNONYMS,
    "smoker": SMOKER_SYNONYMS,
    "readmitted": READMITTED_SYNONYMS,
}

PREDICTION_FEATURE_FIELDS = sorted(
    set(COPD_FEATURE_COLS + ALT_FEATURE_COLS + ["sex"]),
)


def _normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _map_with_synonyms(field: str, value: Any) -> tuple[Any, bool]:
    if value is None:
        return None, False

    if field in {"bmi", "albumin_globulin_ratio"}:
        try:
            mapped = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for '{field}': {value!r}") from exc
        return mapped, mapped != value

    if field == "smoker":
        if isinstance(value, bool):
            return value, False
        parsed = _parse_bool(value)
        if parsed is not None:
            return parsed, parsed != value
        mapped = SMOKER_SYNONYMS.get(_normalize_key(str(value)))
        if mapped is not None:
            return mapped, True
        raise ValueError(f"Unrecognized smoker value: {value!r}")

    if field in {"urban", "readmitted"}:
        if isinstance(value, bool):
            mapped = int(value)
            return mapped, mapped != value
        parsed = _parse_int01(value)
        if parsed is not None:
            return parsed, parsed != value
        table = URBAN_SYNONYMS if field == "urban" else READMITTED_SYNONYMS
        mapped = table.get(_normalize_key(str(value)))
        if mapped is not None:
            return mapped, True
        raise ValueError(f"Unrecognized {field} value: {value!r}")

    text = str(value).strip()
    canonical = CANONICAL_CATEGORICAL.get(field)
    if canonical and text in canonical:
        return text, False

    table = SYNONYM_TABLES.get(field, {})
    mapped = table.get(_normalize_key(text))
    if mapped is not None:
        return mapped, True

    if field == "diagnosis_code":
        upper = text.upper()
        if upper in CANONICAL_CATEGORICAL["diagnosis_code"]:
            return upper, upper != text
        return text, False

    if canonical:
        title = text.title()
        if title in canonical:
            return title, title != text
        raise ValueError(f"Unrecognized {field} value: {value!r}")

    return text, text != value


def map_patient_features(
    features: PatientFeatures | dict[str, Any],
) -> tuple[dict[str, Any], list[FeatureMappingNote]]:
    """Normalize PatientFeatures into raw values understood by `predict.py`."""
    payload = features.model_dump(exclude_none=True) if isinstance(features, PatientFeatures) else {
        key: value for key, value in features.items() if value is not None
    }

    mapped: dict[str, Any] = {}
    notes: list[FeatureMappingNote] = []

    for field, value in payload.items():
        if field not in PREDICTION_FEATURE_FIELDS:
            continue

        new_value, changed = _map_with_synonyms(field, value)
        if new_value is None:
            continue

        mapped[field] = new_value
        if changed:
            notes.append(FeatureMappingNote(field=field, original=value, mapped=new_value))

    return mapped, notes


def prediction_result_to_response(
    result: PredictionResult,
    *,
    mapping_notes: list[FeatureMappingNote] | None = None,
) -> PredictionResponse:
    return PredictionResponse(
        target=result.target,
        prediction=result.prediction,
        can_predict=result.can_predict,
        used_features=result.used_features,
        defaults_used=result.defaults_used,
        missing_required=result.missing_required,
        class_probabilities=result.class_probabilities,
        mapping_notes=mapping_notes or [],
    )


def run_prediction(request: PredictionRequest) -> PredictionResponse | dict[str, PredictionResponse]:
    """Map features, run inference, and return structured API/agent response."""
    mapped_features, notes = map_patient_features(request.features)

    if request.target == PredictionTarget.COPD:
        return prediction_result_to_response(predict_copd(mapped_features), mapping_notes=notes)

    if request.target == PredictionTarget.ALT:
        return prediction_result_to_response(predict_alt(mapped_features), mapping_notes=notes)

    results = predict_all(mapped_features)
    return {
        "copd": prediction_result_to_response(results["copd"], mapping_notes=notes),
        "alt": prediction_result_to_response(results["alt"], mapping_notes=notes),
    }
