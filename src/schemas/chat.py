"""Chat API schemas — boundary between Streamlit UI and FastAPI/LangGraph."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from schemas.prediction import FeatureMappingNote, PredictionResponse, PREDICTION_DISCLAIMER

DEFAULT_USER_ID = "default-user"


class ChatRequest(BaseModel):
    """Natural-language message routed through LangGraph + LLM extraction."""

    message: str = Field(min_length=1, description="User message in natural language.")
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Conversation session identifier (future checkpoint thread).",
    )
    user_id: str = Field(
        default=DEFAULT_USER_ID,
        description="Caller identity for logging and future per-user memory.",
    )


class ChatPredictionDetails(BaseModel):
    """Structured prediction output exposed to the UI."""

    target: str
    prediction: str | float | None
    can_predict: bool
    used_features: dict[str, Any] = Field(default_factory=dict)
    defaults_used: dict[str, dict[str, Any]] = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)
    class_probabilities: dict[str, float] | None = None
    mapping_notes: list[FeatureMappingNote] = Field(default_factory=list)
    disclaimer: str = PREDICTION_DISCLAIMER
    top_global_factors: list[str] = Field(
        default_factory=list,
        description="Optional global drivers from offline SHAP/importance artifacts.",
    )

    @classmethod
    def from_prediction_response(
        cls,
        response: PredictionResponse,
        *,
        top_global_factors: list[str] | None = None,
    ) -> ChatPredictionDetails:
        return cls(
            target=response.target,
            prediction=response.prediction,
            can_predict=response.can_predict,
            used_features=response.used_features,
            defaults_used=response.defaults_used,
            missing_required=response.missing_required,
            class_probabilities=response.class_probabilities,
            mapping_notes=response.mapping_notes,
            disclaimer=response.disclaimer,
            top_global_factors=top_global_factors or [],
        )


class ChatAgentMetadata(BaseModel):
    """Runtime metadata for observability and transparency in the UI sidebar."""

    agent: str = "prediction"
    extraction_method: Literal["llm"] = "llm"
    llm_model: str | None = None
    latency_ms: float | None = None
    routed_to: str | None = "prediction"


class ChatResponse(BaseModel):
    """API response returned to Streamlit — UI does not know about internal agents."""

    text: str = Field(description="Human-readable assistant reply.")
    session_id: str
    prediction: ChatPredictionDetails | None = Field(
        default=None,
        description="Present when a single target (copd or alt) was predicted.",
    )
    predictions: dict[str, ChatPredictionDetails] | None = Field(
        default=None,
        description="Present when target=both (keys: copd, alt).",
    )
    metadata: ChatAgentMetadata = Field(default_factory=ChatAgentMetadata)

    @classmethod
    def from_prediction_results(
        cls,
        *,
        text: str,
        session_id: str,
        result: PredictionResponse | dict[str, PredictionResponse],
        llm_model: str | None = None,
        latency_ms: float | None = None,
        top_global_factors: dict[str, list[str]] | None = None,
    ) -> ChatResponse:
        factors = top_global_factors or {}
        metadata = ChatAgentMetadata(llm_model=llm_model, latency_ms=latency_ms)

        if isinstance(result, dict):
            predictions = {
                key: ChatPredictionDetails.from_prediction_response(
                    value,
                    top_global_factors=factors.get(key, []),
                )
                for key, value in result.items()
            }
            return cls(
                text=text,
                session_id=session_id,
                predictions=predictions,
                metadata=metadata,
            )

        return cls(
            text=text,
            session_id=session_id,
            prediction=ChatPredictionDetails.from_prediction_response(
                result,
                top_global_factors=factors.get(result.target, []),
            ),
            metadata=metadata,
        )


class HealthResponse(BaseModel):
    """GET /health — readiness for UI and deploy checks."""

    status: Literal["ok", "degraded", "error"] = "ok"
    api: Literal["up"] = "up"
    llm_configured: bool = Field(
        description="True when the configured LLM provider API key is set.",
    )
    ml_models_loaded: bool = Field(
        default=False,
        description="True when COPD/ALT model artifacts are available on disk.",
    )
    detail: str | None = None
