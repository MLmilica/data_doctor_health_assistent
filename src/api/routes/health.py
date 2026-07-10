"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import are_ml_models_loaded, get_document_index_status, is_llm_configured
from observability.langsmith import is_langsmith_enabled
from schemas.chat import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    llm_configured = is_llm_configured()
    ml_models_loaded = are_ml_models_loaded()
    documents_indexed, document_chunk_count = get_document_index_status()

    if llm_configured and ml_models_loaded and documents_indexed:
        status = "ok"
        detail = None
    elif llm_configured or ml_models_loaded or documents_indexed:
        status = "degraded"
        missing: list[str] = []
        if not llm_configured:
            missing.append("LLM API key")
        if not ml_models_loaded:
            missing.append("ML model artifacts")
        if not documents_indexed:
            missing.append("indexed clinical documents")
        detail = f"Missing: {', '.join(missing)}"
    else:
        status = "error"
        detail = "Missing LLM API key, ML model artifacts, and document index"

    return HealthResponse(
        status=status,
        llm_configured=llm_configured,
        ml_models_loaded=ml_models_loaded,
        documents_indexed=documents_indexed,
        document_chunk_count=document_chunk_count,
        langsmith_tracing=is_langsmith_enabled(),
        detail=detail,
    )
