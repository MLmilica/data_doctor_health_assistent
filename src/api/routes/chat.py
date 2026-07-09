"""Chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import invoke_chat, is_llm_configured
from schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not is_llm_configured():
        raise HTTPException(
            status_code=503,
            detail="LLM API key is not configured. Set provider credentials in .env.",
        )

    try:
        return invoke_chat(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
