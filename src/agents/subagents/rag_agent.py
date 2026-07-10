"""RAG Agent — retrieve → corrective grade → synthesize → grounding verify."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState, _merge_state, append_agent_step, set_rag_result
from agents.subagents.prediction_agent import (
    configure_llm_environment,
    require_llm_api_key,
    routing_llm,
    synthesis_llm,
)
from agents.tools.rag_retrieval import retrieve_chunks
from config import settings
from memory.persistence import append_run_step_record
from data.vectorstore import get_vectorstore
from schemas.citation import Citation, RetrievedChunk
from schemas.rag import (
    LLMChunkGrade,
    LLMChunkGradingResult,
    LLMGroundingCheck,
    RAG_DISCLAIMER,
    RAGQueryResult,
)

CHUNK_GRADING_SYSTEM_PROMPT = """You grade whether retrieved document chunks are relevant to the user's question.

Rules:
- Mark relevant=true when the chunk helps answer the question, even partially.
- For summarize/list questions, mark relevant=true if the chunk mentions ANY requested topic (e.g. diet or exercise), even if it does not cover every topic alone.
- Mark relevant=false only for clearly off-topic chunks with no useful overlap.
- Provide a short reason for each grade.
- Grade every chunk id provided in the JSON input.
"""

RAG_SYNTHESIS_SYSTEM_PROMPT = """You answer clinical document questions for an analyst audience.

Rules:
- Use ONLY facts from the provided chunks JSON. Do not invent clinical guidance.
- Cite sources naturally (file + section) when stating facts.
- If chunks are insufficient, say the indexed documents do not contain enough detail.
- End with the disclaimer field verbatim from the JSON.
- Be concise (2-4 short paragraphs or bullets).
"""

RAG_SYNTHESIS_STRICT_SYSTEM_PROMPT = """You answer clinical document questions for an analyst audience.

STRICT MODE: A prior answer was not fully grounded in the sources.

Rules:
- Use ONLY verbatim facts supported by the chunks JSON.
- Do not infer, extrapolate, or add medical advice beyond the text.
- Prefer short bullet points tied to explicit source file + section.
- If the chunks do not support an answer, state that clearly.
- End with the disclaimer field verbatim from the JSON.
"""

GROUNDING_VERIFY_SYSTEM_PROMPT = """You verify whether an assistant answer is grounded in retrieved document chunks.

Rules:
- grounded=true when the main factual claims in the answer are supported by the chunks.
- For summarize answers, partial coverage is acceptable if each stated fact appears in a chunk.
- List unsupported_claims for statements not found in the chunks.
- Ignore the disclaimer line when judging grounding.
- Be strict about invented facts, but allow concise synthesis across multiple chunks.
"""

_TOPIC_KEYWORDS: tuple[str, ...] = (
    "diet",
    "exercise",
    "lifestyle",
    "smoking",
    "nutrition",
    "readmission",
    "medication",
    "treatment",
    "walking",
    "swimming",
    "physical activity",
    "rehabilitation",
    "alcohol",
    "weight",
    "bronchodilator",
)


def build_retrieval_query(user_message: str) -> str:
    """Focus vector search on topic terms for broad summarize/document questions."""
    lowered = user_message.lower()
    topics = _topic_keywords_in_message(user_message)
    if ("summarize" in lowered or "summary" in lowered) and topics:
        return " ".join(topics) + " lifestyle modifications treatment plan recommendations"
    if topics and ("document" in lowered or "clinical" in lowered):
        return " ".join(topics) + " recommendations treatment plan lifestyle"
    return user_message


def _topic_keywords_in_message(message: str) -> list[str]:
    lowered = message.lower()
    return [keyword for keyword in _TOPIC_KEYWORDS if keyword in lowered]


def retrieve_chunks_for_query(user_message: str, *, k: int | None = None) -> list[RetrievedChunk]:
    """Retrieve top-k chunks from Chroma using a focused search query when helpful."""
    search_query = build_retrieval_query(user_message)
    return retrieve_chunks(search_query, k=k or settings.rag_retrieve_k)


def grade_chunks(user_message: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Corrective RAG: keep only chunks the LLM marks as relevant."""
    if not chunks:
        return []

    configure_llm_environment()
    llm = routing_llm().with_structured_output(LLMChunkGradingResult)
    payload = {
        "user_message": user_message,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "section_name": chunk.section_name,
                "content": chunk.content,
            }
            for chunk in chunks
        ],
    }
    result = llm.invoke(
        [
            SystemMessage(content=CHUNK_GRADING_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )
    grading = _parse_grading_result(result)
    grades_by_id = {grade.chunk_id: grade for grade in grading.grades}

    graded_chunks: list[RetrievedChunk] = []
    for chunk in chunks:
        grade = grades_by_id.get(chunk.chunk_id)
        if grade is None:
            continue
        updated = chunk.model_copy(
            update={
                "relevant": grade.relevant,
                "relevance_reason": grade.reason or None,
            }
        )
        if grade.relevant:
            graded_chunks.append(updated)

    if not graded_chunks:
        graded_chunks = _keyword_overlap_fallback(
            user_message,
            chunks,
            max_chunks=settings.rag_top_k,
        )
    return graded_chunks[: settings.rag_top_k]


def _keyword_overlap_fallback(
    user_message: str,
    chunks: list[RetrievedChunk],
    *,
    max_chunks: int,
) -> list[RetrievedChunk]:
    """Keep top chunks that mention question topics when the LLM grader rejects all."""
    topics = _topic_keywords_in_message(user_message)
    if not topics:
        return []

    ranked: list[tuple[int, float, RetrievedChunk]] = []
    for chunk in chunks:
        text = chunk.content.lower()
        hits = sum(1 for topic in topics if topic in text)
        if hits > 0:
            ranked.append((hits, chunk.score, chunk))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        chunk.model_copy(
            update={
                "relevant": True,
                "relevance_reason": "Keyword overlap fallback after strict grading.",
            }
        )
        for _, _, chunk in ranked[:max_chunks]
    ]


def synthesize_rag_answer(
    user_message: str,
    chunks: list[RetrievedChunk],
    *,
    strict: bool = False,
) -> str:
    """LLM synthesis from filtered chunks only."""
    configure_llm_environment()
    llm = synthesis_llm()
    facts = _rag_synthesis_payload(user_message, chunks)
    system_prompt = RAG_SYNTHESIS_STRICT_SYSTEM_PROMPT if strict else RAG_SYNTHESIS_SYSTEM_PROMPT
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(facts, indent=2)),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def verify_grounding(
    user_message: str,
    answer: str,
    chunks: list[RetrievedChunk],
) -> LLMGroundingCheck:
    """Self-RAG lite: verify the answer against the same chunks used for synthesis."""
    configure_llm_environment()
    llm = routing_llm().with_structured_output(LLMGroundingCheck)
    payload = {
        "user_message": user_message,
        "answer": answer,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "section_name": chunk.section_name,
                "content": chunk.content,
            }
            for chunk in chunks
        ],
    }
    result = llm.invoke(
        [
            SystemMessage(content=GROUNDING_VERIFY_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, indent=2)),
        ]
    )
    return _parse_grounding_check(result)


def not_found_response() -> str:
    return (
        "I could not find relevant information in the indexed clinical documents for that question. "
        f"{RAG_DISCLAIMER}"
    )


def safe_fallback_response(check: LLMGroundingCheck) -> str:
    unsupported = "; ".join(check.unsupported_claims) if check.unsupported_claims else ""
    detail = f" Unsupported claims: {unsupported}." if unsupported else ""
    return (
        "I cannot provide a fully document-grounded answer for that question based on the indexed sources."
        f"{detail} "
        f"{RAG_DISCLAIMER}"
    )


def build_rag_query_result(
    *,
    user_message: str,
    retrieved_chunks: list[RetrievedChunk],
    relevant_chunks: list[RetrievedChunk],
    grounded: bool,
    grounding_retry_count: int,
) -> RAGQueryResult:
    citations = [Citation.from_chunk(chunk) for chunk in relevant_chunks]
    return RAGQueryResult(
        user_message=user_message,
        retrieved_count=len(retrieved_chunks),
        relevant_count=len(relevant_chunks),
        citations=citations,
        chunks_used=relevant_chunks,
        grounded=grounded,
        grounding_retry_count=grounding_retry_count,
    )


def run_rag_agent(state: AgentState) -> AgentState:
    """LangGraph node: retrieval pipeline with corrective grading and grounding verify."""
    started = time.perf_counter()
    user_message = state.get("user_message", "")
    llm_model = f"{settings.llm_model_routing}+{settings.llm_model_synthesis}"
    retrieved_chunks: list[RetrievedChunk] = []
    relevant_chunks: list[RetrievedChunk] = []

    try:
        require_llm_api_key()
        vectorstore = get_vectorstore()
        if not vectorstore.is_indexed():
            response_text = (
                "The document index is empty. Run "
                "`uv run python scripts/index_documents.py` after adding files to data/documents/."
            )
            updated = _merge_state(
                state,
                response_text=response_text,
                error="rag_index_missing",
                llm_model=llm_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return append_agent_step(append_run_step_record(updated), "rag")

        retrieved_chunks = retrieve_chunks_for_query(user_message)
        relevant_chunks = grade_chunks(user_message, retrieved_chunks)

        if len(relevant_chunks) < settings.rag_min_relevant_chunks:
            rag_result = build_rag_query_result(
                user_message=user_message,
                retrieved_chunks=retrieved_chunks,
                relevant_chunks=[],
                grounded=False,
                grounding_retry_count=0,
            )
            state = set_rag_result(state, rag_result)
            updated = _merge_state(
                state,
                response_text=not_found_response(),
                llm_model=llm_model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return append_agent_step(append_run_step_record(updated), "rag")

        answer = synthesize_rag_answer(user_message, relevant_chunks)
        grounding = verify_grounding(user_message, answer, relevant_chunks)
        retry_count = 0

        if not grounding.grounded and settings.rag_max_grounding_retries > 0:
            retry_count = 1
            answer = synthesize_rag_answer(user_message, relevant_chunks, strict=True)
            grounding = verify_grounding(user_message, answer, relevant_chunks)

        if not grounding.grounded:
            response_text = safe_fallback_response(grounding)
            grounded = False
        else:
            response_text = answer
            grounded = True

        rag_result = build_rag_query_result(
            user_message=user_message,
            retrieved_chunks=retrieved_chunks,
            relevant_chunks=relevant_chunks,
            grounded=grounded,
            grounding_retry_count=retry_count,
        )
        state = set_rag_result(state, rag_result)
        prior_latency = state.get("latency_ms") or 0.0
        updated = _merge_state(
            state,
            response_text=response_text,
            llm_model=llm_model,
            latency_ms=round(prior_latency + (time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(append_run_step_record(updated), "rag")

    except FileNotFoundError as exc:
        updated = _merge_state(
            state,
            response_text=f"Clinical documents are not available: {exc}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(append_run_step_record(updated), "rag")
    except Exception as exc:
        updated = _merge_state(
            state,
            response_text=f"Document search failed: {exc}",
            error=str(exc),
            llm_model=llm_model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return append_agent_step(append_run_step_record(updated), "rag")


def _rag_synthesis_payload(user_message: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "user_message": user_message,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "section_name": chunk.section_name,
                "content": chunk.content,
                "score": chunk.score,
            }
            for chunk in chunks
        ],
        "disclaimer": RAG_DISCLAIMER,
    }


def _parse_grading_result(result: Any) -> LLMChunkGradingResult:
    if isinstance(result, LLMChunkGradingResult):
        return result
    return LLMChunkGradingResult.model_validate(result)


def _parse_grounding_check(result: Any) -> LLMGroundingCheck:
    if isinstance(result, LLMGroundingCheck):
        return result
    return LLMGroundingCheck.model_validate(result)
