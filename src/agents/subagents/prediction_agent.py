"""Prediction Agent — LLM extract → ML inference → LLM synthesis."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agents.state import (
    _merge_state,
    AgentState,
    append_agent_step,
    get_conversation_history,
    get_session_facts,
    set_extraction,
    set_prediction_result,
)
from config import settings
from memory.context import build_prediction_extraction_prompt, merge_patient_features
from ml.feature_mapper import run_prediction
from schemas.prediction import (
    LLMPredictionExtraction,
    PREDICTION_DISCLAIMER,
    PredictionRequest,
    PredictionResponse,
    PredictionTarget,
)

EXTRACTION_SYSTEM_PROMPT = """You extract structured patient features for ML prediction from user messages.

The orchestrator has already routed this message to the prediction agent.
Your job is feature extraction only — not intent classification.

Targets:
- copd: chronic obstructive pulmonary disease severity class (A/B/C/D)
- alt: alanine aminotransferase lab value (numeric regression)
- both: when the user explicitly wants both predictions

Feature fields (use null when not mentioned):
bmi, diet_quality, exercise_frequency, income_bracket, urban, diagnosis_code,
smoker, readmitted, albumin_globulin_ratio, sex

Rules:
- Always set is_prediction_request=true.
- Populate target when inferable; use null only when the user did not specify COPD, ALT, or both.
- Populate features mentioned in the message; leave others null.
- For follow-up messages, update only fields the user changed; session context lists prior features.
- Use assistant_message only when target is unclear and you need clarification.
"""

SYNTHESIS_SYSTEM_PROMPT = """You format clinical analytics prototype replies for an analyst audience.

Rules:
- Use ONLY the facts in the user-provided JSON. Do not invent or change any numbers or class labels.
- If missing_required is non-empty, politely ask the user to provide those fields.
- If defaults_used is non-empty, mention which optional fields were filled from dataset defaults.
- Mention top_global_factors briefly when present (model-level drivers, not patient-specific SHAP).
- End with the disclaimer field verbatim from the JSON.
- Be concise (2-5 short paragraphs or bullets).
"""


def _provider_api_key_env() -> str:
    return "ANTHROPIC_API_KEY" if settings.llm_provider == "anthropic" else "OPENAI_API_KEY"


def require_llm_api_key() -> None:
    env_var = _provider_api_key_env()
    configured_key = settings.anthropic_api_key if settings.llm_provider == "anthropic" else settings.openai_api_key
    if not configured_key and not os.environ.get(env_var):
        raise ValueError(
            f"{env_var} is not configured. Set it in .env to use the prediction agent."
        )


def configure_llm_environment() -> None:
    """Sync configured provider key into expected env var for LangChain integrations."""
    require_llm_api_key()
    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        return
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key


def _model_name_for_provider(model_name: str) -> str:
    if ":" in model_name:
        return model_name
    return f"{settings.llm_provider}:{model_name}"


def _chat_model(*, model_name: str, temperature: float) -> BaseChatModel:
    # LangChain docs: use init_chat_model and provider-prefixed model IDs.
    return init_chat_model(_model_name_for_provider(model_name), temperature=temperature)


def _extraction_llm() -> BaseChatModel:
    return _chat_model(model_name=settings.llm_model_routing, temperature=0)


def routing_llm() -> BaseChatModel:
    """Low-temperature model shared by orchestrator routing and prediction extraction."""
    return _extraction_llm()


def synthesis_llm() -> BaseChatModel:
    """Higher-level model for analyst-facing prose synthesis."""
    return _chat_model(model_name=settings.llm_model_synthesis, temperature=0.2)


def _synthesis_llm() -> BaseChatModel:
    return synthesis_llm()


def _parse_structured_extraction(result: Any) -> LLMPredictionExtraction:
    if isinstance(result, LLMPredictionExtraction):
        return result
    return LLMPredictionExtraction.model_validate(result)


def extract_with_llm(
    user_message: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    session_facts: dict[str, Any] | None = None,
) -> LLMPredictionExtraction:
    """LLM #1: natural language → structured extraction schema."""
    configure_llm_environment()
    llm = _extraction_llm().with_structured_output(LLMPredictionExtraction)
    prompt = build_prediction_extraction_prompt(
        user_message,
        conversation_history=conversation_history,
        session_facts=session_facts,
    )
    result = llm.invoke(
        [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    return _parse_structured_extraction(result)


def synthesize_response_text(facts: dict[str, Any]) -> str:
    """LLM #2: polish prose using read-only facts JSON (numbers stay from ML)."""
    configure_llm_environment()
    llm = _synthesis_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(facts, indent=2)),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def load_top_global_factors(target: str, *, limit: int = 3) -> list[str]:
    """Read offline SHAP summary JSON (no runtime SHAP computation)."""
    path = settings.artifacts_dir / "insights" / f"{target}_shap_summary.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    top_features = payload.get("top_features", [])
    return [str(item.get("feature", "")) for item in top_features[:limit] if item.get("feature")]


def _top_factors_for_result(
    result: PredictionResponse | dict[str, PredictionResponse],
) -> dict[str, list[str]]:
    if isinstance(result, dict):
        return {key: load_top_global_factors(key) for key in result}
    return {result.target: load_top_global_factors(result.target)}


def _facts_payload(
    *,
    user_message: str,
    result: PredictionResponse | dict[str, PredictionResponse],
    top_global_factors: dict[str, list[str]],
) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            "user_message": user_message,
            "predictions": {key: value.model_dump() for key, value in result.items()},
            "top_global_factors": top_global_factors,
            "disclaimer": PREDICTION_DISCLAIMER,
        }
    return {
        "user_message": user_message,
        **result.model_dump(),
        "top_global_factors": top_global_factors.get(result.target, []),
    }


def format_fallback_response(
    result: PredictionResponse | dict[str, PredictionResponse],
    *,
    top_global_factors: dict[str, list[str]] | None = None,
) -> str:
    """Deterministic fallback if synthesis LLM fails."""
    factors = top_global_factors or _top_factors_for_result(result)
    lines = ["Prediction results (prototype):"]

    def _append_single(single: PredictionResponse, label: str | None = None) -> None:
        prefix = f"{label.upper()}: " if label else ""
        if not single.can_predict:
            missing = ", ".join(single.missing_required) or "unknown"
            lines.append(f"{prefix}Cannot predict yet — missing required field(s): {missing}.")
            return
        lines.append(f"{prefix}Prediction = {single.prediction}")
        if single.defaults_used:
            defaults = ", ".join(single.defaults_used)
            lines.append(f"  Defaults applied: {defaults}")
        factor_list = factors.get(single.target, [])
        if factor_list:
            lines.append(f"  Top global factors: {', '.join(factor_list)}")

    if isinstance(result, dict):
        for key, single in result.items():
            _append_single(single, key)
    else:
        _append_single(result)

    lines.append(PREDICTION_DISCLAIMER)
    return "\n".join(lines)


def run_prediction_agent(state: AgentState) -> AgentState:
    """
    LangGraph node: extract → infer → synthesize → update AgentState.

    Structured ChatResponse.prediction is built later from prediction_result in state,
    not from synthesis prose.
    """
    started = time.perf_counter()
    user_message = state.get("user_message", "")
    llm_models = f"{settings.llm_model_routing}+{settings.llm_model_synthesis}"

    try:
        require_llm_api_key()
        session_facts = get_session_facts(state)
        extraction = extract_with_llm(
            user_message,
            conversation_history=get_conversation_history(state),
            session_facts=session_facts.model_dump(),
        )
        merged_features = merge_patient_features(
            session_facts.last_features or None,
            extraction.features,
        )
        extraction = extraction.model_copy(update={"features": merged_features})
        if extraction.target is None and session_facts.last_target:
            try:
                extraction = extraction.model_copy(
                    update={"target": PredictionTarget(session_facts.last_target)},
                )
            except ValueError:
                pass
        state = set_extraction(state, extraction)

        if extraction.target is None:
            response_text = (
                extraction.assistant_message
                or "Please specify whether you want a COPD, ALT, or both predictions."
            )
            updated = _merge_state(
                state,
                response_text=response_text,
                llm_model=llm_models,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return append_agent_step(updated, "prediction")

        request = PredictionRequest(
            target=extraction.target,
            features=extraction.features,
            raw_query=user_message,
        )
        prediction_result = run_prediction(request)
        top_global_factors = _top_factors_for_result(prediction_result)
        facts = _facts_payload(
            user_message=user_message,
            result=prediction_result,
            top_global_factors=top_global_factors,
        )

        try:
            response_text = synthesize_response_text(facts)
        except Exception:
            response_text = format_fallback_response(
                prediction_result,
                top_global_factors=top_global_factors,
            )

        state = set_prediction_result(state, prediction_result)
        updated = _merge_state(
            state,
            response_text=response_text,
            llm_model=llm_models,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            top_global_factors=top_global_factors,
        )
        return append_agent_step(updated, "prediction")

    except Exception as exc:
        return _merge_state(
            state,
            error=str(exc),
            response_text=str(exc),
            llm_model=llm_models,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
