"""Streamlit UI — chat via FastAPI and direct ML form (no LLM)."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import settings  # noqa: E402
from data.profile import load_data_profile  # noqa: E402
from ml.features import (  # noqa: E402
    ALT_FEATURE_COLS,
    ALT_NUM_COLS,
    ALT_OPTIONAL_COLS,
    ALT_REQUIRED_COLS,
    COPD_FEATURE_COLS,
    COPD_OPTIONAL_COLS,
    COPD_REQUIRED_COLS,
    _default_for_column,
)
from schemas.chat import ChatDataQueryDetails, ChatChartDetails, ChatInsightDetails, ChatPredictionDetails, ChatRAGDetails, ChatResponse, HealthResponse  # noqa: E402
from schemas.data import DataProfile  # noqa: E402
from schemas.prediction import PatientFeatures, PredictionRequest, PredictionTarget  # noqa: E402

PAGE_TITLE = "Data Doctor"
EMPTY_SELECT_LABEL = "— select —"
TURN_HISTORY_LIMIT = 8
NUMERIC_FIELDS = set(ALT_NUM_COLS) | {"urban", "readmitted"}
CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "diet_quality": ["Poor", "Average", "Good"],
    "exercise_frequency": ["None", "Low", "Moderate", "High"],
    "income_bracket": ["Low", "Middle", "High"],
    "diagnosis_code": ["D1", "D2", "D3", "D4", "D5"],
}


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = settings.api_base_url
    if "health" not in st.session_state:
        st.session_state.health = None
    if "last_turn" not in st.session_state:
        st.session_state.last_turn = None
    if "turn_history" not in st.session_state:
        st.session_state.turn_history = []


def fetch_health(api_base_url: str) -> HealthResponse | None:
    try:
        response = httpx.get(f"{api_base_url.rstrip('/')}/health", timeout=10.0)
        response.raise_for_status()
        return HealthResponse.model_validate(response.json())
    except httpx.HTTPError:
        return None


def post_chat(api_base_url: str, *, message: str, session_id: str) -> ChatResponse:
    response = httpx.post(
        f"{api_base_url.rstrip('/')}/chat",
        json={"message": message, "session_id": session_id},
        timeout=120.0,
    )
    response.raise_for_status()
    return ChatResponse.model_validate(response.json())


def _render_prediction_details(details: ChatPredictionDetails, *, label: str | None = None) -> None:
    title = label.upper() if label else details.target.upper()
    st.markdown(f"**{title} prediction:** `{details.prediction}`")
    st.caption(f"can_predict = {details.can_predict}")

    if details.missing_required:
        with st.expander("Missing required fields", expanded=True):
            st.write(", ".join(details.missing_required))

    if details.defaults_used:
        with st.expander("Defaults applied"):
            st.json(details.defaults_used)

    if details.used_features:
        with st.expander("Used features"):
            st.json(details.used_features)

    if details.class_probabilities:
        with st.expander("Class probabilities"):
            st.json(details.class_probabilities)

    if details.top_global_factors:
        with st.expander("Top global factors"):
            st.write(", ".join(details.top_global_factors))

    st.caption(details.disclaimer)


def _routing_metadata_parts(response: ChatResponse) -> list[str]:
    meta = response.metadata
    parts: list[str] = []
    if meta.routed_to:
        parts.append(f"routed: {meta.routed_to}")
    if meta.route_confidence is not None:
        parts.append(f"confidence: {meta.route_confidence:.0%}")
    if meta.route_source:
        parts.append(f"via {meta.route_source}")
    if meta.guardrail_blocked:
        parts.append("guardrail blocked")
    return parts


def _detect_data_tool(response: ChatResponse) -> str | None:
    """Infer data-agent sub-path from structured response fields."""
    if response.chart is not None:
        return f"chart ({response.chart.chart_type})"
    if response.insight is not None:
        return f"insight ({response.insight.target})"
    if response.data_query is not None:
        return "sql"
    return None


def _truncate_text(text: str, *, limit: int = 72) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _build_turn_snapshot(*, user_message: str, response: ChatResponse) -> dict[str, Any]:
    """Compact observability payload for the sidebar (UI-only, no API changes)."""
    meta = response.metadata
    data_tool = _detect_data_tool(response)
    snapshot: dict[str, Any] = {
        "user_message": _truncate_text(user_message),
        "routed_to": meta.routed_to,
        "data_tool": data_tool,
        "route_confidence": meta.route_confidence,
        "route_source": meta.route_source,
        "guardrail_blocked": meta.guardrail_blocked,
        "llm_model": meta.llm_model,
        "latency_ms": meta.latency_ms,
    }

    if response.chart is not None:
        chart = response.chart
        snapshot["chart"] = {
            "chart_type": chart.chart_type,
            "title": chart.title,
            "x_column": chart.x_column,
            "y_column": chart.y_column,
            "group_by": chart.group_by,
            "row_count": chart.row_count,
            "sql": chart.sql,
        }
    elif response.insight is not None:
        insight = response.insight
        snapshot["insight"] = {
            "target": insight.target,
            "source": insight.source,
            "top_features": [
                str(item.get("feature", ""))
                for item in insight.top_features[:5]
                if item.get("feature")
            ],
        }
    elif response.data_query is not None:
        query = response.data_query
        snapshot["data_query"] = {
            "row_count": query.row_count,
            "truncated": query.truncated,
            "columns": query.columns,
            "sql": query.sql,
            "explanation": query.explanation,
        }

    if response.prediction is not None:
        prediction = response.prediction
        snapshot["prediction"] = {
            "target": prediction.target,
            "can_predict": prediction.can_predict,
            "prediction": prediction.prediction,
            "top_global_factors": prediction.top_global_factors,
        }
    elif response.predictions:
        snapshot["predictions"] = {
            key: {
                "can_predict": details.can_predict,
                "prediction": details.prediction,
            }
            for key, details in response.predictions.items()
        }

    if response.rag is not None:
        rag = response.rag
        snapshot["rag"] = {
            "retrieved_count": rag.retrieved_count,
            "relevant_count": rag.relevant_count,
            "grounded": rag.grounded,
            "grounding_retry_count": rag.grounding_retry_count,
            "citation_count": len(rag.citations),
        }

    return snapshot


def _append_turn_history(snapshot: dict[str, Any]) -> None:
    history: list[dict[str, Any]] = list(st.session_state.turn_history)
    history.append(
        {
            "prompt": snapshot.get("user_message"),
            "agent": snapshot.get("routed_to"),
            "tool": snapshot.get("data_tool") or "—",
            "latency_ms": snapshot.get("latency_ms"),
        }
    )
    st.session_state.turn_history = history[-TURN_HISTORY_LIMIT:]


def _render_last_turn_observability(turn: dict[str, Any]) -> None:
    st.sidebar.markdown("**Last turn**")
    st.sidebar.caption(f"Prompt: {turn.get('user_message', '')}")

    routed_to = turn.get("routed_to") or "—"
    data_tool = turn.get("data_tool")
    tool_label = data_tool if data_tool else "—"
    st.sidebar.write(f"**Agent:** `{routed_to}`")
    st.sidebar.write(f"**Data tool:** `{tool_label}`")

    confidence = turn.get("route_confidence")
    if confidence is not None:
        st.sidebar.write(f"**Confidence:** {confidence:.0%}")
    if turn.get("route_source"):
        st.sidebar.write(f"**Via:** `{turn['route_source']}`")
    if turn.get("llm_model"):
        st.sidebar.write(f"**Model:** `{turn['llm_model']}`")
    latency = turn.get("latency_ms")
    if latency is not None:
        st.sidebar.write(f"**Latency:** {latency:.0f} ms")
    if turn.get("guardrail_blocked"):
        st.sidebar.warning("Guardrail blocked")

    chart = turn.get("chart")
    if chart:
        st.sidebar.markdown("**Chart output**")
        st.sidebar.write(
            f"`{chart['chart_type']}` · x=`{chart['x_column']}`"
            + (f" · y=`{chart['y_column']}`" if chart.get("y_column") else "")
        )
        st.sidebar.caption(f"Rows plotted: {chart.get('row_count', 0)}")
        with st.sidebar.expander("Chart SQL", expanded=False):
            st.code(chart.get("sql", ""), language="sql")

    insight = turn.get("insight")
    if insight:
        st.sidebar.markdown("**Insight output**")
        st.sidebar.write(f"Target: `{insight.get('target', '').upper()}`")
        top_features = insight.get("top_features") or []
        if top_features:
            st.sidebar.caption("Top features: " + ", ".join(top_features))

    data_query = turn.get("data_query")
    if data_query:
        st.sidebar.markdown("**SQL output**")
        st.sidebar.caption(
            f"Rows: {data_query.get('row_count', 0)}"
            + (" · truncated" if data_query.get("truncated") else "")
        )
        if data_query.get("explanation"):
            st.sidebar.caption(str(data_query["explanation"]))
        with st.sidebar.expander("SQL", expanded=False):
            st.code(data_query.get("sql", ""), language="sql")

    prediction = turn.get("prediction")
    if prediction:
        st.sidebar.markdown("**Prediction output**")
        st.sidebar.write(
            f"`{prediction.get('target', '').upper()}` → `{prediction.get('prediction')}`"
        )
        factors = prediction.get("top_global_factors") or []
        if factors:
            st.sidebar.caption("Top global factors: " + ", ".join(factors))

    predictions = turn.get("predictions")
    if predictions:
        st.sidebar.markdown("**Prediction output**")
        for key, details in predictions.items():
            st.sidebar.write(f"`{key.upper()}` → `{details.get('prediction')}`")

    rag = turn.get("rag")
    if rag:
        st.sidebar.markdown("**RAG output**")
        st.sidebar.caption(
            f"Retrieved: {rag.get('retrieved_count', 0)} · "
            f"Relevant: {rag.get('relevant_count', 0)} · "
            f"Grounded: {rag.get('grounded', False)}"
        )


def _render_turn_history() -> None:
    history: list[dict[str, Any]] = st.session_state.get("turn_history") or []
    if not history:
        return
    st.sidebar.markdown("**Turn history**")
    for index, entry in enumerate(reversed(history), start=1):
        latency = entry.get("latency_ms")
        latency_text = f" · {latency:.0f} ms" if latency is not None else ""
        st.sidebar.caption(
            f"{index}. `{entry.get('agent', '—')}` / `{entry.get('tool', '—')}`"
            f"{latency_text} — {entry.get('prompt', '')}"
        )


def _render_data_query_details(details: ChatDataQueryDetails) -> None:
    if details.explanation:
        st.markdown(f"**Query:** {details.explanation}")
    st.caption(f"Rows returned: {details.row_count}")
    if details.truncated:
        st.caption(f"Results truncated to displayed row cap.")

    with st.expander("SQL", expanded=False):
        st.code(details.sql, language="sql")

    if details.rows:
        with st.expander("Result table", expanded=True):
            st.dataframe(details.rows, use_container_width=True)

    st.caption(details.disclaimer)


def _render_chart_details(details: ChatChartDetails) -> None:
    if details.explanation:
        st.markdown(f"**Chart:** {details.explanation}")
    st.caption(
        f"Type: {details.chart_type} | Rows: {details.row_count} | "
        f"x={details.x_column}"
        + (f" | y={details.y_column}" if details.y_column else "")
    )

    with st.expander("SQL", expanded=False):
        st.code(details.sql, language="sql")

    if details.plotly_json:
        st.plotly_chart(details.plotly_json, use_container_width=True)

    st.caption(details.disclaimer)


def _render_insight_details(details: ChatInsightDetails) -> None:
    st.caption(f"Target: {details.target.upper()} | Source: {details.source}")
    if details.top_features:
        with st.expander("Top features", expanded=True):
            st.dataframe(details.top_features, use_container_width=True)
    st.caption(details.disclaimer)


def _render_rag_details(details: ChatRAGDetails) -> None:
    st.caption(
        f"Retrieved: {details.retrieved_count} | Relevant: {details.relevant_count} | "
        f"Grounded: {details.grounded} | Retries: {details.grounding_retry_count}"
    )

    if details.citations:
        with st.expander("Sources", expanded=True):
            for index, citation in enumerate(details.citations, start=1):
                st.markdown(
                    f"**{index}. {citation.source_file} — {citation.section_name}**"
                )
                if citation.score is not None:
                    st.caption(f"score: {citation.score:.2f}")
                st.write(citation.snippet)
    else:
        st.caption("No relevant citations were retained after grading.")

    st.caption(details.disclaimer)


def _render_chat_response(response: ChatResponse) -> None:
    st.markdown(response.text)

    if response.prediction is not None:
        _render_prediction_details(response.prediction)

    if response.predictions:
        for key, details in response.predictions.items():
            _render_prediction_details(details, label=key)

    if response.data_query is not None:
        with st.expander("Data query details", expanded=True):
            _render_data_query_details(response.data_query)

    if response.chart is not None:
        with st.expander("Chart", expanded=True):
            _render_chart_details(response.chart)

    if response.insight is not None:
        with st.expander("Model insights", expanded=True):
            _render_insight_details(response.insight)

    if response.rag is not None:
        with st.expander("Document sources", expanded=True):
            _render_rag_details(response.rag)

    meta_parts = _routing_metadata_parts(response)
    if response.metadata.llm_model:
        meta_parts.append(f"model: {response.metadata.llm_model}")
    if response.metadata.latency_ms is not None:
        meta_parts.append(f"latency: {response.metadata.latency_ms:.0f} ms")
    if meta_parts:
        st.caption(" | ".join(meta_parts))


def _render_sidebar() -> None:
    st.sidebar.title("Settings")
    st.session_state.api_base_url = st.sidebar.text_input(
        "API base URL",
        value=st.session_state.api_base_url,
    )

    api_url = st.session_state.api_base_url or settings.api_base_url
    if st.sidebar.button("Check /health"):
        st.session_state.health = fetch_health(api_url)

    health = st.session_state.health
    if health is None:
        st.sidebar.warning("Health not checked yet.")
    elif health.status == "ok":
        st.sidebar.success(f"API {health.status}")
    elif health.status == "degraded":
        st.sidebar.warning(f"API {health.status}")
    else:
        st.sidebar.error(f"API {health.status}")

    if health is not None:
        st.sidebar.write(
            {
                "llm_configured": health.llm_configured,
                "ml_models_loaded": health.ml_models_loaded,
                "documents_indexed": health.documents_indexed,
                "document_chunk_count": health.document_chunk_count,
                "langsmith_tracing": health.langsmith_tracing,
                "detail": health.detail,
            }
        )

    st.sidebar.divider()
    st.sidebar.caption(f"Session ID: `{st.session_state.session_id[:8]}…`")

    last_turn = st.session_state.get("last_turn")
    if last_turn:
        _render_last_turn_observability(last_turn)
        st.sidebar.divider()
        _render_turn_history()

    if st.sidebar.button("New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_turn = None
        st.session_state.turn_history = []
        st.rerun()


def _chat_tab() -> None:
    st.subheader("Chat")
    st.caption(
        "Natural-language requests via FastAPI → LangGraph orchestrator → specialist agents "
        "(prediction, data, rag, fallback)."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("response"):
                _render_chat_response(message["response"])
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask for a prediction, SQL query, or document search...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    api_url = st.session_state.api_base_url or settings.api_base_url

    try:
        with st.spinner("Calling /chat..."):
            response = post_chat(
                api_url,
                message=prompt,
                session_id=st.session_state.session_id,
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": response.text, "response": response},
        )
        snapshot = _build_turn_snapshot(user_message=prompt, response=response)
        st.session_state.last_turn = snapshot
        _append_turn_history(snapshot)
        st.rerun()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"API error ({exc.response.status_code}): {detail}",
            },
        )
        st.rerun()
    except httpx.HTTPError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Could not reach API: {exc}"},
        )
        st.rerun()


def _run_form_prediction(
    *,
    target: PredictionTarget,
    features: dict[str, Any],
) -> dict[str, Any]:
    from ml.feature_mapper import run_prediction

    request = PredictionRequest(
        target=target,
        features=PatientFeatures.model_validate(features),
    )
    result = run_prediction(request)
    if isinstance(result, dict):
        return {key: value.model_dump() for key, value in result.items()}
    return result.model_dump()


@st.cache_data
def _data_profile() -> DataProfile:
    return load_data_profile()


def _feature_sets_for_target(
    target: PredictionTarget,
) -> tuple[list[str], set[str], set[str]]:
    if target == PredictionTarget.COPD:
        required = set(COPD_REQUIRED_COLS)
        optional = set(COPD_OPTIONAL_COLS)
        return list(COPD_FEATURE_COLS), required, optional
    if target == PredictionTarget.ALT:
        required = set(ALT_REQUIRED_COLS)
        optional = set(ALT_OPTIONAL_COLS)
        return list(ALT_FEATURE_COLS), required, optional

    feature_cols = list(dict.fromkeys([*COPD_FEATURE_COLS, *ALT_FEATURE_COLS]))
    required = set(COPD_REQUIRED_COLS) | set(ALT_REQUIRED_COLS)
    optional = set(feature_cols) - required
    return feature_cols, required, optional


def _field_target_label(field: str) -> str:
    in_copd = field in COPD_FEATURE_COLS
    in_alt = field in ALT_FEATURE_COLS
    if in_copd and in_alt:
        return "COPD + ALT"
    if in_copd:
        return "COPD only"
    return "ALT only"


def _dataset_default(profile: DataProfile, field: str) -> Any:
    if field == "smoker":
        from ml.features import _parse_bool

        raw = _default_for_column(profile, field, is_numeric=False)
        parsed = _parse_bool(raw)
        return False if parsed is None else parsed
    return _default_for_column(profile, field, is_numeric=field in NUMERIC_FIELDS)


def _format_default(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _optional_field_caption(profile: DataProfile, field: str) -> str:
    default = _dataset_default(profile, field)
    if field in {"urban", "readmitted"}:
        default_label = "Urban" if int(default) == 1 else "Rural"
    else:
        default_label = _format_default(default)
    return f"Optional. If you don't enter a value, default value {default_label} will be used."


def _render_bool_field(
    field: str,
    *,
    required: bool,
    profile: DataProfile,
    targets: str,
) -> bool | None:
    default_display = _format_default(_dataset_default(profile, field))
    label = f"{field.replace('_', ' ').title()} *" if required else field.replace("_", " ").title()
    st.markdown(f"**{label}** ({targets})")
    if required:
        st.caption("Required. Enter a value to run prediction.")
    else:
        st.caption(_optional_field_caption(profile, field))

    if required:
        choice = st.selectbox(
            "Value",
            [EMPTY_SELECT_LABEL, "No", "Yes"],
            index=0,
            key=f"form_{field}",
            label_visibility="collapsed",
        )
        if choice == EMPTY_SELECT_LABEL:
            return None
        return choice == "Yes"

    choice = st.selectbox(
        "Value",
        [EMPTY_SELECT_LABEL, "No", "Yes"],
        index=0,
        key=f"form_{field}",
        label_visibility="collapsed",
    )
    if choice == EMPTY_SELECT_LABEL:
        return None
    return choice == "Yes"


def _render_numeric_field(
    field: str,
    *,
    required: bool,
    profile: DataProfile,
    targets: str,
) -> float | None:
    default_display = _format_default(_dataset_default(profile, field))
    label = f"{field.replace('_', ' ').title()} *" if required else field.replace("_", " ").title()
    st.markdown(f"**{label}** ({targets})")
    if required:
        st.caption("Required. Enter a value to run prediction.")
    else:
        st.caption(_optional_field_caption(profile, field))

    raw = st.number_input(
        "Value",
        min_value=0.0,
        value=None,
        placeholder=default_display,
        step=0.1,
        key=f"form_{field}",
        label_visibility="collapsed",
    )
    return None if raw is None else float(raw)


def _render_categorical_field(
    field: str,
    *,
    required: bool,
    profile: DataProfile,
    targets: str,
) -> str | None:
    default_display = _format_default(_dataset_default(profile, field))
    options = CATEGORICAL_OPTIONS[field]
    label = f"{field.replace('_', ' ').title()} *" if required else field.replace("_", " ").title()
    st.markdown(f"**{label}** ({targets})")
    if required:
        st.caption("Required. Enter a value to run prediction.")
    else:
        st.caption(_optional_field_caption(profile, field))

    select_options = [EMPTY_SELECT_LABEL, *options]
    choice = st.selectbox(
        "Value",
        select_options,
        index=0,
        key=f"form_{field}",
        label_visibility="collapsed",
    )
    if choice == EMPTY_SELECT_LABEL:
        return None
    return str(choice)


def _render_binary_int_field(
    field: str,
    *,
    required: bool,
    profile: DataProfile,
    targets: str,
) -> int | None:
    default = int(_dataset_default(profile, field))
    default_display = "Urban" if default == 1 else "Rural"
    label = f"{field.replace('_', ' ').title()} *" if required else field.replace("_", " ").title()
    st.markdown(f"**{label}** ({targets})")
    if required:
        st.caption("Required. Enter a value to run prediction.")
    else:
        st.caption(_optional_field_caption(profile, field))

    options: list[Any] = [EMPTY_SELECT_LABEL, 0, 1]
    choice = st.selectbox(
        "Value",
        options,
        index=0,
        format_func=lambda value: (
            EMPTY_SELECT_LABEL
            if value == EMPTY_SELECT_LABEL
            else ("Urban" if value == 1 else "Rural")
        ),
        key=f"form_{field}",
        label_visibility="collapsed",
    )
    if choice == EMPTY_SELECT_LABEL:
        return None
    if not isinstance(choice, int):
        return None
    return choice


def _missing_required_fields(
    features: dict[str, Any],
    required_cols: set[str],
) -> list[str]:
    return sorted(field for field in required_cols if field not in features)


def _collect_form_features(
    feature_cols: list[str],
    *,
    required_cols: set[str],
    profile: DataProfile,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for field in feature_cols:
        required = field in required_cols
        targets = _field_target_label(field)

        if field == "smoker":
            value = _render_bool_field(field, required=required, profile=profile, targets=targets)
        elif field in NUMERIC_FIELDS and field not in {"urban", "readmitted"}:
            value = _render_numeric_field(field, required=required, profile=profile, targets=targets)
        elif field in {"urban", "readmitted"}:
            value = _render_binary_int_field(field, required=required, profile=profile, targets=targets)
        elif field in CATEGORICAL_OPTIONS:
            value = _render_categorical_field(field, required=required, profile=profile, targets=targets)
        else:
            continue

        if value is not None:
            features[field] = value
    return features


def _form_tab() -> None:
    st.subheader("Form (v0)")
    st.caption("Direct ML inference without LLM. Fields match the selected target feature contract.")

    target_label = st.selectbox("Target", ["alt", "copd", "both"], index=0)
    target = PredictionTarget(target_label)
    profile = _data_profile()
    feature_cols, required_cols, optional_cols = _feature_sets_for_target(target)

    st.info(
        f"Showing {len(feature_cols)} model feature(s) for **{target_label}**: "
        f"{len(required_cols)} required, {len(optional_cols)} optional."
    )

    st.caption(
        "Required fields start empty and must be entered before prediction. "
        "Optional fields can be left empty."
    )

    with st.expander("Training-data reference values", expanded=False):
        for field in feature_cols:
            requirement = "required" if field in required_cols else "optional"
            default = _format_default(_dataset_default(profile, field))
            st.write(f"- `{field}` ({requirement}, {_field_target_label(field)}): `{default}`")

    features = _collect_form_features(feature_cols, required_cols=required_cols, profile=profile)

    if st.button("Run prediction", type="primary"):
        missing_required = _missing_required_fields(features, required_cols)
        if missing_required:
            st.error(
                "Cannot run prediction. Missing required field(s): "
                + ", ".join(missing_required)
            )
            return

        try:
            with st.spinner("Running ML inference..."):
                payload = _run_form_prediction(target=target, features=features)
            st.success("Prediction complete.")
            if isinstance(payload, dict) and "target" in payload:
                _render_prediction_details(ChatPredictionDetails.model_validate(payload))
            else:
                for key, value in payload.items():
                    _render_prediction_details(
                        ChatPredictionDetails.model_validate(value),
                        label=key,
                    )
        except Exception as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🩺", layout="wide")
    _init_session_state()

    st.title(PAGE_TITLE)
    st.caption("Clinical analytics prototype — COPD and ALT predictions.")

    _render_sidebar()

    chat_tab, form_tab = st.tabs(["Chat", "Form"])
    with chat_tab:
        _chat_tab()
    with form_tab:
        _form_tab()


if __name__ == "__main__":
    main()
