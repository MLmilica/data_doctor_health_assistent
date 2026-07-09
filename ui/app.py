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
from schemas.chat import ChatPredictionDetails, ChatResponse, HealthResponse  # noqa: E402
from schemas.data import DataProfile  # noqa: E402
from schemas.prediction import PatientFeatures, PredictionRequest, PredictionTarget  # noqa: E402

PAGE_TITLE = "Data Doctor"
EMPTY_SELECT_LABEL = "— select —"
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


def _render_chat_response(response: ChatResponse) -> None:
    st.markdown(response.text)

    if response.prediction is not None:
        _render_prediction_details(response.prediction)

    if response.predictions:
        for key, details in response.predictions.items():
            _render_prediction_details(details, label=key)

    if response.metadata.llm_model or response.metadata.latency_ms is not None:
        meta_parts: list[str] = []
        if response.metadata.llm_model:
            meta_parts.append(f"model: {response.metadata.llm_model}")
        if response.metadata.latency_ms is not None:
            meta_parts.append(f"latency: {response.metadata.latency_ms:.0f} ms")
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
                "detail": health.detail,
            }
        )

    st.sidebar.divider()
    st.sidebar.caption(f"Session ID: `{st.session_state.session_id}`")
    if st.sidebar.button("New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()


def _chat_tab() -> None:
    st.subheader("Chat")
    st.caption("Natural-language requests routed through FastAPI → LangGraph → prediction agent.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("response"):
                _render_chat_response(message["response"])
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask for a COPD or ALT prediction...")
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
