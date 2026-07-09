"""Smoke test runner for the LangGraph orchestrated chat flow."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

from agents.graph import build_graph
from agents.state import AgentState, chat_response_from_state, initial_state_from_chat_request
from schemas.chat import ChatRequest
from schemas.routing import AgentRoute

_EXAMPLE_MESSAGES: dict[str, str] = {
    "prediction": "Predict ALT for a patient with BMI 30",
    "data": "Show me a SQL query for readmissions by month",
    "rag": "What does the COPD guideline say about exercise?",
    "fallback": "What medication should the patient take for COPD?",
    "clarify": "hello there",
}


def _default_message() -> str:
    return _EXAMPLE_MESSAGES["prediction"]


def _routing_summary(response_dict: dict[str, Any], final_state: AgentState) -> dict[str, Any]:
    metadata = response_dict.get("metadata") or {}
    return {
        "routed_to": metadata.get("routed_to"),
        "route_confidence": metadata.get("route_confidence"),
        "route_source": metadata.get("route_source"),
        "guardrail_blocked": metadata.get("guardrail_blocked"),
        "agent_steps": final_state.get("agent_steps") or [],
        "route_reasoning": final_state.get("route_reasoning"),
    }


def run_smoke(
    message: str,
    session_id: str,
    user_id: str | None,
    *,
    expect_route: str | None = None,
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "message": message,
        "session_id": session_id,
    }
    if user_id is not None:
        request_payload["user_id"] = user_id
    request = ChatRequest(**request_payload)

    graph = build_graph()
    initial_state = initial_state_from_chat_request(request)
    final_state = cast(AgentState, graph.invoke(initial_state))
    response = chat_response_from_state(final_state)
    response_dict = response.model_dump()
    routing = _routing_summary(response_dict, final_state)

    result: dict[str, Any] = {
        "request": request.model_dump(),
        "routing": routing,
        "response": response_dict,
        "state_error": final_state.get("error"),
    }

    if expect_route is not None:
        result["expect_route"] = expect_route
        result["route_match"] = routing.get("routed_to") == expect_route

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a smoke test for the LangGraph orchestrated chat flow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example messages by expected route:\n"
            f"  prediction  { _EXAMPLE_MESSAGES['prediction']!r}\n"
            f"  data        { _EXAMPLE_MESSAGES['data']!r}\n"
            f"  rag         { _EXAMPLE_MESSAGES['rag']!r}\n"
            f"  fallback    { _EXAMPLE_MESSAGES['fallback']!r}\n"
            f"  clarify     { _EXAMPLE_MESSAGES['clarify']!r}\n"
            "\n"
            "Quick cases:\n"
            "  uv run python scripts/smoke_chat_graph.py --example prediction\n"
            "  uv run python scripts/smoke_chat_graph.py --example data --expect-route data\n"
            "  uv run python scripts/smoke_chat_graph.py --message 'Predict COPD' --session-id t4\n"
            "\n"
            "Note: prediction routes need LLM + ML artifacts. "
            "data/rag/fallback stubs use rule-based routing when keywords match."
        ),
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="User message for the graph (default: prediction example).",
    )
    parser.add_argument(
        "--example",
        choices=sorted(_EXAMPLE_MESSAGES),
        default=None,
        help="Use a built-in example message for a route (overrides default message).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="smoke-session",
        help="Session id for state tracking.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Optional user id.",
    )
    parser.add_argument(
        "--expect-route",
        choices=[route.value for route in AgentRoute],
        default=None,
        help="Optional assertion: fail unless metadata.routed_to matches this route.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.message and args.example:
        parser.error("Use either --message or --example, not both.")

    if args.example:
        message = _EXAMPLE_MESSAGES[args.example]
        expect_route = args.expect_route or (
            AgentRoute.FALLBACK.value if args.example == "clarify" else args.example
        )
    else:
        message = args.message or _default_message()
        expect_route = args.expect_route

    output = run_smoke(
        message=message,
        session_id=args.session_id,
        user_id=args.user_id,
        expect_route=expect_route,
    )

    routing = output["routing"]
    print(
        (
            f"routed_to={routing.get('routed_to')} "
            f"confidence={routing.get('route_confidence')} "
            f"source={routing.get('route_source')} "
            f"guardrail_blocked={routing.get('guardrail_blocked')}"
        ),
        file=sys.stderr,
    )

    if expect_route is not None and output.get("route_match") is False:
        print(
            json.dumps(output, indent=2, ensure_ascii=True),
            file=sys.stderr,
        )
        raise SystemExit(
            f"Expected route {expect_route!r}, got {routing.get('routed_to')!r}",
        )

    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
