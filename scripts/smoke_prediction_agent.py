"""Smoke test runner for the prediction agent with real LLM calls."""

from __future__ import annotations

import argparse
import json
from typing import Any

from agents.state import chat_response_from_state, initial_state_from_chat_request
from agents.subagents.prediction_agent import run_prediction_agent
from schemas.chat import ChatRequest


def _default_message() -> str:
    return "Predict ALT for a patient with BMI 30."


def run_smoke(message: str, session_id: str, user_id: str | None) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "message": message,
        "session_id": session_id,
    }
    if user_id is not None:
        request_payload["user_id"] = user_id
    request = ChatRequest(**request_payload)
    initial_state = initial_state_from_chat_request(request)
    final_state = run_prediction_agent(initial_state)
    response = chat_response_from_state(final_state)

    return {
        "request": request.model_dump(),
        "response": response.model_dump(),
        "state_error": final_state.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a smoke test for prediction agent.")
    parser.add_argument(
        "--message",
        type=str,
        default=_default_message(),
        help="User message for the agent.",
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
    args = parser.parse_args()

    output = run_smoke(
        message=args.message,
        session_id=args.session_id,
        user_id=args.user_id,
    )
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
