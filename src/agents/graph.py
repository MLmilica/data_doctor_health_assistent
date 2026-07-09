"""LangGraph graph — minimal prediction slice (START -> predict -> END)."""

from __future__ import annotations

from typing import cast

from langgraph.graph import END, START, StateGraph

from agents.state import (
    AgentState,
    chat_response_from_state,
    initial_state_from_chat_request,
)
from agents.subagents.prediction_agent import run_prediction_agent
from schemas.chat import ChatRequest, ChatResponse

PREDICT_NODE = "predict"


def build_graph():
    """Compile the prediction-only graph used by /chat."""
    builder = StateGraph(AgentState)
    builder.add_node(PREDICT_NODE, run_prediction_agent)
    builder.add_edge(START, PREDICT_NODE)
    builder.add_edge(PREDICT_NODE, END)
    return builder.compile()


def run_chat_graph(request: ChatRequest) -> ChatResponse:
    """Invoke the graph for one chat request and return the public API response."""
    graph = build_graph()
    initial_state = initial_state_from_chat_request(request)
    final_state = cast(AgentState, graph.invoke(initial_state))
    return chat_response_from_state(final_state)
