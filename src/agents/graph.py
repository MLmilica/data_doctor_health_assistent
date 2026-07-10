"""LangGraph graph — orchestrator routes to specialist agents with multi-step loop."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.orchestrator import run_orchestrator
from agents.state import AgentState
from agents.subagents.data_agent import run_data_agent
from agents.subagents.fallback_agent import run_fallback_agent
from agents.subagents.prediction_agent import run_prediction_agent
from agents.subagents.rag_agent import run_rag_agent
from agents.subagents.synthesize_agent import run_synthesize_agent
from schemas.chat import ChatRequest, ChatResponse
from schemas.routing import AgentRoute, OrchestratorAction

ORCHESTRATOR_NODE = "orchestrator"
PREDICT_NODE = AgentRoute.PREDICTION.value
DATA_NODE = AgentRoute.DATA.value
RAG_NODE = AgentRoute.RAG.value
FALLBACK_NODE = AgentRoute.FALLBACK.value
SYNTHESIZE_NODE = "synthesize"


def _route_selector(state: AgentState) -> str:
    """Read orchestrator route and map to the next graph node."""
    route = state.get("route", AgentRoute.FALLBACK.value)
    if route in {PREDICT_NODE, DATA_NODE, RAG_NODE, FALLBACK_NODE}:
        return route
    return FALLBACK_NODE


def _after_orchestrator_selector(state: AgentState) -> str:
    """Route from orchestrator to specialist, synthesize, or END."""
    action = state.get("orchestrator_action", OrchestratorAction.ROUTE.value)
    if action == OrchestratorAction.SYNTHESIZE.value:
        return SYNTHESIZE_NODE
    if action == OrchestratorAction.FINISH.value:
        return END
    return _route_selector(state)


def build_graph():
    """Compile the orchestrated multi-agent graph used by /chat."""
    builder = StateGraph(AgentState)
    builder.add_node(ORCHESTRATOR_NODE, run_orchestrator)
    builder.add_node(PREDICT_NODE, run_prediction_agent)
    builder.add_node(DATA_NODE, run_data_agent)
    builder.add_node(RAG_NODE, run_rag_agent)
    builder.add_node(FALLBACK_NODE, run_fallback_agent)
    builder.add_node(SYNTHESIZE_NODE, run_synthesize_agent)

    builder.add_edge(START, ORCHESTRATOR_NODE)
    builder.add_conditional_edges(
        ORCHESTRATOR_NODE,
        _after_orchestrator_selector,
        {
            PREDICT_NODE: PREDICT_NODE,
            DATA_NODE: DATA_NODE,
            RAG_NODE: RAG_NODE,
            FALLBACK_NODE: FALLBACK_NODE,
            SYNTHESIZE_NODE: SYNTHESIZE_NODE,
            END: END,
        },
    )
    builder.add_edge(PREDICT_NODE, ORCHESTRATOR_NODE)
    builder.add_edge(DATA_NODE, ORCHESTRATOR_NODE)
    builder.add_edge(RAG_NODE, ORCHESTRATOR_NODE)
    builder.add_edge(FALLBACK_NODE, END)
    builder.add_edge(SYNTHESIZE_NODE, END)
    return builder.compile()


def invoke_chat_graph(graph, request: ChatRequest) -> ChatResponse:
    """Invoke a compiled graph for one chat request with session memory."""
    from memory.persistence import run_chat_with_memory

    response, _final_state = run_chat_with_memory(graph, request)
    return response


def run_chat_graph(request: ChatRequest) -> ChatResponse:
    """Invoke the graph for one chat request and return the public API response."""
    return invoke_chat_graph(build_graph(), request)
