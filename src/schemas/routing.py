"""Routing schemas for the orchestrator and LangGraph conditional edges."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AgentRoute(str, Enum):
    """Target specialist agent selected by the orchestrator."""

    PREDICTION = "prediction"
    DATA = "data"
    RAG = "rag"
    FALLBACK = "fallback"


class OrchestratorAction(str, Enum):
    """What the orchestrator node tells the graph to do next."""

    ROUTE = "route"
    SYNTHESIZE = "synthesize"
    FINISH = "finish"


SPECIALIST_AGENT_ROUTES: frozenset[AgentRoute] = frozenset(
    {AgentRoute.PREDICTION, AgentRoute.DATA, AgentRoute.RAG},
)


RoutingSource = Literal["rules", "llm", "guardrail", "multi_step"]


class RoutingDecision(BaseModel):
    """Structured routing outcome stored in AgentState."""

    route: AgentRoute
    confidence: float = Field(ge=0.0, le=1.0, description="Router confidence in the chosen route.")
    reasoning: str = ""
    requires_clarification: bool = False
    clarification_prompt: str | None = None
    source: RoutingSource = "rules"


class LLMRoutingExtraction(BaseModel):
    """Structured output for ambiguous messages routed via LLM."""

    route: AgentRoute
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0-1 confidence that this route matches user intent.",
    )
    reasoning: str = Field(description="Short justification for the route choice.")
    requires_clarification: bool = Field(
        default=False,
        description="True when the message is too vague to execute safely.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="Question to ask the user when clarification is required.",
    )


class LLMMultiStepPlan(BaseModel):
    """Structured output for orchestrator loop decisions after the first agent."""

    action: OrchestratorAction
    route: AgentRoute | None = Field(
        default=None,
        description="Required when action=route; target specialist agent.",
    )
    reasoning: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
