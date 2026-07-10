# Orchestration and Memory

This document explains **why the system is orchestrated this way** and **how memory supports multi-turn chat**. It is written for reviewers evaluating the take-home exercise — not as API reference, but as the reasoning behind routing, multi-step flows, and session continuity.

**Companion docs:** [ARCHITECTURE.md](../ARCHITECTURE.md) (system diagram) · [SUBMISSION.md](../SUBMISSION.md) §6 (component summary) · [FUTURE_WORK.md](../FUTURE_WORK.md) (planned improvements)

**Key source files:** `src/agents/graph.py`, `src/agents/orchestrator.py`, `src/agents/multi_step.py`, `src/memory/persistence.py`, `src/memory/context.py`

---

## 1. Why multi-agent orchestration?

A single LLM answering every question would mix responsibilities: SQL generation, ML inference, document retrieval, and clinical prose in one prompt. That leads to hallucinated numbers, wrong tool choice, and no audit trail.

Data Doctor separates concerns into **specialist agents**, each with a narrow job:

| Agent | Responsibility | Grounding source |
|-------|----------------|------------------|
| **Prediction** | Extract patient features → run ML → explain result | Trained models + `data_profile.json` |
| **Data** | SQL analytics, charts, or model insights | DuckDB (`patients` table) or offline SHAP JSON |
| **RAG** | Retrieve and cite clinical documents | Chroma index over `data/documents/` |
| **Fallback** | Greetings, out-of-scope, blocked requests | Static templates |

**LangGraph** (`src/agents/graph.py`) wires these as explicit nodes with conditional edges. The benefit for a POC is not complexity for its own sake — it is **testability** (each node has unit tests), **observability** (LangSmith traces per step), and **controlled tool use** (the prediction agent cannot invent SQL; the data agent cannot search documents unless routed elsewhere).

---

## 2. Graph topology — one turn, many possible paths

```text
START → orchestrator ─┬→ prediction ──┐
                      ├→ data ────────┼→ orchestrator (loop)
                      ├→ rag ─────────┘
                      ├→ fallback → END
                      ├→ synthesize → END
                      └→ END (finish)
```

**Design choices worth noting:**

1. **Specialists loop back to the orchestrator** — this enables multi-step questions in a single `/chat` call without hard-coding pipelines like `data_then_prediction`.
2. **Chart and insight are tools inside the data agent**, not separate graph nodes. Assignment questions still route to `data`; the data agent then picks `sql`, `chart`, or `insight` internally. This keeps the graph small while supporting the full feature set.
3. **Fallback does not loop back** — blocked or out-of-scope messages end immediately.
4. **Synthesize runs once** when two or more specialists completed work in the same turn.

---

## 3. Message lifecycle — single-step example

Consider: *"Predict ALT for BMI 30."*

```text
1. POST /chat { message, session_id }
2. Load ChatSession from InMemorySessionStore
3. Enrich AgentState with windowed transcript + session facts + prior steps
4. graph.invoke:
   a. orchestrator → guardrails pass
   b. plan_next_step → route to prediction (explicit "predict" + "alt")
   c. prediction agent → LLM extracts { bmi: 30 } → Ridge inference → LLM synthesis
   d. orchestrator → finish (one specialist sufficient)
5. Build ChatResponse with prediction details
6. Persist user turn, assistant turn, step record, updated SessionFacts
7. Return response to UI
```

The user sees one assistant message. The sidebar (observability panel) can show `routed_to: prediction` and structured metadata — but the graph may have visited orchestrator twice (entry + finish check).

---

## 4. Routing — how the orchestrator decides

Routing is **hybrid**: deterministic rules first, LLM fallback for ambiguous cases.

### 4.1 Guardrails (before routing)

`src/agents/guardrails.py` blocks empty messages, oversized input, and direct clinical advice patterns (e.g. "prescribe", "diagnose me"). Blocked messages go to **fallback** with an explanation — the system does not attempt ML or RAG on out-of-scope requests.

### 4.2 Rule-based fast path

`route_with_rules()` in `src/agents/orchestrator.py` scores keyword signals:

| Signal family | Typical route | Example |
|---------------|---------------|---------|
| Analytics (`how many`, `average`, `group by`) | `data` → sql | "How many smokers?" |
| Model insight (`risk factors`, `what drives`, `shap`) | `data` → insight | "Main risk factors for COPD?" |
| Chart (`chart`, `compare`, `distribution`) | `data` → chart | "Compare BMI by diet quality" |
| Document search (`guideline`, `according to`, `summarize`) | `rag` | "What do documents recommend for exercise?" |
| Clinical knowledge (`symptoms`, `what are the`) | `rag` | "Symptoms of seasonal allergies?" |
| Prediction (`predict`, `forecast`) | `prediction` | "Predict COPD for good diet" |

**Why rules before LLM?**

- **Speed** — no API call for obvious intents.
- **Testability** — unit tests assert signal → route without mocking an LLM.
- **Safety** — analytics keywords like `compare` or `how many` should not drift to RAG because the message also mentions "patients."

### 4.3 Priority and overrides

When multiple signals appear, precedence matters. Two fixes that shaped the final design:

1. **Insight before broad clinical phrases** — "What are the main risk factors for COPD?" contains `what are the` (normally RAG) but also `risk factors` (insight). Insight signals win → `data` → insight tool. Without this, the system would search documents instead of reading offline SHAP artifacts.

2. **Analytics before prediction** — "Compare average BMI among smokers and predict ALT for BMI 30" has both analytics and prediction signals. Multi-step detection (§5) handles this; for single-intent messages, analytics signals route to `data` unless `predict` is explicit.

### 4.4 LLM routing fallback

When rules do not match confidently, `decide_route()` calls a structured-output LLM with:

- Current user message
- Windowed conversation history
- Recent backend step summaries from the session

The LLM returns `RoutingDecision`: route, confidence, optional clarification prompt.

**Low-confidence handling:** if confidence is below threshold, the orchestrator may set `requires_clarification` rather than guessing — preferring a question over a wrong specialist.

### 4.5 Second routing layer — data task classifier

Once routed to `data`, `data_task_classifier.py` chooses `sql`, `chart`, or `insight`:

- SQL priority signals (`count`, `average`, `how many`) can override chart signals when both appear — e.g. "compare average BMI" is SQL, not a chart.
- Insight signals route to offline SHAP JSON, not SQL.

This is a deliberate **two-level** design: orchestrator picks the specialist; the specialist picks the tool.

---

## 5. Multi-step orchestration — one message, multiple agents

### 5.1 When is a request multi-step?

`detect_required_agents()` estimates which specialists a message needs:

```text
"Compare average BMI in the dataset and predict ALT for BMI 30"
  → required: { data, prediction }
```

Detection uses the same signal families as routing (analytics, prediction intent, document search, insight). If **more than one** specialist is required, the orchestrator enters a loop.

### 5.2 Execution order

Priority when multiple agents remain:

```text
DATA → PREDICTION → RAG
```

**Rationale:** dataset analytics often provide context before a prediction; document answers are typically supplementary. Example flow:

```text
User: "Average BMI in the dataset and ALT prediction for BMI 30"

orchestrator → data (SQL: AVG(bmi))
            → orchestrator → prediction (ALT for bmi=30)
            → orchestrator → synthesize (2 agents done)
            → END
```

The user receives **one** synthesized answer covering both the dataset metric and the prediction.

### 5.3 Orchestrator actions

`plan_next_step()` returns one of:

| Action | When |
|--------|------|
| `route` | More specialist work needed (max **3** specialists per message) |
| `synthesize` | ≥2 specialists completed; merge via `synthesize_agent` |
| `finish` | Single specialist sufficient, or clarification required |

An optional LLM planner (`_plan_with_llm`) assists when multiple agents remain — but rule-based priority is the fallback if the LLM fails.

### 5.4 Synthesize agent

`src/agents/subagents/synthesize_agent.py` receives `step_records` from the current turn (SQL result summary, prediction value, RAG citations) and produces one coherent narrative. Rules:

- Use only facts from step JSON — no invented numbers.
- Do not mention internal plumbing ("the data agent said…").
- End with a prototype disclaimer.

If synthesis LLM is unavailable, a deterministic fallback concatenates step summaries.

---

## 6. Memory model — three layers

Memory is often confused with "chat history." This system separates **three concepts** intentionally:

| Layer | What it stores | Purpose |
|-------|----------------|---------|
| **Transcript** | `ChatSession.turns` — user and assistant messages | Natural language context for routing and extraction |
| **Step ledger** | `ChatSession.steps` — `StepRecord` per agent/tool run | Backend audit trail: what ran, with artifact summaries |
| **Session facts** | `ChatSession.facts` — structured `SessionFacts` | Deterministic carry-forward for follow-ups |

Do not conflate them:

- The **transcript** is what the user said and what they were told.
- The **step ledger** is what the backend did (SQL executed, chart rendered, prediction made).
- **Session facts** are the minimum structured state needed to continue without re-extracting everything.

### 6.1 Per-request flow

```text
load ChatSession
  → enrich AgentState (windowed turns, facts, prior steps)
  → graph.invoke
  → persist: append turns, step records, update facts
  → save session
```

**Windowing** (`config.py`):

- `memory_max_turns = 10` — recent transcript for prompts
- `memory_max_prior_steps = 5` — recent step summaries for routing context
- `memory_sql_sample_rows = 5` — sample rows stored in step artifacts

Windows prevent unbounded context growth; they are a POC trade-off (summarized memory would be better long-term — see [FUTURE_WORK.md](../FUTURE_WORK.md)).

### 6.2 Session facts — what gets carried forward

After a **prediction** turn, `update_session_facts()` may set:

| Field | Example |
|-------|---------|
| `last_route` | `"prediction"` |
| `last_target` | `"alt"` |
| `last_features` | `{ "bmi": 28, "diet_quality": "Good", ... }` |
| `missing_required` | `[]` |

After a **data** SQL turn:

| Field | Example |
|-------|---------|
| `last_sql` | `"SELECT AVG(bmi) FROM patients"` |

These facts are injected into the **prediction extraction prompt** (`build_prediction_extraction_prompt`) so follow-ups can merge new values onto prior features.

### 6.3 Follow-up example — memory in action

```text
Turn 1: "Predict ALT for BMI 28"
  → prediction → facts: { last_target: "alt", last_features: { bmi: 28, ... } }

Turn 2: "What if BMI is 35?"
  → routing: _looks_like_prediction_follow_up() → prediction
  → extraction prompt includes session facts
  → merge_patient_features(base=last_features, delta={ bmi: 35 })
  → ALT prediction with BMI 35, other fields carried forward
```

This is the primary memory success case in the POC: **numeric and categorical follow-ups on an in-progress prediction scenario**.

### 6.4 What memory does not do (yet)

| Gap | Impact |
|-----|--------|
| **In-memory store only** | Sessions lost on API restart |
| **No LangGraph checkpointer** | Graph state not persisted mid-turn across restarts |
| **Insight does not inherit `last_target`** | After predicting ALT, "What are the main risk factors?" requires the user to say ALT or COPD |
| **No entity memory** | "That patient" does not resolve to a stored profile beyond `last_features` |
| **No cross-turn synthesize** | Multi-step is within one message, not across turns |

These are documented trade-offs, not oversights — see [FUTURE_WORK.md](../FUTURE_WORK.md) for the persistence and follow-up resolver roadmap.

---

## 7. How memory feeds routing and agents

Memory is not passive storage — it actively shapes behavior:

| Consumer | Memory inputs | Effect |
|----------|---------------|--------|
| **Orchestrator LLM** | `conversation_history`, `prior_steps` | Disambiguate "what about exercise?" after a prediction turn |
| **Rule routing** | `last_route` from session facts | Prediction follow-up fast path |
| **Prediction extraction** | `session_facts` (last features, target, missing) | Merge follow-up deltas |
| **Multi-step planner** | `step_records` within current turn | Decide synthesize vs next agent |
| **UI sidebar** | `ChatResponse.metadata`, turn history | Observability for reviewer/demo |

**Key insight:** routing uses both **transcript** (what was said) and **facts** (what was structurally true after the last prediction). The transcript helps the LLM; the facts enable deterministic merge without re-parsing prior assistant prose.

---

## 8. Design trade-offs — what we chose and why

| Decision | Alternative considered | Why we chose this |
|----------|------------------------|-------------------|
| Hybrid rules + LLM routing | LLM-only router | Testable fast path; fewer misroutes on analytics |
| Orchestrator loop | Fixed DAG per combo | Flexible; new multi-step patterns without new graph edges |
| Chart/insight inside data agent | Separate graph nodes | Simpler topology; same assignment coverage |
| Three memory layers | Transcript only | Facts enable deterministic prediction follow-up |
| In-memory sessions | SQLite/DynamoDB from day one | POC speed; architecture documents AWS path |
| Max 3 specialists per turn | Unlimited loop | Cost and latency cap; matches assignment scope |
| Synthesize as separate node | Each agent writes final prose | Single voice when multiple sources contribute |

---

## 9. Routing examples — edge cases that matter

| User message | Route | Internal path | Why |
|--------------|-------|---------------|-----|
| How many smokers? | `data` | sql | Pure analytics |
| Compare lab results: readmitted vs not | `data` | chart | Compare signal → visualization |
| Main risk factors for COPD? | `data` | insight | Insight beats "what are the" → RAG |
| Symptoms of seasonal allergies? | `rag` | retrieve → synthesize | Clinical knowledge, not CSV |
| Predict ALT for BMI 30 | `prediction` | extract → ML → synthesize | Explicit prediction |
| Avg BMI + predict ALT for BMI 30 | multi | data → prediction → synthesize | Two required agents |
| What medication should I take for COPD? | `fallback` | guardrail | No prescribing |
| Turn 1: Predict ALT BMI 28 → Turn 2: What if BMI 35? | `prediction` (both) | session facts merge | Memory follow-up |

For executable test prompts and expected metadata, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — sections *Multi-step Orchestrator* and *Testing Initial Orchestration*.

---

## 10. What works well in the POC

- **End-to-end multi-agent loop** with synthesize for combo questions
- **Inspectable routing** — rules, confidence, source (`rules` vs `llm`) exposed in metadata
- **Prediction follow-ups** via `SessionFacts` and feature merge
- **Separation of grounding** — SQL/ML/RAG each tied to a real backend, not LLM invention
- **144 tests** including orchestrator, graph, guardrails, and memory persistence

---

## 11. What would come next

The highest-impact improvements are in [FUTURE_WORK.md](../FUTURE_WORK.md):

1. **Routing evaluation dataset** — golden prompts with expected routes; CI regression gate
2. **Persistent session store** — survive API restarts; DynamoDB on AWS
3. **Richer session facts** — inherit insight target, active cohort filters
4. **Streaming status events** — show routing and tool progress during long multi-step turns
5. **LangGraph checkpointer** — durable graph state for production deployments

---

## 12. Summary for reviewers

| Question | Answer |
|----------|--------|
| **Why LangGraph?** | Explicit specialists, testable nodes, multi-step loop, LangSmith traces |
| **How is routing decided?** | Guardrails → rules → LLM fallback; data agent has a second classifier for sql/chart/insight |
| **How does multi-step work?** | Detect required agents → loop orchestrator → specialist → orchestrator → synthesize if ≥2 ran |
| **What is stored in memory?** | Transcript, step ledger, session facts — three layers, different jobs |
| **What does memory enable?** | Prediction follow-ups ("What if BMI is 35?"); routing context for ambiguous turns |
| **What are the limits?** | In-memory only; no insight target inheritance; max 3 specialists per message |

The orchestration and memory layers exist to make the right tool run on the right data, remember enough to continue a conversation, and return one honest answer — even when a single user message requires dataset analytics, ML inference, and document search at once.

---

*Synthetic data only — not for clinical use.*
