# Data Doctor Project Overview

## What This Project Covers

Data Doctor is an AI prototype for clinical analysts.  
It combines:

- **Tabular ML predictions** for:
  - `COPD` severity class (A/B/C/D)
  - `ALT` lab value regression
- **LLM-based extraction + synthesis** around deterministic ML results
- **LangGraph multi-agent orchestration** (guardrails → orchestrator → specialist agents)
- **FastAPI + Streamlit UI** (Chat via orchestrated graph; Form for direct ML)

Current implemented core:

- **Orchestration:** guardrails, hybrid routing (rules + LLM), stub `data` / `rag` / `fallback` agents
- **Prediction agent:** feature mapping, LLM extract → ML → LLM synthesis
- Deterministic prediction response contract (LLM does not invent prediction values)
- LangGraph flow: `START -> orchestrator -> predict | data | rag | fallback -> END`

---

## Tech Stack

- Python 3.12+
- LangChain / LangGraph
- Pydantic / Pydantic Settings
- scikit-learn, XGBoost, LightGBM
- FastAPI, Streamlit
- DuckDB (planned SQL flow)

---

## Quick Start

From project root:

```bash
uv sync
uv sync --extra dev
```

---

## Environment Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Minimal required fields:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
```

If you want Anthropic instead:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Optional LangSmith tracing:

```env
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=data-doctor
```

---

## Running and Testing

### Run unit tests

```bash
uv run pytest
```

Target only prediction agent tests:

```bash
uv run pytest tests/test_agents/test_prediction_agent.py -v
```

Target only LangGraph tests:

```bash
uv run pytest tests/test_agents/test_graph.py -v
```

### Run type checks

```bash
uv run pyright
```

---

## LangGraph Prediction Flow

Graph entry point: `src/agents/graph.py`

Flow:

```text
ChatRequest -> initial AgentState -> graph.invoke() -> final AgentState -> ChatResponse
```

Graph topology:

```text
START -> orchestrator -> predict | data | rag | fallback -> END
```

- `orchestrator` applies guardrails and routes to a specialist agent
- `predict` calls `run_prediction_agent` (LLM extract → ML → synthesis)
- `data` / `rag` are stubs until DuckDB and vector search ship
- `fallback` handles guardrail blocks, low-confidence routing, and unclear requests
- `run_chat_graph(request)` is the high-level helper used by API/UI

For orchestration testing, see **Testing Initial Orchestration** at the end of this document.
For runtime smoke testing, see **Runtime Smoke Tests** below.

---

## Runtime Smoke Tests

Use these scripts to verify real graph behavior (not mocked unit tests).

| Script | What it exercises |
|--------|-------------------|
| `scripts/smoke_chat_graph.py` | Full path: `ChatRequest -> LangGraph -> orchestrator -> specialist agent -> ChatResponse` |
| `scripts/smoke_prediction_agent.py` | Prediction agent only (bypasses orchestrator and graph routing) |

`smoke_chat_graph.py` prints JSON with `request`, **`routing`**, `response`, and `state_error`.  
A one-line routing summary is printed to **stderr** before the JSON.

### Prerequisites

1. Dependencies installed (`uv sync`)
2. For **prediction** routes: `.env` with a valid provider API key + ML artifacts (`uv run python -m ml.train`)
3. For **data / rag / fallback** routes via smoke script: LLM key **not** required (rule-based orchestrator + stub agents)

`POST /chat` via API still requires an LLM key for all routes until that check is relaxed.

### Shared output checks

| Field | Pass condition |
|-------|----------------|
| `state_error` | `null` for successful runs |
| `routing.routed_to` | `prediction`, `data`, `rag`, or `fallback` |
| `routing.route_source` | `rules`, `llm`, or `guardrail` |
| `routing.agent_steps` | e.g. `["orchestrator:data", "data"]` |
| `response.session_id` | matches `--session-id` (or default `smoke-session`) |
| `response.text` | readable answer text |
| `response.prediction` | present for single-target **prediction** prompts |
| `response.predictions` | present for `both` prompts (`copd` + `alt`) |
| `response.prediction.can_predict` | `true` when required features are available |
| `response.prediction.defaults_used` | may be non-empty when optional fields are imputed |
| `response.metadata.llm_model` | populated for prediction routes |
| `response.metadata.latency_ms` | positive number |

### Example prompts

- `Predict ALT for a patient with BMI 30`
- `Predict COPD for smoker with poor diet and low exercise`
- `I need both predictions for BMI 29, moderate exercise, middle income`
- `Predict COPD`
- `Show me a SQL query for readmissions by month`

### Test `smoke_chat_graph.py`

Basic run (prediction example — needs LLM + ML):

```bash
uv run python scripts/smoke_chat_graph.py
```

Built-in examples by route (`--expect-route` asserts `routing.routed_to`):

```bash
uv run python scripts/smoke_chat_graph.py --example data --expect-route data
uv run python scripts/smoke_chat_graph.py --example rag --expect-route rag
uv run python scripts/smoke_chat_graph.py --example fallback --expect-route fallback
uv run python scripts/smoke_chat_graph.py --example clarify --expect-route fallback
```

Custom message:

```bash
uv run python scripts/smoke_chat_graph.py --message "Predict ALT for a patient with BMI 30"
```

Custom session id:

```bash
uv run python scripts/smoke_chat_graph.py --message "Predict COPD for smoker with poor diet and low exercise" --session-id demo-1
```

Run `uv run python scripts/smoke_chat_graph.py --help` for all example prompts.

Suggested manual cases:

```bash
uv run python scripts/smoke_chat_graph.py --message "Predict ALT for a patient with BMI 30" --session-id t1 --expect-route prediction
uv run python scripts/smoke_chat_graph.py --message "Predict COPD for smoker with poor diet and low exercise" --session-id t2 --expect-route prediction
uv run python scripts/smoke_chat_graph.py --message "I need both predictions for BMI 29, moderate exercise, middle income" --session-id t3 --expect-route prediction
uv run python scripts/smoke_chat_graph.py --message "Predict COPD" --session-id t4 --expect-route prediction
uv run python scripts/smoke_chat_graph.py --example data --session-id t5 --expect-route data
uv run python scripts/smoke_chat_graph.py --example rag --session-id t6 --expect-route rag
uv run python scripts/smoke_chat_graph.py --example fallback --session-id t7 --expect-route fallback
```

Expected by case:

- **t1/t2:** `state_error = null`, `routing.routed_to = prediction`, `can_predict = true`
- **t3:** `response.predictions` contains both `copd` and `alt`
- **t4:** clarification text or `can_predict = false` with `missing_required`
- **t5:** `routing.routed_to = data`, no ML prediction payload; data-agent stub text
- **t6:** `routing.routed_to = rag`, RAG stub text
- **t7:** `routing.routed_to = fallback`, `routing.guardrail_blocked = true` on guardrail example

### Test `smoke_prediction_agent.py`

Basic run:

```bash
uv run python scripts/smoke_prediction_agent.py
```

Custom message:

```bash
uv run python scripts/smoke_prediction_agent.py --message "Predict ALT for a patient with BMI 30"
```

Custom session id:

```bash
uv run python scripts/smoke_prediction_agent.py --message "Predict COPD for smoker with poor diet and low exercise" --session-id demo-1
```

Use the prediction manual cases (`t1`–`t4`) above; expected output shape is the same JSON contract but **without** a top-level `routing` block and without orchestrator steps.

Difference vs graph smoke test: this script calls `run_prediction_agent` directly, so it is useful when debugging prediction logic without LangGraph wiring.

### Common failure signals

- `state_error` contains API key/config errors -> check `.env` (`LLM_PROVIDER`, key variables)
- `FileNotFoundError` for model artifacts -> run `uv run python -m ml.train`
- empty `response.text` -> inspect provider/model settings in `src/config.py`

---

## Running the API

API entry point: `src/api/main.py`

Endpoints:

- `GET /health` — readiness check (`llm_configured`, `ml_models_loaded`)
- `POST /chat` — natural-language chat routed through LangGraph

### Prerequisites

1. Dependencies installed (`uv sync`)
2. `.env` configured with provider API key
3. ML artifacts available (if missing: `uv run python -m ml.train`)

### Start server

From project root:

```bash
uv run uvicorn api.main:app --reload --app-dir src
```

Default URL: `http://localhost:8000`, 
Swagger UI (interactiv testing): `http://localhost:8000/docs`

On startup, API preloads ML models (if artifacts exist) and compiles the LangGraph instance.

### Health check

```bash
curl http://localhost:8000/health
```

Expected fields:

- `status`: `ok`, `degraded`, or `error`
- `api`: `up`
- `llm_configured`: `true` when provider key is set
- `ml_models_loaded`: `true` when COPD/ALT artifacts were loaded

### Chat request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Predict ALT for a patient with BMI 30","session_id":"api-demo-1"}'
```

Request body (`ChatRequest`):

```json
{
  "message": "Predict ALT for a patient with BMI 30",
  "session_id": "api-demo-1",
  "user_id": "default-user"
}
```

`session_id` and `user_id` are optional (defaults are generated/used by schema).

### API tests

```bash
uv run pytest tests/test_api/test_chat.py -v
```

### Common API issues

- `503` on `/chat` with missing LLM key -> configure `.env` (`LLM_PROVIDER` + provider key)
- `degraded` on `/health` -> usually missing LLM key or ML artifacts
- model artifact errors -> run `uv run python -m ml.train`

---

## Running the Streamlit UI

UI entry point: `ui/app.py`

The UI has two tabs:

- **Chat** — sends natural-language messages to `POST /chat` (full E2E via API + LangGraph)
- **Form (v0)** — direct ML inference from structured fields (no LLM)

### Step-by-step startup

**Terminal 1 — start API:**

```bash
uv run uvicorn api.main:app --reload --app-dir src
```

**Terminal 2 — start Streamlit UI:**

```bash
uv run streamlit run ui/app.py
```

**Browser:**

- UI: `http://localhost:8501`
- API docs (optional): `http://localhost:8000/docs`

### Prerequisites

1. Dependencies installed (`uv sync`)
2. `.env` configured with provider API key (required for **Chat** tab)
3. ML artifacts available (if missing: `uv run python -m ml.train`)

### First run checklist

1. Open `http://localhost:8501`
2. In sidebar, keep API URL as `http://localhost:8000` (unless you changed API port)
3. Click **Check /health**
4. Confirm:
   - `llm_configured: true` (for Chat tab)
   - `ml_models_loaded: true` (for Chat + Form tab)
5. Go to **Chat** or **Form** tab and run examples below

### Chat tab examples

Type these prompts in the chat input:

- `Predict ALT for a patient with BMI 30`
- `Predict COPD for smoker with poor diet and low exercise`
- `I need both predictions for BMI 29, moderate exercise, middle income`
- `Predict COPD` (clarification / missing fields case)
- `Show me a SQL query for readmissions by month` (routes to data agent stub)

Expected behavior:

- assistant text appears in chat history
- for prediction prompts: prediction block with value/class
- expandable sections may show `defaults_used`, `missing_required`, `top_global_factors`
- disclaimer shown at the bottom of prediction details

### Form tab examples

Form fields change by target. Required fields start empty; prediction is blocked until they are filled.

**ALT example** (`alt` fields)

- BMI: `30` (required — must be entered manually)
- Optional fields (`diet_quality`, `exercise_frequency`, etc.) can be left empty or filled explicitly

Open **Training-data reference values** in the Form tab to see typical values from the training dataset (informational only).

Expected behavior:

- result appears immediately (no LLM call)
- single target -> one prediction block
- `both` -> separate COPD and ALT blocks

### Session controls (sidebar)

- **Check /health** — verifies API readiness
- **New session** — clears chat history and creates a new `session_id`

Chat history is kept only in the current browser session (`st.session_state`), not on the server.

### Common UI issues

- `Could not reach API` -> API is not running, or wrong API URL in sidebar
- `API error (503)` on Chat -> missing LLM key in `.env`
- Form tab error about model artifacts -> run `uv run python -m ml.train`
- Health shows `degraded` -> usually missing LLM key and/or ML artifacts

---

## Testing Initial Orchestration

The orchestration slice routes each `/chat` message through guardrails and the orchestrator before calling a specialist agent.

```text
START -> orchestrator -> predict | data | rag | fallback -> END
```

Check `response.metadata` for routing transparency:

| Field | Meaning |
|-------|---------|
| `routed_to` | Specialist that handled the message (`prediction`, `data`, `rag`, `fallback`) |
| `route_confidence` | Orchestrator confidence (0–1) |
| `route_source` | `rules`, `llm`, or `guardrail` |
| `guardrail_blocked` | `true` when input was blocked before routing |

In Streamlit, routing also appears under each assistant reply and in the sidebar **Last routing** panel.

### 1. Automated tests (no LLM required)

Fastest way to verify routing logic:

```bash
uv run pytest tests/test_agents/test_graph.py tests/test_agents/test_orchestrator.py tests/test_agents/test_guardrails.py -v
```

Or use the smoke script without an API key for stub routes:

```bash
uv run python scripts/smoke_chat_graph.py --example data --expect-route data
```

These tests mock the prediction agent and exercise rule-based routing, guardrails, and graph wiring.

### 2. Manual API tests (real E2E)

**Prerequisites:** API running (`uv run uvicorn api.main:app --reload --app-dir src`).

**Important:** `POST /chat` currently requires an LLM API key in `.env` for all routes, even data/rag/fallback stubs. Without a key you get `503`.

Check readiness first:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

#### A) SQL message → `data` stub

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Show me a SQL query for readmissions by month","session_id":"test-data"}' \
  | python3 -m json.tool
```

**Expected:**

- `metadata.routed_to` = `"data"`
- `metadata.route_source` = `"rules"` (keyword match)
- `prediction` = `null`
- `text` mentions the data agent stub

#### B) Guardrail block → `fallback`

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What medication should the patient take for COPD?","session_id":"test-guard"}' \
  | python3 -m json.tool
```

**Expected:**

- `metadata.routed_to` = `"fallback"`
- `metadata.guardrail_blocked` = `true`
- `prediction` = `null`
- `text` explains the request is out of scope

#### C) Document search → `rag` stub

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What does the COPD guideline say about exercise?","session_id":"test-rag"}' \
  | python3 -m json.tool
```

**Expected:**

- `metadata.routed_to` = `"rag"`
- `text` mentions the RAG agent stub

#### D) Prediction → `prediction` agent

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Predict ALT for a patient with BMI 30","session_id":"test-pred"}' \
  | python3 -m json.tool
```

**Expected:**

- `metadata.routed_to` = `"prediction"`
- `prediction` object present with `can_predict`, `prediction` value, etc.

**Also requires:** `llm_configured: true` and `ml_models_loaded: true` on `/health`.

### 3. Swagger UI (click-through)

1. Open `http://localhost:8000/docs`
2. Expand `POST /chat` → **Try it out**
3. Example body:

```json
{
  "message": "Show me a SQL query for readmissions by month",
  "session_id": "swagger-1"
}
```

4. **Execute** and inspect `metadata.routed_to` in the response.

### 4. Streamlit UI

**Terminal 1 — API:**

```bash
uv run uvicorn api.main:app --reload --app-dir src
```

**Terminal 2 — UI:**

```bash
uv run streamlit run ui/app.py
```

Open `http://localhost:8501` → **Chat** tab. Try the same prompts as in section 2.

**What to look for:**

- Under each assistant reply: `routed: data | confidence: … | via rules`
- Sidebar **Last routing**: JSON with `routed_to`, `route_confidence`, `route_source`, `guardrail_blocked`

### What works without an LLM key?

| Test method | Works without LLM key? |
|-------------|------------------------|
| `pytest` (orchestrator/graph/guardrails) | **Yes** |
| `curl` / Swagger / Streamlit Chat | **No** — `/chat` returns `503` without provider credentials |
| Streamlit **Form** tab | **Yes** — direct ML, no LLM |

### Quick reference — example prompts

| Prompt | Expected `routed_to` | Notes |
|--------|----------------------|-------|
| `Predict ALT for a patient with BMI 30` | `prediction` | Needs LLM + ML artifacts |
| `Show me a SQL query for readmissions by month` | `data` | Stub response for now |
| `What does the COPD guideline say about exercise?` | `rag` | Stub response for now |
| `What medication should the patient take for COPD?` | `fallback` | Guardrail block |
| `hello there` | `fallback` | Low confidence / clarification |

