# Data Doctor — AI Health Assistant

AI prototype for clinical analysts: structured patient data queries, outcome predictions, and RAG search over clinical documents.

## Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- Local data files (not tracked in Git):
  - `data/raw/patient_data.csv`
  - `data/documents/*.md`

## Installation

```bash
# from the project root
uv sync

# include dev tools (pytest, jupyter, ruff)
uv sync --extra dev
```

Activate the virtual environment (optional — `uv run` works without this):

```bash
source .venv/bin/activate
```

## Running the project

### 1. Generate data profile

Builds a statistical profile of the CSV dataset and saves it to `artifacts/data_profile.json`:

```bash
uv run python main.py
```

Expected output:

```
Rows: 10000
Columns: 18
Saved data profile to: artifacts/data_profile.json
```

### 2. Tests

```bash
uv run pytest
```

Data module only:

```bash
uv run pytest tests/test_data/
```

### 3. EDA notebook

**In Cursor / VS Code** (recommended):

1. Open `notebooks/01_eda.ipynb`
2. Select kernel: `.venv (Python 3.12)`
3. Run cells with `Shift+Enter`

**In the browser:**

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

### 4. Train ML models (if needed)

```bash
uv run python -m ml.train
```

### 5. API and Streamlit UI

The **Chat** tab uses FastAPI + LangGraph orchestration. The **Form** tab calls ML directly (no LLM).

**Terminal 1 — API:**

```bash
uv run uvicorn api.main:app --reload --app-dir src
```

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

**Terminal 2 — UI:**

```bash
uv run streamlit run ui/app.py
```

- UI: `http://localhost:8501`

**Graph flow:**

```text
START → orchestrator → predict | data | rag | fallback
              ↑___________|  (specialists loop back)
              → synthesize? → END   (2+ agents)
              → END                 (single agent or fallback)
```

![LangGraph chat workflow](docs/assets/chat_graph.png)

Regenerate after graph changes: `PYTHONPATH=src uv run python scripts/regenerate_graph_png.py`

Index clinical documents for RAG: `uv run python scripts/index_documents.py` (see `docs/RAG.md`).

- `prediction` — COPD/ALT ML predictions (needs LLM + trained models)
- `data` — LLM SQL → DuckDB → LLM synthesis; SQL and result table stay in the UI expander for verification (needs LLM + CSV)
- `rag` — document search over indexed `data/documents/` via Chroma (needs LLM + OpenAI embeddings + index)
- `fallback` — guardrail blocks, unclear requests, low-confidence routing

`POST /chat` currently requires an LLM API key in `.env` for all routes.

### 6. Smoke tests (real LLM / graph)

**Full graph** (orchestrator + specialist agents):

```bash
uv run python scripts/smoke_chat_graph.py
uv run python scripts/smoke_chat_graph.py --example data --expect-route data
uv run python scripts/smoke_chat_graph.py --example rag --expect-route rag
uv run python scripts/smoke_chat_graph.py --example fallback --expect-route fallback
```

JSON output includes a top-level `routing` block (`routed_to`, `route_confidence`, `route_source`, `agent_steps`). A one-line routing summary is printed to stderr.

**Prediction agent only** (bypasses orchestrator):

```bash
uv run python scripts/smoke_prediction_agent.py
```

**Orchestration unit tests** (no LLM):

```bash
uv run pytest tests/test_agents/test_graph.py tests/test_agents/test_orchestrator.py tests/test_agents/test_guardrails.py -v
```

See `docs/PROJECT_OVERVIEW.md` → **Testing Initial Orchestration** for curl/Swagger/UI examples.

## Configuration

Create `.env` from the example file:

```bash
cp .env.example .env
```

Then fill in values in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=data-doctor
```

Set `LLM_PROVIDER` to `openai` or `anthropic`. The app reads the matching key automatically.

Paths are configured in `src/config.py` (defaults: `data/raw/`, `data/documents/`, `artifacts/`).

## Project structure

```
src/
  data/       # loader, schema, profile, document parser
  ml/         # feature engineering, training, prediction
  agents/     # LangGraph orchestrator and sub-agents
  memory/     # session store, context helpers, persistence
  api/        # FastAPI
  schemas/    # Pydantic models
ui/           # Streamlit
notebooks/    # EDA
tests/        # pytest
artifacts/    # generated profiles, models, insights
data/         # local data (gitignored)
```

## Cursor / IDE

The project uses a `src/` layout. `.vscode/settings.json` points the Python interpreter to `.venv` and adds `extraPaths` for imports (`config`, `data`, etc.).

If you see `reportMissingImports` warnings:

1. Cmd+Shift+P → **Python: Select Interpreter** → choose `.venv`
2. Cmd+Shift+P → **Developer: Reload Window**

## Session memory (Phase 4)

Multi-turn chat context is stored **server-side** per `(user_id, session_id)`. The Streamlit UI already sends a stable `session_id`; the API loads and saves session state around each `POST /chat`.

### What was implemented

| Component | Role |
|-----------|------|
| `schemas/memory.py` | `ChatTurn` (transcript), `StepRecord` (backend step ledger), `SessionFacts` (structured follow-up context), `ChatSession` (container) |
| `memory/session_store.py` | In-memory `SessionStore` (POC; replace with DynamoDB/Redis on AWS) |
| `memory/context.py` | History windowing, prompt formatting, `merge_patient_features` |
| `memory/persistence.py` | Load session → enrich `AgentState` → persist turn after graph |
| `agents/state.py` | New fields: `conversation_history`, `session_facts`, `prior_steps`, `step_records` |
| `agents/orchestrator.py` | Routing uses conversation history; follow-up rule for prediction (e.g. *"What if BMI is 35?"*) |
| `agents/subagents/prediction_agent.py` | Extraction with session context; merge `last_features`; inherit `last_target` on follow-up |

**Per `/chat` request:**

```text
load ChatSession → enrich AgentState → graph.invoke → append turns + step + facts → save
```

**Config** (`src/config.py`):

- `memory_max_turns` (default 10) — transcript window for LLM prompts
- `memory_max_prior_steps` (default 5) — recent step ledger window
- `memory_sql_sample_rows` (default 5) — max SQL rows stored in step artifacts

**Limitation:** memory lives in **process RAM**. Restarting the API clears all sessions. LangGraph checkpointer is planned for Phase 5b (see [Multi-step orchestrator](#multi-step-orchestrator-phase-5)).

More detail: `docs/MEMORY.md`.

### How to test memory

**Unit tests (no LLM):**

```bash
uv run pytest tests/test_memory/ -v
```

**Full suite** (includes memory + follow-up graph test):

```bash
uv run pytest
```

**Manual — Streamlit (recommended):**

1. Start API and UI (see [§5](#5-api-and-streamlit-ui)).
2. In Chat, send: *"Predict ALT for BMI 28"*
3. In the **same session** (do not click **New session**), send: *"What if BMI is 35?"*
4. Expect a new ALT prediction using BMI 35; sidebar shows the same Session ID.

**Manual — smoke script** (same `session_id` for two calls):

```bash
uv run python scripts/smoke_chat_graph.py \
  --message "Predict ALT for BMI 28" \
  --session-id demo-memory \
  --expect-route prediction

uv run python scripts/smoke_chat_graph.py \
  --message "What if BMI is 35?" \
  --session-id demo-memory \
  --expect-route prediction
```

**Verify isolation:** click **New session** in Streamlit (new UUID) or use a different `--session-id` — prior features should not carry over.

**Verify restart behavior:** stop and restart the API, then send a follow-up with the old `session_id` — memory is empty (expected for in-memory POC).

## Multi-step orchestrator (Phase 5)

One user message can trigger **multiple specialist agents** in a single `/chat` call. The orchestrator runs in a **loop**, each agent appends a `StepRecord`, and a **synthesize** node merges results when two or more agents ran. The user still sees **one** assistant reply.

### What was implemented

| Component | Role |
|-----------|------|
| `agents/graph.py` | Loop: specialists → orchestrator; `synthesize` node; `finish` → END |
| `agents/orchestrator.py` | Guardrails + `plan_next_step()`; sets `orchestrator_action` |
| `agents/multi_step.py` | Rule/LLM planner: which agents are needed, next action |
| `agents/subagents/synthesize_agent.py` | LLM merges `step_records` into final `response_text` |
| `schemas/routing.py` | `OrchestratorAction` (`route`, `synthesize`, `finish`), `LLMMultiStepPlan` |
| `agents/state.py` | `orchestrator_action`; metadata `routed_to: "multi"` when applicable |
| `memory/persistence.py` | `append_run_step_record()`; persist all `step_records` per turn |
| Specialist agents | Each appends a compact `StepRecord` after execution |

**Per `/chat` request (multi-agent):**

```text
load ChatSession → orchestrator → agent → orchestrator → … → synthesize? → persist → save
```

**Orchestrator actions:**

| `orchestrator_action` | Meaning |
|-----------------------|---------|
| `route` | Call next specialist (`data`, `prediction`, or `rag`) |
| `synthesize` | All required work done — merge step summaries |
| `finish` | Single agent is enough — use its `response_text` |

**Rules (MVP):**

- Max **3 specialist agents** per message (`data`, `prediction`, `rag`)
- `synthesize` runs only when **≥ 2** specialists completed
- Single-agent questions behave as before (no synthesize step)
- `fallback` still goes directly to END (no loop)
- LangGraph checkpointer deferred to Phase 5b

**Config** (`src/config.py`):

- `orchestrator_max_agent_steps` (default 3) — cap on specialist agents per message

More detail: `docs/MULTI_ORCHESTRATION.md`.

### How to test multi-step

**Unit tests (no LLM):**

```bash
uv run pytest tests/test_agents/test_multi_step.py -v
```

**Graph integration** (mocked LLM synthesis):

```bash
uv run pytest tests/test_agents/test_multi_step_graph.py -v
```

**Full suite:**

```bash
uv run pytest
```

**Manual — Streamlit:**

1. Start API and UI (see [§5](#5-api-and-streamlit-ui)).
2. Send a combo question, e.g. *"Compare average BMI in the dataset with ALT prediction for BMI 30"*
3. Expect one assistant message combining SQL + prediction; metadata may show `routed_to: multi`.

**Manual — smoke script:**

```bash
uv run python scripts/smoke_chat_graph.py \
  --message "Compare average BMI in the dataset with ALT prediction for BMI 30" \
  --session-id demo-multistep
```

**Single-agent sanity check** (no synthesize — should still work):

```bash
uv run python scripts/smoke_chat_graph.py \
  --message "How many patients per income bracket?" \
  --expect-route data
```
