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
START -> orchestrator -> predict | data | rag | fallback -> END
```

- `prediction` — COPD/ALT ML predictions (needs LLM + trained models)
- `data` / `rag` — stubs for now (routing works; full agents coming in Phase 2)
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
