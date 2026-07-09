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

### 4. API and UI (in progress)

FastAPI and Streamlit are not implemented yet (Day 5). When ready:

```bash
# API
uv run uvicorn api.main:app --reload --app-dir src

# Streamlit UI
uv run streamlit run ui/app.py
```

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
