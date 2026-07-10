# Data Doctor — AI Health Assistant

POC for clinical analysts: **ML predictions** (COPD / ALT), **dataset analytics** (SQL, charts, model insights), and **RAG** over indexed clinical documents — orchestrated by **LangGraph** with a Streamlit chat UI.

**Documentation:**

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and AWS mapping
- [SUBMISSION.md](SUBMISSION.md) — component-by-component walkthrough for reviewers
- [FUTURE_WORK.md](FUTURE_WORK.md) — planned improvements beyond the POC
- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) — detailed test matrices and smoke commands
- [docs/DATA_PREDICT.md](docs/DATA_PREDICT.md) — data insights and model decision rationale
- [docs/ORCHESTRATION_MEMORY.md](docs/ORCHESTRATION_MEMORY.md) — orchestration and memory reasoning

---

## Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/)
- Local data (not in Git):
  - `data/raw/patient_data.csv`
  - `data/documents/*.md`
- LLM API key (OpenAI or Anthropic) in `.env`

---

## Quick start

```bash
# 1. Install
uv sync --extra dev

# 2. Configure
cp .env.example .env
# Edit .env — at minimum OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 3. Data profile
uv run python main.py

# 4. Train ML models + SHAP artifacts
uv run python -m ml.train

# 5. Index clinical documents (for RAG)
uv run python scripts/index_documents.py

# 6. Run API (terminal 1)
uv run uvicorn api.main:app --reload --app-dir src

# 7. Run UI (terminal 2)
uv run streamlit run ui/app.py
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

---

## Graph overview

```text
START → orchestrator → prediction | data | rag | fallback
              ↑___________|  (specialists loop back)
              → synthesize → END   (2+ agents in one turn)
              → END                  (single agent / clarification)
```

![LangGraph chat workflow](docs/assets/chat_graph.png)

Regenerate after graph changes:

```bash
PYTHONPATH=src uv run python scripts/regenerate_graph_png.py
```

---

## What each part does

| Component | Chat route | What it does |
|-----------|------------|--------------|
| **Prediction** | `prediction` | NL → features → COPD/ALT models → synthesis |
| **Data — SQL** | `data` | NL → validated DuckDB query → synthesis |
| **Data — Chart** | `data` | NL → ChartSpec → Plotly (boxplot, histogram, …) |
| **Data — Insight** | `data` | Offline SHAP/importance → synthesis |
| **RAG** | `rag` | Chroma search → grade → cite → grounding check |
| **Fallback** | `fallback` | Guardrails, unclear requests |

**UI tabs:**

- **Chat** — full orchestration via `POST /chat` (needs LLM)
- **Form** — direct ML inference without LLM

The sidebar shows **agent**, **data tool** (sql/chart/insight), and turn history after each message.

---

## Running parts individually

### Tests

```bash
uv run pytest
```

### EDA / ML notebooks

```bash
uv run jupyter notebook notebooks/01_eda.ipynb
```

### Smoke tests (real LLM)

```bash
uv run python scripts/smoke_chat_graph.py --example data --expect-route data
uv run python scripts/smoke_prediction_agent.py
```

### Optional: LangSmith tracing

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=data-doctor
```

Restart the API after changing `.env`. Check `/health` for `langsmith_tracing: true`.

---

## Configuration

Key settings in `src/config.py` and `.env`:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `openai` or `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | LLM access |
| `LANGCHAIN_*` | Optional LangSmith tracing |

---

## Project layout

```text
src/
  agents/       # LangGraph orchestrator + sub-agents + tools
  api/          # FastAPI
  data/         # loader, schema, documents, Chroma
  ml/           # train, predict, features, SHAP
  memory/       # session store, persistence
  schemas/      # Pydantic contracts
  observability/# LangSmith setup
ui/             # Streamlit
notebooks/      # EDA, ML development
tests/          # pytest (144 tests)
artifacts/      # models, profile, insights (generated)
docs/           # PROJECT_OVERVIEW, graph PNG
```

---

## Assignment demo prompts

See [SUBMISSION.md §10](SUBMISSION.md#10-assignment-question-checklist) for the full checklist. Examples:

1. *Predict COPD for 55yo male, BMI 27.5, poor diet…* → `prediction`
2. *How many smokers?* → `data` (sql)
3. *Compare lab results across readmitted vs non-readmitted* → `data` (chart)
4. *What are the main risk factors for COPD?* → `data` (insight)
5. *Summarize treatment plan for diabetic patients over 60* → `rag`

---




