# Data Doctor — System Architecture

This document describes the **implemented** POC architecture as of the current codebase. It is the source of truth for how components connect locally and how they would map to AWS.

---

## 1. System context

Data Doctor is an internal prototype for clinical analysts. It combines:

| Asset | Role |
|-------|------|
| `data/raw/patient_data.csv` | Tabular patient records (10k rows, 18 columns) |
| `data/documents/*.md` | Clinical briefs (diagnosis sections, treatment plans, etc.) |
| `artifacts/` | Trained models, data profile, SHAP summaries |

The user interacts via **Streamlit** (`ui/app.py`) or **FastAPI** (`POST /chat`). Both call the same **LangGraph** workflow.

```text
User → Streamlit / FastAPI → LangGraph → specialist agent(s) → ChatResponse
                                      ↓
                              session memory (in-process)
```

---

## 2. LangGraph topology

![LangGraph chat workflow](docs/assets/chat_graph.png)

```text
START → orchestrator
           ├─ route → prediction | data | rag
           │              └─ (loop back to orchestrator)
           ├─ synthesize → END        (≥2 specialists completed)
           ├─ finish → END            (single specialist or clarification)
           └─ fallback → END          (guardrail / low confidence)
```

**Key files:** `src/agents/graph.py`, `src/agents/orchestrator.py`, `src/agents/multi_step.py`

| Node | Responsibility |
|------|----------------|
| `orchestrator` | Guardrails, multi-step planning, routing metadata |
| `prediction` | NL → features → ML inference → synthesis |
| `data` | SQL, chart, or insight paths (see §5) |
| `rag` | Document retrieval, grading, synthesis, grounding check |
| `synthesize` | Merge multiple specialist `step_records` into one reply |
| `fallback` | User-facing message for blocked/unclear requests |

**Design choice:** Chart and insight are **internal tools inside the data agent**, not separate graph nodes. This keeps orchestration simple while still supporting assignment questions for charts and model insights.

---

## 3. Orchestrator and routing

**Hybrid routing:** deterministic rules first, LLM fallback for ambiguous messages.

| Signal type | Typical route |
|-------------|---------------|
| `predict`, explicit COPD/ALT | `prediction` |
| counts, averages, SQL analytics | `data` |
| `risk factors`, `what drives` | `data` → insight tool |
| `compare`, `distribution`, `chart` | `data` → chart tool |
| documents, symptoms, treatment plans | `rag` |
| guardrail block / low confidence | `fallback` |

**Multi-step:** `detect_required_agents()` can require multiple specialists in one turn (e.g. average BMI + ALT prediction). The graph loops orchestrator → agent → orchestrator until:

| `orchestrator_action` | Behavior |
|-----------------------|----------|
| `route` | Call next specialist (max 3 per message) |
| `synthesize` | ≥2 specialists done — merge via `synthesize_agent` |
| `finish` | Single specialist sufficient, or clarification needed |

**Key files:** `src/agents/orchestrator.py`, `src/agents/multi_step.py`, `src/schemas/routing.py`

**Why rules before LLM:** Faster, testable, and avoids routing analytics to RAG when keywords like “compare” or “how many” are present. Insight questions explicitly override broad clinical phrases like “what are the”.

---

## 4. Guardrails

Deterministic input filter before routing (`src/agents/guardrails.py`):

- Empty / oversized messages
- Direct clinical advice patterns (prescribe, diagnose me, dosage)

Blocked messages route to `fallback` with an explanation — the system does not attempt prediction or RAG on out-of-scope advice requests.

---

## 5. Data agent

Single graph node with three internal paths (`src/agents/tools/data_task_classifier.py`):

```text
                    ┌─ sql    → LLM SQL → SqlLayer → DuckDB → synthesis
User message → data ┼─ chart  → LLM ChartSpec → SqlLayer → Plotly renderer → synthesis
                    └─ insight → load SHAP JSON → synthesis (no runtime SHAP)
```

### 5.1 SQL layer (shared)

```text
LLM → LLMSQLExtraction → SqlLayer.validate_sql() → DuckDB (patients table) → DataQueryResult
```

- Read-only `SELECT` / `WITH … SELECT` only
- Rejects invented table names, DDL/DML, multiple statements
- Row cap via `sql_max_rows` (default 1000)

**Key files:** `src/agents/tools/sql_layer.py`, `src/data/loader.py`, `src/data/schema_registry.py`

### 5.2 Chart tool

- LLM outputs **ChartSpec only** (never Plotly code)
- Deterministic `SELECT` built from column names in the spec
- Predefined renderers: histogram, bar, boxplot, scatter, line, heatmap

**Key files:** `src/agents/tools/chart_tool.py`, `src/schemas/chart.py`

### 5.3 Insight tool (fast path)

- Loads offline artifacts from `artifacts/insights/*_shap_summary.json` and feature importance JSON
- LLM synthesizes prose from pre-computed rankings — **does not compute SHAP at runtime**
- Targets: `copd` or `alt` (resolved from user message)

**Key files:** `src/agents/tools/insight_tool.py`, `src/ml/shap_insights.py`, `src/schemas/insight.py`

**Not in POC:** dynamic subgroup insights, `statistics_tool` stub (SQL path covers aggregates).

---

## 6. Prediction agent

```text
User message → LLM extraction (PatientFeatures) → merge session facts
            → ML inference (COPD classifier / ALT regressor)
            → LLM synthesis (numbers from ML only)
```

| Model | Algorithm | Features |
|-------|-----------|----------|
| COPD | XGBoost multiclass | 6 categorical (diet, income, urban, diagnosis, exercise, smoker) |
| ALT | Ridge regression | bmi, readmitted, exercise, albumin_globulin_ratio, diagnosis, diet |

Required vs optional features and median imputation are defined in `src/ml/features.py` using `artifacts/data_profile.json`.

**Important:** Assignment questions mention age, sex, medication count, hospital days — the LLM may extract them, but they are **not** in the trained feature sets (EDA-driven reduction). The UI shows `used_features`, `defaults_used`, and `missing_required` for transparency.

**Key files:** `src/agents/subagents/prediction_agent.py`, `src/ml/predict.py`, `src/ml/train.py`, `src/ml/feature_mapper.py`

---

## 7. RAG agent

```text
User message → Chroma retrieval (top-k)
            → LLM chunk grading (corrective RAG)
            → LLM synthesis with citations
            → LLM grounding verification (self-RAG style)
            → retry with strict prompt if not grounded
```

**Indexing pipeline:**

1. Parse each `.md` file by `## ` sections (`src/data/document_parser.py`)
2. One chunk per section with metadata (`source_file`, `section_name`, `diagnosis_codes`)
3. Embed with OpenAI `text-embedding-3-small` → Chroma persistent store (`data/chroma/`)

**Key files:** `src/agents/subagents/rag_agent.py`, `src/data/vectorstore.py`, `scripts/index_documents.py`

**Limitation:** Documents describe synthetic patient briefs — they are **not** joined to CSV rows. Questions like “heart attack patient medications” depend on corpus content; the agent should say when information is not in indexed documents.

---

## 8. Memory model

Three concepts — do not conflate:

| Layer | What it stores | POC storage |
|-------|----------------|-------------|
| **Transcript** | User/assistant messages | `ChatSession.turns` |
| **Step ledger** | Per-agent `StepRecord` (agent, tool, artifact summary) | `ChatSession.steps` |
| **Session facts** | `last_target`, `last_features`, `last_sql`, `last_route` | `ChatSession.facts` |

**Per `/chat` request:**

```text
load ChatSession → enrich AgentState → graph.invoke → persist turn + steps + facts → save
```

**Key files:** `src/memory/session_store.py`, `src/memory/persistence.py`, `src/memory/context.py`, `src/schemas/memory.py`

**POC limitation:** `InMemorySessionStore` — lost on API restart. LangGraph SQLite checkpointer (`src/memory/checkpointer.py`) is a stub for a future phase.

---

## 9. API and UI boundary

| Layer | Role |
|-------|------|
| `src/schemas/chat.py` | `ChatRequest` / `ChatResponse` — UI does not import agents directly |
| `src/api/routes/chat.py` | `POST /chat` |
| `ui/app.py` | Chat tab (orchestrated), Form tab (direct ML, no LLM) |

`ChatResponse` exposes structured blocks: `prediction`, `data_query`, `chart`, `insight`, `rag`, plus routing metadata for the sidebar observability panel.

---

## 10. Observability

**LangSmith** (optional): `src/observability/langsmith.py` mirrors `LANGCHAIN_*` env vars at startup. Each graph invoke uses `thread_id = {user_id}:{session_id}` for trace grouping.

**Not implemented:** custom token/latency callbacks (`src/observability/callbacks.py` stub).

---

## 11. Local → AWS mapping

This POC runs entirely on a developer machine. A production deployment on AWS would map components as follows:

| POC (local) | AWS target | Notes |
|-------------|------------|-------|
| FastAPI (`uvicorn`) | **API Gateway** + **ECS Fargate** or **Lambda** | Stateless API; session store externalized |
| Streamlit UI | **Amplify** or **CloudFront** + **ECS** | Or internal tool behind VPN |
| DuckDB in-process | **Athena** / **Redshift** / **RDS** | Patient table as curated dataset |
| Chroma on disk | **OpenSearch** or **Bedrock Knowledge Bases** | Managed vector search |
| `InMemorySessionStore` | **DynamoDB** (+ optional **ElastiCache**) | Durable sessions + checkpoint state |
| `artifacts/` (models, SHAP) | **S3** | Versioned model artifacts |
| ML training (offline) | **SageMaker Training** / scheduled job | Same pipelines, S3 I/O |
| OpenAI API | **Amazon Bedrock** (Claude, Titan embeddings) | Loka/AWS alignment |
| LangSmith | **CloudWatch** + **X-Ray** | Trace and metrics |
| `.env` secrets | **Secrets Manager** / **SSM Parameter Store** | No keys in images |

**Not required for take-home:** actual deploy. The mapping above shows cloud-compatible thinking without incurring AWS cost during development.

---

## 12. POC boundaries (intentional)

| Item | Status |
|------|--------|
| Chart tool (6 Plotly renderers) | Implemented |
| Insight tool (offline SHAP) | Implemented |
| Multi-step orchestration + synthesize | Implemented |
| Session memory (in-process) | Implemented |
| LangSmith tracing | Implemented (optional) |
| Dynamic insight / runtime SHAP | Not implemented |
| `statistics_tool` | Stub (SQL covers aggregates) |
| LangGraph checkpointer | Stub |
| Persistent memory on AWS | Documented only |

---

## 13. Key entry points

| Task | Command / file |
|------|----------------|
| Run API | `uv run uvicorn api.main:app --reload --app-dir src` |
| Run UI | `uv run streamlit run ui/app.py` |
| Train models | `uv run python -m ml.train` |
| Index documents | `uv run python scripts/index_documents.py` |
| Regenerate graph PNG | `PYTHONPATH=src uv run python scripts/regenerate_graph_png.py` |
| Tests | `uv run pytest` |

See [README.md](README.md) for full setup and [SUBMISSION.md](SUBMISSION.md) for component-by-component rationale.
