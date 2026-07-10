# Data Doctor — Architecture

Internal AI prototype for clinical analysts. Answers questions about patient data, predicts COPD and ALT outcomes, and retrieves knowledge from clinical documents via a conversational interface.

## High-Level Overview

The system connects three capabilities:

1. **Structured data** — SQL/statistics/charts over `data/raw/patient_data.csv`
2. **Predictive models** — COPD (4-class) and ALT (regression) inference from natural language
3. **Document knowledge** — RAG over 1050 clinical `.md` files with citations

**Important:** CSV patients (`P00000`) and document patients (`PAT-789012`) are **separate data sources** with no direct join.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│  LangGraph Orchestrator │
│  UI         │◀────│   (async)    │◀────│                         │
└─────────────┘     └──────────────┘     └───────────┬─────────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────────┐
                    │                                │                    │
              ┌─────▼─────┐              ┌───────────▼──┐      ┌────────▼────────┐
              │ Data Agent │              │  Prediction  │      │   RAG Agent     │
              │            │              │    Agent     │      │                 │
              └─────┬─────┘              └──────┬───────┘      └────────┬────────┘
                    │                             │                       │
         ┌──────────┼──────────┐                  │               ┌───────▼───────┐
         │          │          │                  │               │    Chroma     │
    SQL Layer  Statistics  Chart/Insight         │               │  (1050 .md)   │
         │          │          │                  │               └───────────────┘
    ┌────▼────┐                              ┌────▼────┐
    │ DuckDB  │                              │ ML Models│
    │ (CSV)   │                              │+Insights│
    └─────────┘                              └─────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + uvicorn (async) |
| Agents | LangGraph + LangChain tools |
| Validation | Pydantic v2 |
| Patient data | Pandas + DuckDB (SQL) |
| Documents | Chroma (section-based chunking) |
| ML | scikit-learn + XGBoost/LightGBM/CatBoost |
| Charts | Plotly (6 predefined render functions) |
| UI | Streamlit |
| Tracing | LangSmith + custom token/latency callbacks |
| Memory | LangGraph SQLite checkpointer + in-memory sessions |
| Tests | pytest + pytest-asyncio |
| Package manager | uv |

## Data Assets

### `data/raw/patient_data.csv` (~10,000 rows, 18 columns)

| Column | Type | Notes |
|--------|------|-------|
| `patient_id` | ID | P00000... |
| `age`, `sex`, `bmi`, `smoker` | Features | |
| `diagnosis_code` | Categorical | D1, D4... |
| `medication_count`, `days_hospitalized`, `readmitted` | Features | readmitted: 0/1 |
| `exercise_frequency` | Categorical | None / Low / Moderate / High |
| `diet_quality` | Categorical | Poor / Average / Good |
| `urban`, `income_bracket`, `education_level` | Features | |
| `last_lab_glucose`, `albumin_globulin_ratio` | Numeric | |
| `chronic_obstructive_pulmonary_disease` | **Target** | 4 classes: A, B, C, D (~2500 each) |
| `alanine_aminotransferase` | **Target** | Continuous regression |

### `data/documents/` (1050 `.md` files)

Structured clinical documents with consistent sections:

`Patient Information`, `Diagnosis`, `Medications`, `Treatment Plan`, `Past Medical History`, `Chief Complaint`, etc.

No direct link to CSV patient IDs — documents serve as a **knowledge base**, not per-patient chart lookup.

## Agent Architecture

### 1 Orchestrator + 3 Sub-agents + Shared SQL Layer

| Component | Role | Example questions |
|-----------|------|-------------------|
| **Orchestrator** | Routes queries, parallelizes, synthesizes final answer | All |
| **Data Agent** | Counts, statistics, charts, risk insights | "How many smokers?", "Compare lab results" |
| **Prediction Agent** | NL → features → inference → explanation | "Predict COPD for 55yo male..." |
| **RAG Agent** | Document search, citations, summarization | "Symptoms of allergies?", "Medications for heart attack" |
| **SQL Layer** (shared) | Generates and executes SQL, returns DataFrame | Used by Data, Chart, Insight |

### Data Agent Internals

```
Data Agent
├── SQL Layer (shared data access service)
├── Statistics Tool     → count, mean, distributions
├── Chart Tool          → ChartSpec → SQL → predefined Plotly functions
└── Insight Tool        → pre-computed SHAP/importance + dynamic analysis
```

**Rule:** Numeric answers always come from SQL/Statistics — never from LLM memory.

## Query Routing

| Question | Agent | Path |
|----------|-------|------|
| "How many smokers?" | Data → SQL | SQL COUNT |
| "Males >40 readmitted?" | Data → SQL | SQL WHERE + COUNT |
| "Compare lab results readmitted vs not" | Data → Chart | ChartSpec boxplot → SQL → Plotly |
| "Main risk factors for COPD?" | Data → Insight | Pre-computed SHAP → LLM synthesis |
| "Predict COPD for 55yo male..." | Prediction | NL → Pydantic → model → explain |
| "Predict ALT for 44yo woman..." | Prediction | NL → Pydantic → model → explain |
| "Symptoms of seasonal allergies?" | RAG | Chroma search → citations |
| "Medications for heart attack patient?" | RAG | Chroma filter Medications section |
| "Treatment plan for diabetic 60+" | RAG | Chroma → summarization + citations |

## LangGraph — States and Transitions

### Graph Flow

```
START → ingest_message → route_intent
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         data_agent    prediction_agent   rag_agent
              │               │               │
         chart_tool      clarify_user?        │
         insight_tool         │               │
              │               │               │
              └───────────────► synthesize_response → persist_memory → END
```

### Nodes

| Node | Input | Output | Uses LLM? |
|------|-------|--------|-----------|
| `ingest_message` | `user_message` | `language`, history update | No |
| `route_intent` | `user_message` + history | `intent`, `agents_to_invoke` | Yes → `RouteDecision` |
| `data_agent` | message | `sql_result` / `statistics_result` | Yes |
| `chart_tool` | message + context | `chart_result` | Yes → `ChartSpec` only |
| `insight_tool` | message | `insight_result` | Yes (interprets pre-computed) |
| `prediction_agent` | message | `prediction_result` or `needs_clarification` | Yes → `PredictionRequest` |
| `rag_agent` | message | `rag_result` + `citations` | Yes (synthesis) |
| `synthesize_response` | all results | `final_answer`, `charts`, `metadata` | Yes |
| `clarify_user` | missing fields | `clarification_question` | Yes |
| `persist_memory` | final state | checkpoint save | No |

### Conditional Routing

**After `route_intent`:**
- Single agent → that agent node
- Multiple agents → `parallel_dispatch` (fan-out, then fan-in at synthesize)
- Needs clarification → `clarify_user`

**After `data_agent`:**
- `sql` / `statistics` → `synthesize_response`
- `chart` → `chart_tool` → `synthesize_response`
- `insight` → `insight_tool` → `synthesize_response`

**After `prediction_agent`:**
- Missing required fields → `clarify_user`
- Complete → `synthesize_response`

### Session Config

```python
config = {
    "configurable": {
        "thread_id": f"{user_id}:{session_id}",
    },
    "callbacks": [TokenTrackingCallback(), LangSmithCallback()],
}
```

## Chart Tool Design

LLM **never** generates Plotly/Matplotlib code.

```
User question → LLM → ChartSpec (Pydantic)
    → SQL Layer (fetch data)
    → predefined function (create_boxplot, create_histogram, ...)
    → Plotly JSON → UI
```

**6 render functions:** `create_histogram`, `create_bar_chart`, `create_boxplot`, `create_scatter_plot`, `create_line_chart`, `create_heatmap`

## Insight Tool Design

| Mode | When | Source |
|------|------|--------|
| **Fast insights** | "Risk factors for COPD/ALT" | Pre-computed JSON from train pipeline |
| **Dynamic insights** | "Risk factors for readmitted >60" | SQL filter → on-the-fly analysis |

Pre-computed artifacts stored in `artifacts/insights/`:
- Feature importance per target
- SHAP summary
- Correlation matrices
- Target class distributions

LLM does **not** compute — only interprets pre-computed findings.

## Prediction Agent Design

```
User sentence
    → LLM structured output (Pydantic PredictionRequest)
    → Feature mapper ("athlete"→High, "woman"→Female, "center"→urban=1)
    → Validate missing fields
        → critical missing: ask user + suggest values from data_profile
        → optional missing: fill with median, tell user
    → Model inference
    → Response: prediction + features used + top factors + disclaimer
```

**Two models:**
- COPD → multi-class classifier (A/B/C/D)
- ALT → regressor

### Natural Language Feature Mapping

| User says | Maps to |
|-----------|---------|
| athlete, sportista | `exercise_frequency="High"` |
| doesn't exercise | `exercise_frequency="None"` |
| poor diet | `diet_quality="Poor"` |
| woman, žena | `sex="Female"` |
| center of city | `urban=1` |
| readmitted | `readmitted=1` |

## RAG Agent Design

### Ingestion (once at startup)

1. Parse 1050 `.md` files by `## ` sections
2. Each section = one chunk with metadata: `source_file`, `section_name`, `diagnosis_codes`
3. Embed → Chroma (`data/chroma/`)

### Retrieval

- Semantic search (top-k)
- Optional metadata filter (`section_name="Medications"`)
- LLM synthesis with **mandatory citations**

### Treatment suggestions

RAG + filter on `Treatment Plan` / `Medications` sections + disclaimer in response.

## Memory Model

Three separate concepts — do not conflate:

| Type | What it stores | Local → AWS |
|------|----------------|-------------|
| **Agent memory** | Last N messages, graph checkpoint state | SQLite checkpointer → DynamoDB |
| **Computed artifacts** | Models, SHAP, Chroma index, DuckDB, data_profile | Disk → S3 |
| **Display output** | Messages, chart JSON, citations, tokens, latency | API response (stateless) |

Session isolation: `thread_id = f"{user_id}:{session_id}"`

## Observability

| Metric | How |
|--------|-----|
| Agent steps | LangSmith tracing |
| Input/output tokens | Custom `BaseCallbackHandler` |
| Latency per node | `time.perf_counter()` |
| Tool calls | LangSmith + structured logs |

Response metadata includes: `input_tokens`, `output_tokens`, `total_latency_ms`, `agents_invoked`, `node_timings_ms`.

## UI (Streamlit)

- **Chat panel** — message history, markdown responses
- **Chart panel** — Plotly charts from API response
- **Sidebar** — session info, token/latency stats, user_id
- **Citations** — source display for RAG answers

UI sends requests to FastAPI — it does not know about agents directly.

## Multi-language Support

- Detect user language → respond in same language
- Internal: English (column names, SQL, schemas, metadata)
- NL → feature mapping works in Serbian and English
- Include 2–3 Serbian questions in eval set

## Project Structure

```
loka1/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/chat.py
│   │   └── dependencies.py
│   ├── agents/
│   │   ├── graph.py
│   │   ├── orchestrator.py
│   │   ├── state.py
│   │   ├── subagents/
│   │   │   ├── data_agent.py
│   │   │   ├── prediction_agent.py
│   │   │   └── rag_agent.py
│   │   └── tools/
│   │       ├── sql_layer.py
│   │       ├── statistics_tool.py
│   │       ├── chart_tool.py
│   │       └── insight_tool.py
│   ├── ml/
│   │   ├── eda.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── features.py
│   │   └── feature_mapper.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── schema_registry.py
│   │   ├── document_parser.py
│   │   └── vectorstore.py
│   ├── memory/
│   │   ├── session_store.py
│   │   └── checkpointer.py
│   ├── observability/
│   │   ├── callbacks.py
│   │   └── metrics.py
│   └── schemas/
│       ├── chat.py
│       ├── chart.py
│       ├── prediction.py
│       └── citation.py
├── ui/
│   └── app.py
├── tests/
├── notebooks/
├── artifacts/
│   ├── models/
│   ├── insights/
│   └── data_profile.json
├── data/
│   ├── raw/patient_data.csv
│   ├── documents/
│   └── chroma/
└── docs/
```

## Startup Lifecycle

```
1. Load `data/raw/patient_data.csv` → DuckDB table "patients"
2. Generate data_profile.json (min/max/mode/distributions)
3. Index 1050 MD files → Chroma (if index does not exist)
4. Load ML models from artifacts/models/
5. Load insights JSON from artifacts/insights/
6. Initialize LangGraph with checkpointer
7. Start FastAPI + Streamlit
```

All of this runs **once** — not per-request.

## AWS Mapping

| Local (POC) | AWS (production) |
|-------------|------------------|
| FastAPI | API Gateway + ECS Fargate |
| Streamlit UI | Amplify / CloudFront + S3 |
| DuckDB | Athena / RDS |
| Chroma | OpenSearch Serverless / Bedrock Knowledge Bases |
| SQLite checkpointer | DynamoDB |
| In-memory sessions | ElastiCache Redis |
| ML models | S3 + SageMaker Endpoints |
| Insights artifacts | S3 |
| LangSmith | LangSmith (unchanged) |
| CSV + MD files | S3 |
| OpenAI API | Bedrock (Claude) |

Use abstractions (`SessionStore`, `VectorStore`, `ModelRegistry`) with local and AWS implementations.

## Testing Strategy

| Component | Approach |
|-----------|----------|
| SQL/Data | Golden tests — exact counts |
| Chart | ChartSpec validation (type + columns) |
| RAG | Retrieval recall@3 |
| Prediction | Known input → expected range/class |
| Routing | Question → expected agent (15 questions from spec) |
| E2E | All spec questions as integration tests |

## 5-Day Implementation Plan

| Day | Focus | Deliverable |
|-----|-------|-------------|
| **1** | EDA, schema registry, data_profile, DuckDB loader, document parser | `01_eda.ipynb`, `artifacts/data_profile.json` |
| **2** | ML pipeline: COPD classifier + ALT regressor, SHAP, insights JSON | `artifacts/models/`, `artifacts/insights/` |
| **3** | SQL layer, Statistics/Chart/Insight tools, RAG ingestion | Tools work standalone |
| **4** | LangGraph agents, Prediction Agent, memory, observability | Agent system end-to-end |
| **5** | FastAPI, Streamlit UI, tests, README, demo script | Presentable POC |

## Demo Script (Interview)

1. "How many smokers are in the dataset?" → SQL count
2. "Compare lab results across readmitted vs non-readmitted patients" → Chart (boxplot)
3. "What are the main risk factors for COPD?" → Insight (SHAP)
4. "Predict COPD for 55 year old male, BMI 27.5, 3 medications, no exercise, poor diet" → Prediction
5. "What are the symptoms of seasonal allergies?" → RAG with citation
6. "Summarize treatment plan for diabetic patients over 60" → RAG summarization

## Pydantic Models Reference

Key models (full definitions to be implemented in `src/schemas/`):

- **API:** `ChatRequest`, `ChatResponse`, `ChatMessage`, `ResponseMetadata`, `TokenUsage`
- **Routing:** `RouteDecision`, `AgentState`
- **Data:** `DatasetSchema`, `ColumnSchema`, `DataProfile`, `SqlQueryRequest`, `SqlResult`
- **Statistics:** `StatisticsRequest`, `StatisticsResult`
- **Charts:** `ChartSpec`, `ChartOutput`, `ChartResult`
- **Insights:** `InsightArtifact`, `InsightRequest`, `InsightResult`, `FeatureImportance`
- **Prediction:** `PatientFeatures`, `PredictionRequest`, `PredictionOutput`, `PredictionResult`
- **RAG:** `DocumentChunk`, `RetrievalRequest`, `Citation`, `RagResult`
- **Mapping:** `FeatureMapping`, `MappedFeatures`

## Guardrails

1. Numeric answers only from tools — LLM must not invent counts or predictions
2. COPD is 4-class (A/B/C/D) — not binary
3. Prediction transparency — disclose which values were defaulted
4. RAG citations mandatory for document-based answers
5. Disclaimer for treatment suggestions — internal prototype, not clinical advice
6. Pinned dependencies, random seed for ML reproducibility
