# Data Doctor — Submission

Take-home submission for the Loka **Machine Learning Engineer** exercise. This document explains what was built, which code implements each part, key conclusions, and known limitations.

**Companion docs:** [README.md](README.md) (quick start) · [ARCHITECTURE.md](ARCHITECTURE.md) (system design + AWS) · [FUTURE_WORK.md](FUTURE_WORK.md) (planned improvements)

---

## Executive summary

Data Doctor is a working POC that:

- Trains and serves **COPD** (multiclass) and **ALT** (regression) models over a synthetic patient CSV
- Answers **natural-language questions** via a **LangGraph multi-agent** system (prediction, SQL/chart/insight, RAG)
- **Cites indexed clinical documents** where appropriate
- Supports **multi-step** questions in a single chat turn (e.g. dataset analytics + prediction)
- Persists **session memory** for follow-up prediction questions
- Includes **144 automated tests** and optional **LangSmith** tracing

The prototype is **not production-ready** by design. Trade-offs favor demonstrable end-to-end behavior, clear architecture, and a documented path to AWS.

---

## 1. Data exploration and understanding

### What was done

| Activity | Output |
|----------|--------|
| Exploratory analysis | `notebooks/01_eda.ipynb` |
| ML feature experiments | `notebooks/02_ml_development.ipynb` |
| Dataset profile (stats, medians for imputation) | `artifacts/data_profile.json` via `main.py` |
| Schema for agents/SQL | `src/data/schema_registry.py` — 18 columns, types, allowed values |
| DuckDB access | `src/data/loader.py` — loads CSV as `patients` table |

### Key files

| File | Role |
|------|------|
| `src/data/profile.py` | Build/load `DataProfile` |
| `src/data/schema_registry.py` | Column definitions for SQL LLM prompts |
| `src/data/loader.py` | PatientDataLoader + DuckDB queries |
| `notebooks/01_eda.ipynb` | Distributions, correlations, target analysis |

### Conclusions

- **10,000 patients**, 18 features including COPD severity (A–D) and ALT as targets
- EDA drove a **reduced feature set** for models (not every CSV column is a predictor)
- `data_profile.json` powers transparent **default imputation** when optional fields are missing at inference time
- Clinical `.md` documents are a **separate knowledge source** from the CSV — no row-level join

**Full reasoning (data insights, model phases, limitations):** [docs/DATA_PREDICT.md](docs/DATA_PREDICT.md)

---

## 2. Machine learning — prediction agent

### What was done

| Component | Implementation |
|-----------|----------------|
| Training pipeline | `src/ml/train.py` |
| Inference | `src/ml/predict.py`, `ModelRegistry` |
| Feature contract | `src/ml/features.py` |
| NL → features (chat) | `src/agents/subagents/prediction_agent.py` |
| Direct form (no LLM) | `ui/app.py` Form tab → `src/ml/feature_mapper.py` |
| Global factor hints | `load_top_global_factors()` reads offline SHAP JSON |

### Models chosen

| Target | Model | Rationale |
|--------|-------|-----------|
| **COPD** (`chronic_obstructive_pulmonary_disease`) | **XGBoost** multiclass | Handles mixed categoricals, class imbalance via sample weights |
| **ALT** (`alanine_aminotransferase`) | **Ridge** regression | Strong linear BMI relationship in EDA; simple, interpretable POC |

Artifacts: `artifacts/models/copd_pipeline.joblib`, `alt_pipeline.joblib`, `copd_label_encoder.joblib`

### Feature sets (what actually enters the model)

**COPD** — all categorical:

| Feature | Required? |
|---------|-----------|
| `diet_quality` | **Yes** |
| `exercise_frequency` | **Yes** |
| `income_bracket` | No — median imputed |
| `urban` | No |
| `diagnosis_code` | No |
| `smoker` | No |

**ALT**:

| Feature | Required? |
|---------|-----------|
| `bmi` | **Yes** |
| `readmitted`, `exercise_frequency`, `albumin_globulin_ratio`, `diagnosis_code`, `diet_quality` | No — imputed from profile when missing |

**Not in models** (may appear in user questions or LLM extraction): `age`, `sex`, `medication_count`, `days_hospitalized`, lab columns beyond those listed.

### Chat flow

```text
User → LLM extraction (LLMPredictionExtraction)
     → merge session facts (follow-up BMI, last_target)
     → run_prediction()
     → LLM synthesis (uses ML numbers only — never invents prediction)
```

### What is stored for the orchestrator / memory

After a prediction turn, `SessionFacts` (`src/schemas/memory.py`) may contain:

- `last_target` — `copd`, `alt`, or `both`
- `last_features` — merged feature dict for follow-ups
- `missing_required` — if prediction could not complete
- `last_route` — `prediction`

This enables follow-ups like *"What if BMI is 35?"* without re-stating the full patient description.

### Conclusions

- Assignment prediction questions **work** when the message maps to model features (BMI, diet, exercise, smoker, readmitted, etc.)
- Age/sex/medications in the assignment wording are **extracted for transparency** but **do not change** the model output — this must be explained in a demo
- ALT is largely **BMI-driven** in this dataset (visible in holdout metrics and SHAP)
- Offline SHAP summaries (`artifacts/insights/`) support `top_global_factors` in the prediction UI expander

---

## 3. Data agent

### What was done

One graph node, three internal tools:

| Path | Trigger examples | Tool |
|------|------------------|------|
| **SQL** | "How many smokers?", "males over 40 readmitted" | `sql_layer` |
| **Chart** | "Compare glucose… readmitted", "distribution of BMI" | `chart_tool` |
| **Insight** | "Main risk factors for COPD", "what drives ALT" | `insight_tool` |

Classifier: `src/agents/tools/data_task_classifier.py`  
Orchestrator: `src/agents/orchestrator.py` (insight signals beat broad RAG phrases)

### SQL layer

```text
┌─────────────┐     ┌──────────────┐     ┌─────────┐     ┌────────────────┐
│ LLM SQL     │────▶│ SqlLayer     │────▶│ DuckDB  │────▶│ LLM synthesis  │
│ extraction  │     │ validate_sql │     │ patients│     │ (facts only)   │
└─────────────┘     └──────────────┘     └─────────┘     └────────────────┘
```

| File | Role |
|------|------|
| `src/agents/subagents/data_agent.py` | Agent node, synthesis prompts |
| `src/agents/tools/sql_layer.py` | Validation + execution |
| `src/schemas/sql.py` | `LLMSQLExtraction`, `DataQueryResult` |

**Guardrail:** Numeric answers come from SQL results JSON — the synthesis LLM must not invent counts or averages.

### Chart tool

```text
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│ LLM ChartSpec│────▶│ SqlLayer     │────▶│ create_boxplot()  │
│ (no Plotly)  │     │ deterministic│     │ create_histogram()│
└──────────────┘     │ SELECT       │     │ … (6 renderers)   │
                     └──────────────┘     └───────────────────┘
```

| File | Role |
|------|------|
| `src/agents/tools/chart_tool.py` | Spec validation, SQL build, Plotly JSON |
| `src/schemas/chart.py` | `ChartSpec`, `ChartResult` |

### Insight tool

| File | Role |
|------|------|
| `src/agents/tools/insight_tool.py` | Load SHAP/importance JSON, synthesis |
| `src/ml/shap_insights.py` | Offline SHAP during `ml.train` |
| `artifacts/insights/copd_shap_summary.json` | Pre-computed rankings |

**Not implemented:** dynamic subgroup analysis (e.g. "risk factors for readmitted patients over 60") — documented as future work.

### Conclusions

- SQL path covers all **count/aggregate** assignment questions
- Chart path covers assignment #9 (*compare lab results*) via boxplot
- Insight path covers the assignment bullet on **risk factors / feature patterns**
- `statistics_tool.py` remains a stub — SQL aggregates are sufficient for POC

---

## 4. RAG agent

### Indexing and vectorization

| Step | Implementation |
|------|----------------|
| Parse `.md` by `## ` sections | `src/data/document_parser.py` |
| Embed + store | `src/data/vectorstore.py` — Chroma persistent client |
| Embedding model | OpenAI `text-embedding-3-small` (config: `rag_embedding_model`) |
| Index script | `scripts/index_documents.py` |

Each chunk metadata: `source_file`, `section_name`, `diagnosis_codes`.

### Retrieval pipeline (corrective + self-RAG)

```text
Retrieve (top-k from Chroma)
    → LLM chunk grading (corrective RAG — drop irrelevant chunks)
    → LLM synthesis with citations
    → LLM grounding check (self-RAG)
    → optional strict retry if not grounded
```

| File | Role |
|------|------|
| `src/agents/subagents/rag_agent.py` | Full pipeline |
| `src/agents/tools/rag_retrieval.py` | Vector search wrapper |
| `src/schemas/rag.py` | Grading/grounding schemas |

### Conclusions

- Document questions route to RAG with **mandatory citation** metadata in the UI
- Corpus is **synthetic briefs** — not a clinical guideline library; some assignment questions (seasonal allergy symptoms, heart attack medications) may return honest "not found" responses
- This is acceptable for POC when the system **attempts search** and does not hallucinate sources

---

## 5. Memory

### What memory contains

| Concept | Schema | Purpose |
|---------|--------|---------|
| Transcript | `ChatTurn` | User/assistant messages for routing prompts |
| Step ledger | `StepRecord` | What each agent did (`agent`, `tool`, artifact summary) |
| Session facts | `SessionFacts` | Structured follow-up context (features, target, last SQL) |

### Where it is stored (POC)

| Store | Location | Lifetime |
|-------|----------|----------|
| `InMemorySessionStore` | Process RAM | Until API restart |
| LangGraph state | Per `graph.invoke` | Single request |
| Artifacts/models/SHAP | `artifacts/` on disk | Persistent |
| Chroma index | `data/chroma/` | Persistent |

**Key files:** `src/memory/session_store.py`, `src/memory/persistence.py`, `src/memory/context.py`

### Why this split

- **Transcript** — natural language context for orchestrator and prediction extraction
- **Session facts** — deterministic merge for follow-ups without re-parsing full history
- **Step ledger** — audit trail and multi-step synthesize input; also powers sidebar/history in UI

### Conclusions

- Follow-up prediction works within the same `session_id`
- Restarting the API clears sessions — documented limitation
- AWS path: DynamoDB for sessions, optional ElastiCache for hot state (see [ARCHITECTURE.md](ARCHITECTURE.md))

**Full reasoning (three memory layers, follow-ups, limits):** [docs/ORCHESTRATION_MEMORY.md](docs/ORCHESTRATION_MEMORY.md)

---

## 6. Orchestration

### Why this design

| Decision | Reason |
|----------|--------|
| LangGraph | Explicit multi-agent loop, testable nodes, LangSmith integration |
| Rules + LLM routing | Fast path for clear intents; LLM for ambiguous combo questions |
| Specialists loop back to orchestrator | Enables multi-step without hard-coded pipelines |
| Synthesize node | Single user-facing answer when ≥2 agents run |
| Chart/insight inside `data` | Avoids graph explosion; assignment analytics still routes to `data` |
| Insight before RAG on "risk factors" | Prevents document search when user asks about **model drivers** |

### Routing examples

| Question | Route | Why |
|----------|-------|-----|
| How many smokers? | `data` → sql | Analytics signal |
| Compare lab results readmitted vs not | `data` → chart | Compare + no SQL-only aggregate |
| Main risk factors for COPD? | `data` → insight | Model insight signal overrides "what are the" |
| Symptoms of seasonal allergies? | `rag` | Clinical knowledge, not CSV/ML |
| Predict ALT for BMI 30 | `prediction` | Explicit prediction intent |
| Avg BMI + ALT for BMI 30 | `multi` → data then prediction → synthesize | Multiple required agents |

**Key files:** `src/agents/graph.py`, `src/agents/orchestrator.py`, `src/agents/multi_step.py`, `src/agents/subagents/synthesize_agent.py`

**Full reasoning (routing, multi-step, memory layers):** [docs/ORCHESTRATION_MEMORY.md](docs/ORCHESTRATION_MEMORY.md)

---

## 7. Guardrails and fallback

| Check | File |
|-------|------|
| Input guardrails | `src/agents/guardrails.py` |
| Fallback responses | `src/agents/subagents/fallback_agent.py` |
| Low-confidence routing | `src/agents/orchestrator.py` → `_apply_low_confidence` |

Blocks direct prescribing/diagnosis requests; returns helpful clarification for vague messages.

---

## 8. LangSmith tracing

| Item | Detail |
|------|--------|
| Config | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` in `.env` |
| Setup | `src/observability/langsmith.py` — synced at API startup |
| Trace grouping | `thread_id = {user_id}:{session_id}` per chat invoke |
| Health check | `/health` returns `langsmith_tracing: true/false` |

Disable anytime with `LANGCHAIN_TRACING_V2=false` (no code change).

---

## 9. Testing and quality

| Layer | Coverage |
|-------|----------|
| Unit tests | Agents, SQL validation, chart/insight tools, memory, schemas |
| Graph tests | Routing, multi-step, mocked LLM |
| ML tests | Training artifacts, inference, feature contract |
| **Total** | **144 tests** (`uv run pytest`) |

**Smoke scripts (real LLM):**

```bash
uv run python scripts/smoke_chat_graph.py --example data --expect-route data
uv run python scripts/smoke_prediction_agent.py
```

---

## 10. Assignment question checklist

Prerequisites: trained models, indexed documents, LLM key in `.env`, API + UI running.

| # | Prompt (short) | Route | Status | Notes |
|---|----------------|-------|--------|-------|
| 1 | COPD, 55yo male, BMI 27.5… | `prediction` | ✅ | Model uses diet/exercise/smoker etc.; age/sex not in model |
| 2 | ALT, woman 44, hospital 5 days… | `prediction` | ⚠️ | athlete→High, urban, readmitted work; age/hospital days not in model |
| 3 | How many smokers? | `data` sql | ✅ | |
| 4 | Males >40 readmitted | `data` sql | ✅ | |
| 5 | Heart attack medications | `rag` | ⚠️ | Routes correctly; corpus may not contain answer |
| 6 | >5 medications | `data` sql | ✅ | |
| 7 | Seasonal allergy symptoms | `rag` | ⚠️ | Routes correctly; answer quality depends on corpus |
| 8 | Diabetic treatment plan 60+ | `rag` | ✅ / ⚠️ | Treatment-plan chunks when indexed |
| 9 | Compare lab results readmitted | `data` chart | ✅ | Boxplot via chart tool |

Full copy-paste prompts for the assignment table: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) (assignment section).

### Additional test prompts

Extended manual test catalog from [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) — single-step routing, multi-step orchestration, and chart/insight tools.

| Category | # | Prompt | Expected `routed_to` | Data tool / Synthesize | What to verify |
|----------|---|--------|----------------------|--------------------------|----------------|
| **Single-step** | 1 | How many patients are in each income bracket? | `data` | sql / No | SQL `GROUP BY income_bracket`; **Data query details** expander |
| **Single-step** | 2 | Predict ALT for a patient with BMI 30 | `prediction` | — / No | ALT value + `prediction` metadata |
| **Single-step** | 3 | What does the COPD guideline say about exercise? | `rag` | — / No | Citations or honest not-in-documents |
| **Single-step** | 4 | Compare readmission counts by month | `data` | sql / No | May ask clarification (no month column in dataset) |
| **Multi-step** | 5 | Compare average BMI in the dataset with ALT prediction for BMI 30 | `multi` | sql + prediction / **Yes** | One answer; `data_query` + `prediction` metadata |
| **Multi-step** | 6 | What is the average BMI in the dataset and what do documents recommend for low-impact exercise? | `multi` | sql + rag / **Yes** | Dataset metric + document recommendations |
| **Multi-step** | 7 | Predict COPD for good diet and moderate exercise, and summarize what the guideline says about diet | `multi` | prediction + rag / **Yes** | COPD class + document summary; no `data` agent |
| **Multi-step** | 8 | Compare average BMI in the dataset, predict ALT for BMI 30, and what do documents say about exercise? | `multi` | sql + prediction + rag / **Yes** | All three sources synthesized; max 3 specialists |
| **Multi-step** | 9 | What medication should the patient take for COPD? | `fallback` | — / No | `guardrail_blocked: true`; no specialist loop |
| **Multi-step** | 10 | Tell me something interesting about patients | `fallback` | — / No | Clarification or help text |
| **Chart** | 11 | Compare glucose levels between readmitted and non-readmitted patients. | `data` | chart (boxplot) / No | **Chart** expander; `x=readmitted`, `y=last_lab_glucose` |
| **Chart** | 12 | Show the distribution of BMI in the dataset. | `data` | chart (histogram) / No | Histogram on `bmi` |
| **Chart** | 13 | Show the relationship between BMI and alanine aminotransferase. | `data` | chart (scatter) / No | Scatter `bmi` vs `alanine_aminotransferase` |
| **Insight** | 14 | What are the main risk factors for COPD? | `data` | insight (copd) / No | **Model insights** expander; not RAG |
| **Insight** | 15 | What drives ALT predictions in this dataset? | `data` | insight (alt) / No | Top features dominated by `bmi` |
| **Insight** | 16 | Compare COPD and ALT risk factors | `data` | insight / No | Clarification — specify COPD or ALT |
| **Memory** | 17 | Turn 1: `Predict ALT for BMI 28` → Turn 2: `What if BMI is 35?` | `prediction` (both) | — / No | Same `session_id`; BMI 35 in second prediction |
| **Insight** | 18 | Turn 1: `Predict ALT for BMI 20` → Turn 2: `What are the main risk factors for ALT?` | `prediction` → `data` | insight (alt) / No | Turn 2 must not route to RAG |

**Prerequisites for chart/insight rows:** `uv run python -m ml.train` (SHAP artifacts). Restart API after routing changes.

More detail and smoke commands: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) — sections *Multi-step Orchestrator*, *Chart & Insight tool test prompts*.

---

## 11. If I had another week

1. **Persistent memory** — DynamoDB session store + LangGraph checkpointer
2. **E2E pytest** for all 9 assignment prompts (recorded routing expectations)
3. **RAG corpus curation** — seed chunks for allergy symptoms and medication lists
4. **AWS deploy sketch** — CDK/Terraform for API + S3 artifacts + Bedrock
5. **Dynamic insight** — filtered cohort SQL + subgroup feature importance
6. **Evaluator dashboard** — extend Streamlit sidebar with step ledger from API debug payload
7. **Model cards** — document COPD accuracy / ALT MAE in SUBMISSION with holdout numbers from `artifacts/insights/ml_metrics.json`

---

## 12. How to evaluate this submission

1. Read [README.md](README.md) — setup and run Chat + Form
2. Skim [ARCHITECTURE.md](ARCHITECTURE.md) — graph, AWS map
3. Run `uv run pytest`
4. Try assignment prompts in Streamlit Chat (sidebar shows agent/tool per turn)
5. Optional: enable LangSmith and inspect `chat_graph` traces

---

*Built as a POC for the Loka Data Doctor take-home exercise. Synthetic data only — not for clinical use.*
