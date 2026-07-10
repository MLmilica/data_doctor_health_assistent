# Data Insights and Model Decisions

This document explains **what we learned from the data** and **why the final models look the way they do**. It is written for reviewers evaluating the take-home exercise — not as API documentation, but as the reasoning behind analytical and modeling choices.

**Source notebooks:** `notebooks/01_eda.ipynb`, `notebooks/02_ml_development.ipynb`  
**Production implementation:** `src/ml/train.py`, `src/ml/features.py`

---

## 1. Two separate worlds of data

The exercise provides two data sources that must be understood independently:

| Source | What it is | How the app uses it |
|--------|------------|---------------------|
| **Patient CSV** (`patient_data.csv`) | 10,000 structured rows, 18 columns | ML training, SQL analytics, charts |
| **Clinical documents** (`data/documents/`, ~1,050 `.md` files) | Unstructured notes, guidelines, summaries | RAG retrieval only |

**Key insight:** there is **no join** between a CSV row and a document file. Patient IDs in the spreadsheet do not map to document filenames. A question like “What does the guideline say about this patient’s COPD class?” cannot be answered by linking tables — only by routing to RAG (documents) and prediction (CSV) as separate steps.

This separation shaped the architecture: the CSV powers quantitative answers; documents power qualitative, citation-style answers.

---

## 2. What the patient dataset looks like

### Scale and structure

- **10,000 patients**, **18 columns** — demographics, lifestyle, labs, utilization flags, and two targets.
- Two **targets** drive all supervised learning:
  - `chronic_obstructive_pulmonary_disease` — four ordered severity classes (**A, B, C, D**), roughly balanced.
  - `alanine_aminotransferase` (ALT) — continuous lab value, approximately **10–44** in this synthetic cohort.

### Targets behave very differently

**COPD** is a multiclass classification problem with four equally plausible outcomes at baseline (~25% accuracy if guessing randomly). There is no obvious dominant feature in univariate analysis.

**ALT** is a regression problem where one variable — **BMI** — explains almost everything. Pearson correlation between BMI and ALT is approximately **0.9998**. This is not a subtle relationship; it dominates any model comparison.

That asymmetry is the central story of this dataset: one target is nearly unpredictable with available features; the other is nearly a deterministic function of BMI.

---

## 3. Exploratory insights that shaped decisions

### 3.1 Outliers — present but not actionable

We tested IQR and z-score rules on all continuous features. Outlier rates were low across the board:

- `age` and `albumin_globulin_ratio`: **zero** outliers by both methods.
- Highest rates: `medication_count` and `days_hospitalized` (~1.2% by IQR).
- `bmi` and `last_lab_glucose`: ~0.6–0.8%.

**Decision:** no removal, no winsorization. Extreme values are rare and clinically plausible. Tree models tolerate them; Ridge on ALT did not show instability. Aggressive cleaning would remove little data and risk distorting the one strong signal (BMI).

### 3.2 Multicollinearity — not a problem

Among continuous features, no pair exceeded **|r| > 0.95** (except BMI vs ALT, which is a target relationship, not feature redundancy). We did not drop features preemptively for collinearity.

### 3.3 What actually associates with each target?

| Target | Strongest univariate signals | Interpretation |
|--------|------------------------------|----------------|
| **COPD** | `diet_quality` (χ² p ≈ 0.036 — only categorical passing significance); MI leaders: `income_bracket`, `urban` | Weak, fragmented signal. No continuous feature was significant in Kruskal–Wallis tests. |
| **ALT** | `bmi` (r ≈ 0.9998); `readmitted` (ANOVA p ≈ 0.01) | ALT is effectively a BMI proxy in this synthetic data. Other features add marginal value. |

**Insight:** univariate tests alone would suggest dropping most features for COPD (nothing significant) and using only BMI for ALT. We still ran multivariate models because combinations can matter — but for this dataset, they largely did not change the picture.

### 3.4 Features we deliberately excluded from final models

Not every CSV column became a model input. After three experimental phases (see §5), the production models use **6 features per target**, not all 15 predictors.

| Excluded from models | Why |
|----------------------|-----|
| `patient_id` | Identifier only |
| `age`, `sex` | No strong univariate or MI signal for either target in this dataset |
| `medication_count`, `days_hospitalized` | Weak association; utilization proxies did not improve holdout metrics |
| `last_lab_glucose` | Redundant with metabolic patterns already captured indirectly; did not rank highly in MI |
| `bmi` (COPD model only) | Dominates ALT, not COPD — kept out of COPD feature set intentionally |
| Lifestyle/lab features not in Phase 3 shortlist | Did not improve COPD beyond random baseline; marginal for ALT beyond BMI |

**Important caveat for reviewers:** assignment questions may mention `age`, `sex`, or `medication_count`. Those columns exist in the CSV and are queryable via SQL, but they are **not** inputs to the deployed ML models. This is an intentional modeling choice, not an oversight.

---

## 4. How we experimented before committing

Model development followed three phases in `02_ml_development.ipynb`, each answering a specific question.

### Phase 1 — Baseline comparison (all 15 features)

**Question:** Do simple models work at all?

| Target | Models tested | Best result | Conclusion |
|--------|---------------|-------------|------------|
| COPD | Logistic Regression vs Random Forest | RF marginally better: acc ≈ 0.252, macro F1 ≈ 0.251 | Both at **random chance** (~0.25 for 4 classes) |
| ALT | Ridge vs Random Forest | Ridge: MAE ≈ 0.082, R² ≈ 0.9996 | Excellent metrics, but **BMI-only Ridge** achieves MAE ≈ 0.081 — essentially identical |

**Insight:** throwing all features at the problem did not help COPD. ALT looked “perfect” but was trivially explained by BMI.

### Phase 2 — Stronger algorithms (still all 15 features)

**Question:** Can gradient boosting rescue COPD?

Added LightGBM and XGBoost with **balanced class weights** for COPD; XGBoost regressor for ALT.

**Result:** COPD accuracy remained near **0.25** across all four classifiers (LogReg, RF, LightGBM, XGBoost). Boosting and class balancing did not materially beat random guessing. ALT remained Ridge-dominated.

**Insight:** the COPD limitation is a **data signal problem**, not a model family problem. No amount of algorithm switching on the full feature set moved the needle.

### Phase 3 — Target-specific reduced feature sets (6 each)

**Question:** Can we build a focused, explainable model without sacrificing performance?

We selected six features per target based on EDA (significance tests, mutual information, clinical plausibility):

**COPD features (all categorical):**

| Feature | Rationale |
|---------|-----------|
| `diet_quality` | Only feature with significant χ² vs COPD |
| `income_bracket` | Highest mutual information for COPD |
| `urban` | Strong MI; socioeconomic / care-access proxy |
| `diagnosis_code` | Clinical grouping with solid COPD MI |
| `exercise_frequency` | Lifestyle signal in MI ranking |
| `smoker` | Clinically relevant for pulmonary outcomes |

**ALT features (2 numeric + 4 categorical):**

| Feature | Rationale |
|---------|-----------|
| `bmi` | Dominant predictor (r ≈ 0.9998) |
| `readmitted` | Only categorical with significant ANOVA vs ALT |
| `exercise_frequency` | Top non-BMI MI for ALT |
| `albumin_globulin_ratio` | Lab marker; second-tier MI |
| `diagnosis_code` | Clinical context for liver/metabolic patterns |
| `diet_quality` | Moderate ALT MI; lifestyle factor |

**Result:** Phase 3 did not improve COPD beyond random baseline. ALT performance remained equivalent to BMI-only models. The value of Phase 3 was **clarity and focus** — smaller feature sets that are easier to explain in chat and in SHAP summaries, not higher accuracy.

---

## 5. Final models and why we chose them

These decisions are codified in `src/ml/train.py` and `src/ml/features.py`.

| Target | Final model | Features | Primary rationale |
|--------|-------------|----------|-------------------|
| **COPD** (4-class) | **XGBoost classifier** with balanced sample weights + label encoding (A/B/C/D → 0–3) | 6 categorical (Phase 3 set) | Best among candidates that were all near baseline; handles mixed categorical inputs well; class weights address mild imbalance |
| **ALT** (regression) | **Ridge regression** (α = 1.0) | 6 features (Phase 3 set) | Slightly better MAE/RMSE than Random Forest or XGBoost regressor; simpler and more interpretable |

### Why not use all 15 features?

Including weak or irrelevant predictors did not improve holdout metrics. A smaller feature set:

- Reduces overfitting risk on a synthetic dataset with weak COPD signal.
- Simplifies the prediction agent’s extraction and imputation logic.
- Makes offline SHAP summaries easier to communicate to analysts.

### Why XGBoost for COPD if accuracy is still ~25%?

Because **every candidate performed equally poorly**. XGBoost was selected as the least bad option with a plausible path to improvement if better data arrives — not because it “solved” COPD. The production pipeline records this explicitly (`near_random_baseline` flag in `ml_metrics.json`).

### Why Ridge for ALT if BMI alone is enough?

Ridge with six features performs identically to BMI-only Ridge on holdout data. We kept the six-feature version because:

- It includes clinically meaningful context (`readmitted`, `diagnosis_code`, etc.) for explanation even when BMI dominates the coefficient.
- It provides a consistent feature contract for the chat UI and form (users can supply more than BMI).
- It documents honest secondary drivers in SHAP artifacts, even when their marginal contribution is small.

---

## 6. Honest assessment of model quality

### COPD — integrated but not reliable

- Holdout **accuracy and macro F1 ≈ 0.25** — equivalent to random guessing among four balanced classes.
- Confusion matrices show no clear class structure; LogReg, RF, LightGBM, and XGBoost are nearly identical.
- This aligns with EDA: only `diet_quality` reached conventional significance; no continuous feature separated classes.

**What this means in the product:** COPD predictions are exposed in the chat and form for demonstration, but responses should carry a **limitation disclaimer**. The model is not fit for clinical decision support. Future work would require richer features, more data, or reformulating the target (e.g. binary severe vs non-severe).

### ALT — excellent metrics, trivial mechanism

- Holdout **R² ≈ 0.9996**, **MAE ≈ 0.08** on a range of ~10–44.
- A **BMI-only** model achieves the same error.
- The synthetic dataset encodes ALT as an almost deterministic function of BMI.

**What this means in the product:** ALT predictions look impressive numerically but should be interpreted as **“BMI-driven estimate in a synthetic cohort”**, not evidence of a complex multivariate liver model. The chat insight tool correctly surfaces global feature importance — BMI will dominate — but per-patient explanations would largely repeat the same story.

---

## 7. Preprocessing and inference choices

These decisions connect modeling to the chat experience:

| Choice | Reasoning |
|--------|-----------|
| **Stratified train/test split for COPD** | Preserve balanced A/B/C/D in both sets |
| **Random split for ALT** | Continuous target; stratification not applicable |
| **Refit on full data after holdout eval** | Maximize data for deployed artifacts |
| **Median imputation from `data_profile.json`** | Transparent defaults when users omit optional fields in chat |
| **Required vs optional features** | COPD requires `diet_quality` + `exercise_frequency`; ALT requires `bmi` — prevents silent garbage predictions |
| **Offline SHAP at train time** | Global explanations without per-request latency in the POC |

---

## 8. Clinical documents — what EDA told us (briefly)

`01_eda.ipynb` also parsed the document corpus (~1,050 files). Documents are heterogeneous markdown briefs with section structure (title, condition, treatment, etc.). They are **not** aligned to CSV patient IDs.

**Insight for modeling:** documents do not improve ML features. They are a parallel knowledge source for RAG. Questions about “what guidelines recommend” should not expect the prediction model to know document content — that is a routing problem, not a feature engineering problem.

---

## 9. Summary for reviewers

| Question | Answer |
|----------|--------|
| **What did you learn about the data?** | Two disconnected sources (CSV vs documents). COPD has weak signal; ALT is BMI-dominated. Outliers and multicollinearity are non-issues. |
| **What models did you build?** | XGBoost multiclass (COPD) and Ridge regression (ALT), each on a curated 6-feature set. |
| **Why those features?** | EDA-driven shortlists per target — significance tests, mutual information, clinical plausibility — not “use everything in the CSV.” |
| **Why those algorithms?** | Compared 2–4 candidates per target across three phases; final picks balance holdout metrics, interpretability, and pipeline simplicity. |
| **Are the models good?** | ALT metrics are excellent but mechanistically simple. COPD is near random — documented and disclaimed, not hidden. |
| **What would you do next?** | See [FUTURE_WORK.md](../FUTURE_WORK.md) — feature engineering, target reformulation, per-prediction SHAP, eval datasets, and honest model cards. |

The notebooks contain the full experimental trail (plots, candidate tables, phase comparisons). This document captures the **reasoning** that turned exploration into the models now serving the Data Doctor chat and form.

---

*Synthetic data only — not for clinical use.*
