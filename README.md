# Customer Intelligence & Support Copilot — Teaching Kit

> Educational implementation of the supplied fictional Instamart-style capstone. All generated records are synthetic; no real customer data, hosted LLM, API key, or network call is needed.

## Learning outcomes
Students learn to (1) fetch/read CSV exports and audit data, (2) clean and persist trusted tables in SQLite, (3) write SQL and pandas analytics, (4) segment customers and train a leakage-aware lapse model, (5) redact PII and triage text with retries/output validation, and (6) retrieve policy passages and execute bounded database-backed tools.

## Setup and run
Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_project.py
```

Open `notebooks/Customer_Intelligence_Support_Copilot.ipynb` and run cells from top to bottom. The notebook calls the same modules used by the scripts.

## Repository map
- `generate_data.py`: deterministic fictional CSV and policy generation.
- `mock_llm.py`: offline stand-in for classification, policy answers and agent tool calls.
- `src/data_pipeline.py`: audit, clean, SQLite load, validation.
- `src/customer_models.py`: RFM, K-means, PCA, lapse model with train/test separation.
- `src/triage.py`: token guardrails, PII redaction, retries, response validation, routing.
- `src/policy_rag.py`: chunking, TF-IDF retrieval and grounded policy evaluation.
- `src/refund_agent.py`: order lookup, refund calculation, ledger and bounded tool loop.
- `sql/queries.sql`: seven SQL learning patterns.
- `outputs/`: generated artifacts; `state/`: agent checkpoint examples.

## How students should learn the data-fetching flow
1. `pd.read_csv("data/raw/orders.csv", keep_default_na=False)` loads local export rows. In a production project, a fetch function might download from an approved endpoint or query a warehouse, but this exercise intentionally uses local synthetic data and has no credentials.
2. Inspect shape, data types, null/blank counts, duplicate rows, keys and impossible values before cleaning.
3. Apply one documented cleaning rule at a time and compare row/value counts before and after.
4. Write clean CSVs for easy inspection and SQLite tables for repeatable SQL queries.
5. Run assertions before downstream modelling. A failed assertion should stop the pipeline rather than silently propagate bad data.

## Part A — Data foundation
The pipeline standardizes city/gender, treats invalid ages as missing, resolves duplicate orders by keeping the latest timestamp, imputes missing amounts from catalogue price, sets negative delivery times to missing and imputes only completed-order delivery durations by product category. Review every rule before teaching; the capstone handout's numeric checkpoints should be used as acceptance tests.

## Part B — Segments and lapse
RFM uses completed orders and a fixed snapshot date. K-means diagnostics compare K=2..8; the notebook starts with K=4 for interpretable teaching. Lapse features use orders strictly before 2025-09-01, while the label uses September outcomes. The pipeline uses a preprocessing Pipeline fitted on training rows, GridSearchCV and a saved joblib model. The Premium retention list is illustrative only because the Premium label uses the whole RFM window and is post-hoc; it is not a leakage-free evaluation.

## Part C — Ticket triage and privacy
Empty and overlong tickets are skipped before model calls. Email, phone and order ID patterns are redacted before prompting. Retries handle simulated transient errors. JSON category values are allow-listed; malformed outputs are accepted only when exactly one allowed category is detectable using `re.findall`. Human review receives guardrail/invalid/budget cases, fraud goes to security, premium customers go to the premium desk, and the rest go to the support queue.

## Part D — Policy retrieval and refund tools
Policy documents are chunked with overlap, represented with `TfidfVectorizer`, and ranked using cosine similarity. The offline mock returns context-derived sentences. Refund rules are implemented in Python tools backed by SQLite: completed orders delivered in over 45 minutes qualify for a full refund; orders over INR 1000 escalate instead of auto-issuing. This is a teaching simulation, not a payment integration.

## GitHub suggestions
These public repositories/documentation projects are useful for learning concepts; do not copy a student's capstone solution. Adapt the patterns and cite the source in your notes:
- pandas documentation: https://github.com/pandas-dev/pandas — CSV loading, cleaning, groupby, pivot tables.
- scikit-learn: https://github.com/scikit-learn/scikit-learn — pipelines, preprocessing, clustering, model selection, metrics and TF-IDF.
- Python tutorial: https://github.com/python/cpython/tree/main/Doc/tutorial — Python fundamentals.
- Jupyter Notebook: https://github.com/jupyter/notebook — notebook workflow.
- SQLite: https://github.com/sqlite/sqlite — SQL database engine (official mirror).
- scikit-learn text feature extraction guide: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

## Limitations and next steps
- This is a teaching implementation and may need adjustment to match every numeric checkpoint in a strict grading rubric; compare actual outputs with the supplied brief and investigate differences rather than hardcoding expected answers.
- The offline mock model is deterministic rule-based simulation, not a real language model.
- The refund ledger is a local JSONL audit trail, not a real transaction system.
- Production work would add access controls, secure secret management, monitoring, schema contracts, data lineage, model drift checks and human approval workflows.

## Part A – Data Audit, Cleaning, SQL and EDA

### 1. Raw Data Audit

The raw datasets were inspected for dimensions, data types, missing values, duplicates and invalid values.

Key findings:
- Customers: 1,200 records
- Products: 40 records
- Raw orders: 13,084 records
- Exact duplicate orders: 119
- Pending orders: 656
- Invalid order amounts: 120
- Negative delivery times: 106
- Blank delivery times: 2,541
- Blank customer ages: 61
- Invalid customer ages: 6
- Blank customer genders: 30

### 2. Data Cleaning

**Customers**
- Standardised city names using title case.
- Standardised gender values into male, female and unknown.
- Converted invalid ages into missing values.
- Imputed missing ages using the median age of the respective city.
- Converted signup dates into datetime format.

**Orders**
- Converted order timestamps into datetime format.
- Removed exact duplicate rows.
- Retained the latest timestamp for each order ID.
- Converted order amounts into numeric values.
- Imputed invalid or missing amounts using product price, quantity and discount.
- Converted negative delivery times into missing values.
- Imputed missing delivery times for completed orders using the completed-order median.
- Kept cancelled-order delivery times missing.
- Excluded pending orders from the cleaned order dataset.

### 3. Final Cleaned Data

- Customers: 1,200
- Products: 40
- Orders: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199
- Completed revenue: INR 2,998,668.80

The cleaned CSV files and SQLite database are stored in the `data` directory.

### 4. SQL Analysis

Seven SQL queries were executed using pandas `read_sql_query()`.

The analysis covered:
1. Filtering, grouping and HAVING.
2. Three-table joins and revenue analysis.
3. Subqueries.
4. Customers without completed orders.
5. Common Table Expressions.
6. Window functions and customer ranking.
7. Daily revenue running totals.

All seven query results are saved in the `outputs` directory.

### 5. Exploratory Data Analysis

The completed-order amount distribution is positively skewed, with a small number of high-value orders.

The correlation between order amount and discount percentage is approximately -0.095, indicating a very weak negative linear relationship. Correlation does not establish causation.

Detailed EDA statistics and city revenue summaries are available in `outputs/part_a_summary.txt` and `outputs/city_revenue_summary.csv`.

### 6. Validation

The cleaning validation script printed:

`ALL CHECKPOINTS PASSED`

The independent Part A verification script also completed its database and revenue assertions successfully.
## Part B – Customer Segmentation and Lapse Prediction

### 1. Customer Segmentation

Customer segmentation was performed using RFM analysis:

- **Recency:** Number of days since the customer's last completed order, calculated relative to 1 October 2025.
- **Frequency:** Number of distinct completed orders.
- **Monetary:** Total spending on completed orders in INR.

The analysis included 1,175 customers with at least one completed order.

The three RFM features were standardised using `StandardScaler`. K-Means clustering was evaluated for K values from 2 to 8, using `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.761 | 0.465 |
| 3 | 1268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

K=4 was selected because it achieved the highest silhouette score (0.526) and provided four interpretable customer groups. Although inertia decreases as K increases, lower inertia alone is not sufficient to select the number of clusters.

#### Segment profiles

| Segment | Customers | Average recency (days) | Average frequency | Average monetary (INR) |
|---|---:|---:|---:|---:|
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Frequent | 419 | 8.77 | 19.40 | 3,410.02 |
| At Risk – Low activity | 436 | 28.86 | 3.77 | 1,057.62 |
| At Risk – Inactive | 182 | 109.04 | 2.20 | 685.99 |

The two At Risk groups share the same segment label in the generated CSV. Their different profiles are distinguished in this report for business interpretation.

#### Recommended business actions

- **Premium:** VIP benefits, personalised offers and loyalty rewards.
- **Frequent:** Repeat-order promotions and subscription-style offers.
- **At Risk – Low activity:** Personalised product recommendations and re-engagement offers.
- **At Risk – Inactive:** Win-back campaigns and targeted incentives.

Cluster numbers are arbitrary identifiers and do not represent customer value rankings.

#### PCA analysis

The first two principal components explain approximately:

- Component 1: 63.43%
- Component 2: 21.80%
- Combined: 85.23%

PCA was used to understand how much variation in the standardised RFM data is represented by two dimensions.

Output: `outputs/customer_segments.csv`

### 2. Customer Lapse Prediction

The objective is to identify customers who may stop purchasing so that the business can take retention action.

**Prediction definition:**

- Historical feature cutoff: 1 September 2025.
- Prediction period: 1 September 2025 to 30 September 2025.
- A customer is labelled as lapsed if they had no completed order during the prediction period.
- Historical completed orders before the cutoff were used to create customer features.

The resulting cohort contained 1,156 customers, of whom 443 lapsed.

| Metric | Result |
|---|---:|
| Cohort size | 1,156 |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Training rows | 924 |
| Testing rows | 232 |
| Baseline accuracy | 61.64% |

#### Features

The model used the following historical features:

- Recency
- Frequency
- Monetary
- Average discount
- Cancellation rate
- Average delivery time
- Favourite product category
- City
- Age
- Gender
- Customer tenure in days

Customer IDs, September outcomes and RFM segment labels were not used as model features.

#### Model development

Two classification algorithms were evaluated:

1. Logistic Regression (`max_iter=1000`)
2. Random Forest (`n_estimators=100`, `random_state=42`)

Hyperparameter tuning used `GridSearchCV`, five-fold cross-validation and F1 scoring.

Numeric preprocessing used median imputation and `StandardScaler`. Categorical preprocessing used constant-value imputation and `OneHotEncoder(drop="first", handle_unknown="ignore")`.

The train/test split was performed before fitting preprocessing. The preprocessing steps were placed inside the model pipelines, ensuring they were fitted on training data only.

#### Class imbalance handling

The original models were compared with models using `RandomOverSampler(random_state=42)`.

Oversampling was placed inside an imbalanced-learn pipeline. It was applied only to training folds during cross-validation and to the training data during final fitting. Test data remained untouched.

#### Model comparison

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression – Original | 0.644 | 0.652 | 0.648 |
| Random Forest – Original | 0.633 | 0.640 | 0.637 |
| Logistic Regression – Oversampled | 0.573 | 0.843 | 0.682 |
| Random Forest – Oversampled | 0.570 | 0.865 | 0.688 |

The selected model was Logistic Regression with oversampling because it achieved the highest cross-validation F1-score (0.708). Model selection was based on training cross-validation performance, not test-set performance.

Oversampling improved lapse recall substantially. This is useful for customer retention because identifying more potentially lapsed customers can support proactive outreach, although it also increases false-positive predictions.

#### Selected model evaluation

Confusion matrix (actual rows, predicted columns; class 0 = not lapsed, class 1 = lapsed):

| | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 87 | 56 |
| Actual lapsed | 14 | 75 |

The model identified 75 of the 89 actual lapsed customers in the test set.

The saved model was reloaded and its predictions were verified against the original model predictions.

#### Premium customer retention list

The ten Premium customers with the highest predicted lapse probabilities were saved for retention outreach.

Output: `outputs/retention_list.csv`

**Interpretation limitation:** The Premium flag is based on the full RFM window ending 1 October 2025, including September activity. Therefore, the retention list is illustrative and is not a leakage-free evaluation of a prospective Premium customer list. Segment labels were not used as lapse-model features.

#### Part B output files

- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/lapse_model_comparison.csv`
- `outputs/lapse_model_report.json`

The improved implementation is available in `src/lapse_model_improved.py`.
# Customer segmentation
python -c "from src.customer_models import rfm_and_segments; rfm_and_segments(k=4)"

# Lapse prediction
python -c "from src.lapse_model_improved import build_lapse_model; build_lapse_model()"
## Part C – Customer Support Ticket Triage

### 1. Objective

Built an offline customer-support ticket classification pipeline using the supplied mock LLM. The system classifies tickets into four categories:

- Billing
- Delivery
- Technical
- Fraud

The pipeline also protects customer information, handles invalid responses, retries temporary failures, controls token usage, and routes tickets to appropriate support teams.

### 2. Dataset

- Total tickets: 164
- Dataset columns: ticket_id, customer_id, order_id, text, true_category
- Empty ticket messages: 4
- Long ticket messages: 4

### 3. Privacy and PII Redaction

Before sending a ticket to the mock LLM, the pipeline redacts:

| Personal information | Replacement | Redactions |
|---|---|---:|
| Email addresses | `[EMAIL]` | 55 |
| Mobile numbers | `[PHONE]` | 91 |
| Order IDs | `[ORDER_ID]` | 132 |
| **Total** | | **278** |

Redaction is performed before model classification. Original ticket text is not included in the final triage CSV.

### 4. Guardrails and Output Validation

- Empty messages are marked `skipped_empty`.
- Messages exceeding 400 estimated tokens are marked `flagged_long`.
- The model is instructed to return JSON containing a category and reason.
- Valid JSON responses are marked `classified_json`.
- If JSON parsing fails, a single allowed category found in the response can be recovered using keyword validation (`classified_keyword`).
- Invalid or ambiguous responses are marked `flagged_invalid_output`.
- Temporary model errors are retried up to three times using exponential backoff with jitter.

### 5. Classification Results

| Status | Tickets |
|---|---:|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |
| **Total** | **164** |

A total of 147 tickets were successfully classified.

### 6. Zero-Shot vs Few-Shot Comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100% |
| Total tokens | 12,988 | 22,972 |
| Tokens per classified ticket | 88.35 | 156.27 |
| Recovered transient errors | 19 | 19 |

Few-shot classification achieved higher accuracy because the system prompt included labelled examples covering different ticket categories. However, it required more tokens.

Zero-shot confusion matrix (category order: billing, delivery, technical, fraud):

| Actual / Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

### 7. Token Budget Demonstration

A separate zero-shot run was executed with a token budget of 6,000.

| Status | Tickets |
|---|---:|
| classified_json | 63 |
| classified_keyword | 5 |
| flagged_invalid_output | 4 |
| skipped_empty | 4 |
| flagged_long | 4 |
| budget_exhausted | 84 |
| **Total** | **164** |

The system checks the accumulated token count before each model call. Once the budget is reached, remaining eligible tickets are marked `budget_exhausted`.

### 8. Customer Routing

The final few-shot results were combined with customer segment information from Part B.

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |
| **Total** | **164** |

Routing rules:

- Fraud tickets → Security team, high priority.
- Premium customers → Premium desk, high priority.
- Invalid, empty, and long tickets → Human review.
- Other successfully classified tickets → Support queue.

### 9. Output Files

Part C generates:

- `outputs/triage_results.csv` – final classification and routing results.
- `outputs/triage_budget_demo.csv` – token budget demonstration.
- `outputs/triage_retry_log.csv` – retry attempt records.
- `outputs/triage_comparison.json` – classification metrics, token usage, and redaction counts.

### 10. Reproduction Command

Run the following command from the project root:

```bash
python -c "from src.triage import run_pipeline, FEW; results, summary = run_pipeline(prompt=FEW, token_budget=100000)"
```

### 11. Validation

The validation script confirmed:

- 164 tickets processed.
- 147 successfully classified.
- 19 transient errors recovered in each full classification run.
- 278 PII instances redacted in each full run.
- Zero-shot and few-shot accuracy matched the expected results.
- Final CSV contains no original ticket text.
- Token budget exhaustion was demonstrated.
- Fraud and human-review routing rules passed.

**Result: All Part C validation checks passed.**
## Part D – Policy RAG and Refund Support Agent

### 1. Policy document preparation and retrieval

The system uses five policy documents stored in `data/policies/`:

- `refund_policy.txt`
- `delivery_policy.txt`
- `escalation_policy.txt`
- `payment_policy.txt`
- `privacy_policy.txt`

The documents are divided into text chunks using:
- Chunk size: 40 words
- Overlap: 10 words
- Total chunks: 15

A TF-IDF vectorizer converts policy chunks and user queries into numerical representations. Cosine similarity ranks the most relevant chunks.

For the query **"refund policy late delivery"**, the top three retrieved chunks came from `refund_policy.txt`.

TF-IDF gives importance to words that are informative within a document collection. Cosine similarity measures the similarity between the query vector and each policy chunk.

**Limitations:** Keyword-based retrieval may miss meaning expressed using different words. It also depends on the quality and completeness of the policy documents.

### 2. Policy retrieval evaluation

The evaluation dataset contains 14 questions:
- 12 answerable questions
- 2 questions with no answer in the policy documents

| Evaluation measure | With 10-word overlap | Without overlap |
|---|---:|---:|
| Correct answerable document retrieval | 12/12 | 12/12 |
| Expected facts found | 12/12 | 10/12 |
| Correct fallback responses | 2/2 | 2/2 |

Without overlap, two answerable questions lost their expected facts:
- Completed-order refund eligibility after 45 minutes.
- Payment failure refund timing within 48 hours.

This demonstrates how chunk overlap can preserve important information that crosses chunk boundaries.

The answer function sends only retrieved policy context and the question to the mock language model. If the information is unavailable, it returns:

`I could not find that information in the policy documents.`

### 3. Refund support agent and tools

The refund support agent uses a ReAct-style loop: it receives a request, selects a tool, observes the result, and decides the next action.

Five tools are implemented:

1. `get_order` – retrieves order information from SQLite.
2. `search_policy` – retrieves relevant policy chunks.
3. `calculate_refund` – calculates refund eligibility and amount.
4. `issue_refund` – records an approved simulated refund.
5. `escalate_to_human` – records cases requiring human review.

**Refund business rules:**

- Only completed orders are eligible.
- Delivery taking more than 45 minutes qualifies for a full refund.
- Delivery taking 45 minutes or less qualifies for no refund.
- Refunds above INR 1,000 are escalated to a human lead.
- Refunds of INR 1,000 or less are issued through the simulated refund tool.
- All refund operations are simulated; no external payment is made.

### 4. Agent scenario results

Seven scenarios were executed.

| Scenario | Expected outcome | Actual outcome | Model calls |
|---|---|---|---:|
| S1 – Late delivery, INR 128.25 | REFUND | REFUND | 5 |
| S2 – Late delivery, INR 135 | REFUND | REFUND | 5 |
| S3 – Refund above INR 1,000 | ESCALATE | ESCALATE | 5 |
| S4 – Cancelled order | DENIED | DENIED | 3 |
| S5 – Delivery within 45 minutes | DENIED | DENIED | 4 |
| S6 – Invalid order ID | NOT_FOUND | NOT_FOUND | 2 |
| S7 – Repeated tool loop | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | 6 |

Across the seven scenarios, the agent made 30 model calls and used 6,523 tokens.

S7 deliberately repeated policy searches. The agent stopped after six iterations and handed the request to a human rather than continuing indefinitely.

### 5. Token and time limits

Two safety-limit tests were performed.

| Test | Configuration | Result |
|---|---|---|
| Token limit | 200 tokens | FALLBACK_TOKEN_LIMIT |
| Time limit | 0 seconds | FALLBACK_TIMEOUT |

The token-limit test stopped after the model calls exceeded the configured limit. Two read-only tools had executed, but no refund was issued. The time-limit test stopped before the first model call.

### 6. Checkpointing and crash recovery

The agent saves its execution state after every tool observation in the `state/` directory.

A crash was simulated after the second tool execution. The agent was then resumed using `resume_agent(run_id)`.

The resumed tool sequence was:

`get_order → search_policy → calculate_refund → issue_refund`

The final result was:

`REFUND: INR 128.25 issued.`

The agent did not repeat the first two completed tools. The test refund was written to a separate test ledger, leaving the main scenario ledger unchanged.

### 7. ReAct versus Plan-and-Execute

ReAct selects actions step by step using observations returned by tools. In this project, the seven ReAct scenarios required 30 model calls and 6,523 tokens in total. It handled unexpected observations, including cancelled orders, missing orders, and repeated tool calls. Its main advantage is adaptability, although repeated reasoning can increase token usage.

Plan-and-Execute first creates a sequence of actions and then executes the plan. It can reduce repeated planning for predictable workflows, but a plan may need revision when an order is missing or a policy result changes the next action. A separate Plan-and-Execute benchmark was not implemented, so comparative call and token measurements are not available.

**Recommendation:** ReAct is suitable for this refund-support prototype because refund eligibility depends on information discovered during execution. Iteration, token, and time limits provide safeguards against excessive tool calls.

### 8. Part D output files

| File or folder | Purpose |
|---|---|
| `outputs/agent_runs.jsonl` | Results of seven agent scenarios |
| `outputs/refund_ledger.jsonl` | Simulated refund and escalation records |
| `outputs/resume_test/refund_ledger.jsonl` | Separate checkpoint-test ledger |
| `state/` | Agent execution checkpoints |
| `data/policies/` | Policy source documents |
| `src/policy_rag.py` | RAG retrieval and answer functions |
| `src/refund_agent.py` | Refund tools, agent loop, and resume function |

### 9. Reproduction commands

Run the following commands from the project root after installing the requirements and generating the project data:

```powershell
python -c "from src.policy_rag import PolicyRAG,evaluate; r=PolicyRAG(); print('Chunks:',len(r.chunks)); print(evaluate(r).to_string(index=False))"
python run_agent_scenarios.py
python test_resume.py
```

The scenario and resume commands above have already been executed successfully. Do not rerun them against the existing output files unless you intentionally prepare a clean test environment, because they can create additional simulated ledger entries.

No external API key or live payment integration is required.