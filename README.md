# CreditRisk AI — Agentic Credit Risk Intelligence Platform

An **Agentic AI** platform for credit risk assessment, combining a traditional XGBoost ML model with a **LangGraph-orchestrated agent**, **RAG-based credit policy retrieval**, and a **Groq/Llama3 LLM auditor** — all wrapped in a dark fintech-style Streamlit dashboard.

> **Capstone Project · Semester 4 · AI/ML Programme**

---

## Overview

This project evolves a baseline ML credit risk classifier into a full **Agentic AI** system:

1. **XGBoost ML Engine** — classifies loan applicants as Good/Bad risk with default probability.
2. **RAG Policy Retriever** — retrieves relevant clauses from the Universal Bank Credit Policy using FAISS + Sentence Transformers.
3. **LangGraph Agentic Workflow** — orchestrates the ML and RAG steps, then routes to the LLM auditor.
4. **Groq/Llama3 LLM Auditor** — reasons over the ML prediction and policy context to generate a grounded, explainable final recommendation.

**Dataset:** German Credit Data (1,000 applicants, UCI Repository)  
**ML Model:** XGBoost + GridSearchCV (recall-optimised)  
**Agentic Stack:** LangGraph · LangChain · Groq (Llama3-8B) · FAISS · Sentence Transformers  
**UI:** Streamlit · Plotly

---

## Agentic Workflow Architecture

```
User Input (Form)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│                   LangGraph Agent                        │
│                                                          │
│  Node 1: predict_risk  →  XGBoost ML Model              │
│       ↓                                                  │
│  Node 2: retrieve_policy  →  FAISS + Sentence-BERT RAG  │
│       ↓                                                  │
│  Node 3: generate_audit  →  Groq (Llama3-8B) LLM       │
└──────────────────────────────────────────────────────────┘
       │
       ▼
Dashboard Output:
  • Risk Score Gauge
  • KPI Cards
  • Policy-Grounded Audit Report (Agentic)
  • RAG Policy Snippets
```

---

## Project Structure

```
genai-capstone-sem-4/
├── data/
│   ├── german_credit_data.csv        # Raw dataset (UCI German Credit)
│   └── credit_policy.md              # 🆕 Bank credit policy (RAG knowledge base)
├── models/
│   └── credit_risk_model_v2.pkl      # Trained XGBoost model artifact
├── notebooks/
│   └── train_process.ipynb           # Exploratory analysis notebook
├── report/
│   ├── genai_credit_risk-3.pdf       # Project report (PDF)
│   └── main.tex                      # LaTeX source
├── screenshots/
│   └── dashboard.png                 # UI preview
├── src/
│   ├── data.py                       # load_data() — CSV loading & cleaning
│   ├── preprocess.py                 # build_preprocessor(), split_data()
│   ├── train.py                      # train_model(), evaluate_model()
│   ├── predict.py                    # make_prediction() — inference helper
│   ├── rag.py                        # 🆕 PolicyRetriever — FAISS-based RAG
│   └── agent.py                      # 🆕 LangGraph agentic workflow
├── .streamlit/
│   └── config.toml                   # Dark theme configuration
├── .env.example                      # 🆕 Environment variable template
├── app.py                            # Streamlit dashboard (Agentic UI)
├── run_training.py                   # Training pipeline orchestrator
├── pyproject.toml                    # Project metadata & dependencies
├── requirements.txt                  # Pip-compatible dependency list
├── uv.lock                           # uv lockfile
├── .python-version                   # Pinned Python version (3.12)
└── .gitignore
```

---

## Agentic AI Features

| Component | Description |
|---|---|
| **LangGraph Workflow** | 3-node state machine: ML → RAG → LLM Audit |
| **RAG Retrieval** | FAISS vector search over credit policy rules |
| **LLM Auditor** | Groq (Llama3-8B) provides policy-grounded reasoning |
| **Reasoning Trace** | Dashboard shows each agent step (ML\_INFERRED → POLICY\_RETRIEVED → AUDITED) |
| **Policy Override Detection** | Agent flags when policy rules contradict the ML model |
| **UK Bank Credit Policy KB** | 5-category policy document used as the RAG knowledge base |

---

## Dashboard Features

| Panel | Description |
|---|---|
| **Navbar** | Live system status badge, version indicator |
| **Client Assessment** | Tabbed form — Demographics, Credit Profile, Financials |
| **KPI Cards** | Credit Amount · Duration · Job Level · Savings |
| **Risk Score** | Donut gauge showing default probability (green / amber / red) |
| **Client Profile** | At-a-glance stats grid |
| **Risk Analysis** | 3 key indicator checks with pass/fail signals |
| **Probability Scale** | Gradient bar with labelled percentage |
| **GenAI Auditor Report** | 🆕 LLM-generated, policy-grounded explanation with reasoning trace |
| **RAG Policy Snippets** | 🆕 Relevant credit policy rules retrieved for this applicant |

---

## Local Setup

### Prerequisites

- Python 3.12 or higher
- A free **Groq API Key** — get one at [console.groq.com/keys](https://console.groq.com/keys)

---

### Option A — Using uv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/HusainNST/genai-capstone-sem-4.git
cd genai-capstone-sem-4

# 2. Install all dependencies (ML + Agentic AI)
uv sync

# 3. Set up your API key
cp .env.example .env
# Open .env and fill in your GROQ_API_KEY

# 4. (Optional) Retrain the model from scratch
uv run run_training.py

# 5. Launch the dashboard
uv run streamlit run app.py
```

---

### Option B — Using pip

```bash
# 1. Clone the repository
git clone https://github.com/HusainNST/genai-capstone-sem-4.git
cd genai-capstone-sem-4

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and fill in your GROQ_API_KEY

# 5. (Optional) Retrain the model
python3 run_training.py

# 6. Launch the dashboard
streamlit run app.py
```

---

## API Key Setup

The Agentic AI auditor requires a **Groq API Key** (free tier available).

### Method 1: `.env` File (Persistent)

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_real_key_goes_here
```
The app will automatically load this via `python-dotenv`.

### Method 2: Dashboard Sidebar (Session Only)

No `.env` file needed. Open the app and enter your key in the **⚙️ Settings** sidebar. The key is used in-memory for that session only.

---

## Model Performance

Trained with 5-fold cross-validated GridSearchCV optimising for **Recall** (minimising missed bad loans).

| Metric | Score |
|---|---|
| Accuracy | ~76% |
| ROC-AUC | ~80% |
| Recall (Bad Loans) | ~68% |
| Optimiser | GridSearchCV (recall scoring) |

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Applicant age (18–80) |
| Sex | Categorical | male / female |
| Job | Ordinal | Skill level 0–3 |
| Housing | Categorical | own / free / rent |
| Saving accounts | Categorical | unknown / little / moderate / quite rich / rich |
| Checking account | Categorical | unknown / little / moderate / rich |
| Credit amount | Numeric | Loan amount in Deutsche Marks |
| Duration | Numeric | Loan term in months |
| Purpose | Categorical | car / education / business / etc. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost, scikit-learn, imbalanced-learn |
| Agentic Orchestration | LangGraph |
| LLM | Groq API (Llama3-8B-8192) |
| RAG | FAISS, Sentence Transformers (all-MiniLM-L6-v2), LangChain |
| Dashboard | Streamlit, Plotly |
| Dependency Management | uv |
