# 🤖 Agentic ML Pipeline

An end-to-end autonomous machine learning pipeline powered by **scikit-learn** + **Anthropic Claude**.

Upload any CSV → the agent performs EDA, selects the best model, tunes hyperparameters, evaluates performance, and delivers an LLM-written commentary — all without manual configuration.

## ✨ Features

- **Auto task detection** — classifies your problem as classification, regression, or clustering
- **EDA engine** — missing values, class balance, feature types, correlations
- **Smart preprocessing** — scaling, encoding, imputation chosen per dataset
- **Rule-based model selection** + **LLM commentary** via Claude API
- **Hyperparameter tuning** — GridSearchCV with cross-validation
- **Evaluation report** — metrics, feature importance, confusion matrix
- **FastAPI backend** + clean HTML/JS frontend
- **Streaming LLM commentary** via SSE

## 🗂️ Project Structure

```
agentic-ml-pipeline/
├── app/
│   └── main.py              # FastAPI app, routes, SSE streaming
├── core/
│   ├── eda.py               # EDA engine
│   ├── preprocessor.py      # Auto preprocessing pipeline
│   ├── selector.py          # Rule-based model selector
│   ├── trainer.py           # Training + GridSearchCV
│   └── evaluator.py         # Metrics + feature importance
├── models/
│   └── schemas.py           # Pydantic models
├── frontend/
│   └── index.html           # Single-page app
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/agentic-ml-pipeline
cd agentic-ml-pipeline
pip install -r requirements.txt

# 2. Set your Anthropic API key
cp .env.example .env
# Edit .env and add your key: ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
uvicorn app.main:app --reload

# 4. Open http://localhost:8000
```

## 🧠 How It Works

```
CSV Upload
    │
    ▼
EDA Engine          ← shape, dtypes, missing%, class balance, correlation
    │
    ▼
Preprocessor        ← StandardScaler / OrdinalEncoder / SimpleImputer (auto)
    │
    ▼
Rule-Based Selector ← picks model family from dataset heuristics
    │
    ▼
LLM Commentary      ← Claude explains why + what to watch out for (streamed)
    │
    ▼
GridSearchCV Tuner  ← cross-validated hyperparameter search
    │
    ▼
Evaluator           ← accuracy/F1/AUC or MAE/RMSE/R² + feature importance
    │
    ▼
Final Report        ← JSON + LLM summary streamed to UI
```

## 📊 Supported Task Types

| Task | Models Considered |
|------|-------------------|
| Binary Classification | LogisticRegression, RandomForest, GradientBoosting, SVC |
| Multiclass Classification | RandomForest, GradientBoosting, KNN |
| Regression | Ridge, RandomForestRegressor, GradientBoostingRegressor |
| Clustering | KMeans, DBSCAN |

## 🛠️ Tech Stack

- **Backend**: FastAPI, scikit-learn, pandas, numpy
- **LLM**: Anthropic Claude (claude-sonnet-4-20250514)
- **Frontend**: Vanilla HTML/CSS/JS (zero dependencies)
- **Streaming**: Server-Sent Events (SSE)

## 📄 License

MIT
