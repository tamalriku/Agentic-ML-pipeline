<div align="center">

# Agentic ML Pipeline

**Upload a CSV. Get a trained model, insights, and an AI explanation — automatically.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Claude AI](https://img.shields.io/badge/Powered%20by-Claude%20AI-7B2D8B?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **No code. No config. No ML expertise required.**
> Drop in any CSV file and the agent does the rest — from raw data to trained model with a full LLM-written report.

<br/>

[Quick Start](#-quick-start) &nbsp;·&nbsp; [Features](#-features) &nbsp;·&nbsp; [How It Works](#-how-it-works) &nbsp;·&nbsp; [Supported Tasks](#-supported-task-types) &nbsp;·&nbsp; [Tech Stack](#-tech-stack) &nbsp;·&nbsp; [Contributing](#-contributing)

</div>

---

## Features

| Feature | Description |
|---|---|
| **Auto Task Detection** | Automatically classifies your problem as classification, regression, or clustering |
| **EDA Engine** | Analyzes missing values, class balance, feature types, and correlations |
| **Smart Preprocessing** | Selects scaling, encoding, and imputation strategies per dataset |
| **LLM Commentary** | Claude AI explains model choice and what to watch out for — streamed live |
| **Hyperparameter Tuning** | GridSearchCV with cross-validation for optimal performance |
| **Evaluation Report** | Metrics, feature importance, and confusion matrix in one clean view |
| **Real-time Streaming** | Live AI commentary via Server-Sent Events (SSE) |
| **Zero-dependency Frontend** | Clean HTML/JS UI — no npm, no build step |

---

## Quick Start

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (Claude)

### 1. Clone the repo

```bash
git clone https://github.com/tamalriku/agentic-ml-pipeline.git
cd agentic-ml-pipeline
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

### 4. Run the app

```bash
uvicorn app.main:app --reload
```

### 5. Open in your browser

```
http://localhost:8000
```

> That's it! Upload any CSV, pick a target column, and watch the agent work.

---

## How It Works

The pipeline runs fully autonomously the moment you upload a file:

```
CSV Upload
      |
      v
EDA Engine
   |-- Shape, dtypes, missing %
   |-- Class balance check
   +-- Correlation analysis
      |
      v
Auto Preprocessor
   |-- StandardScaler (numerical)
   |-- OrdinalEncoder (categorical)
   +-- SimpleImputer (missing values)
      |
      v
Rule-Based Model Selector
   +-- Picks best model family from dataset heuristics
      |
      v
Claude AI Commentary          <-- streamed live to UI via SSE
   +-- Explains why + caveats
      |
      v
GridSearchCV Tuner
   +-- Cross-validated hyperparameter search
      |
      v
Evaluator
   |-- Classification: Accuracy, F1, AUC, Confusion Matrix
   +-- Regression: MAE, RMSE, R2 + Feature Importance
      |
      v
Final Report + LLM Summary   <-- streamed to UI
```

---

## Supported Task Types

| Task | Models Evaluated |
|------|-----------------|
| **Binary Classification** | Logistic Regression, Random Forest, Gradient Boosting, SVC |
| **Multiclass Classification** | Random Forest, Gradient Boosting, KNN |
| **Regression** | Ridge, Random Forest Regressor, Gradient Boosting Regressor |
| **Clustering** | KMeans, DBSCAN |

> The agent picks the best-fit model automatically based on dataset size, class count, and feature types.

---

## Project Structure

```
agentic-ml-pipeline/
|
|-- app/
|   +-- main.py              # FastAPI app, routes, SSE streaming
|
|-- core/
|   |-- eda.py               # Exploratory Data Analysis engine
|   |-- preprocessor.py      # Auto preprocessing pipeline
|   |-- selector.py          # Rule-based model selector
|   |-- trainer.py           # Training + GridSearchCV
|   +-- evaluator.py         # Metrics + feature importance
|
|-- models/
|   +-- schemas.py           # Pydantic request/response models
|
|-- frontend/
|   +-- index.html           # Single-page app (no build needed)
|
|-- .env.example             # API key template
|-- requirements.txt
+-- README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **ML Engine** | [scikit-learn](https://scikit-learn.org/) + [pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) |
| **LLM** | [Anthropic Claude](https://www.anthropic.com/) (claude-sonnet) |
| **Streaming** | Server-Sent Events (SSE) |
| **Frontend** | Vanilla HTML / CSS / JavaScript |
| **Config** | [python-dotenv](https://github.com/theskumar/python-dotenv) + [Pydantic](https://docs.pydantic.dev/) |

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Claude API key from [console.anthropic.com](https://console.anthropic.com/) |

---

## FAQ

<details>
<summary><b>What CSV formats are supported?</b></summary>
<br/>
Any standard CSV file with a header row. The agent handles mixed types (numeric + categorical), missing values, and varying scales automatically.
</details>

<details>
<summary><b>Do I need an ML background to use this?</b></summary>
<br/>
Nope! Just upload your CSV, select the target column you want to predict, and the agent handles everything — model selection, tuning, and explaining results in plain English.
</details>

<details>
<summary><b>How is the model selected?</b></summary>
<br/>
A rule-based selector analyzes your dataset (number of samples, features, class balance, task type) and picks the most appropriate algorithm family. Claude then provides reasoning for the choice.
</details>

<details>
<summary><b>Can I run this without an Anthropic API key?</b></summary>
<br/>
The ML pipeline (EDA to training to evaluation) works without an API key. Only the LLM commentary step requires Claude. You can skip it by removing the streaming endpoint calls, though the experience will be less informative.
</details>

---

## Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/agentic-ml-pipeline.git
cd agentic-ml-pipeline
git checkout -b feature/your-feature-name
```

Make your changes, then:

```bash
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

Open a Pull Request!

---

## License

This project is licensed under the **MIT License**.

---

<div align="center">

Made with love by [tamalriku](https://github.com/tamalriku)

**Star this repo if you find it useful!**

</div>
