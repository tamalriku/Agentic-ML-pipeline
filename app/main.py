import os
import io
import json
import asyncio
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from core.eda import run_eda
from core.preprocessor import build_preprocessing_pipeline
from core.selector import select_model
from core.trainer import train_model
from core.evaluator import evaluate_model
from models.schemas import PipelineResult

load_dotenv()

app = FastAPI(title="Agentic ML Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=FRONTEND_PATH.read_text(encoding="utf-8"))


@app.post("/api/run-pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    """
    Main pipeline endpoint. Accepts a CSV, runs full pipeline,
    returns PipelineResult as JSON.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns and 1 row.")

    # 1. EDA
    eda = run_eda(df)

    # 2. Preprocessing
    preprocessor, X, y, prep_config, label_encoder = build_preprocessing_pipeline(df, eda)

    # 3. Model selection
    primary_model, alt_model, selection = select_model(eda)

    # 4. Train
    pipeline, X_test, y_test, best_params, cv_score, cv_std = train_model(
        preprocessor=preprocessor,
        model=primary_model,
        X=X,
        y=y,
        task_type=eda.task_type,
        hp_grid=selection.hyperparameter_grid,
        model_name=selection.selected_model,
    )

    # 5. Evaluate
    evaluation = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        eda=eda,
        selection=selection,
        best_params=best_params,
        cv_score=cv_score,
        cv_std=cv_std,
        X_full=X,
    )

    result = PipelineResult(
        eda=eda,
        preprocessing=prep_config,
        selection=selection,
        evaluation=evaluation,
    )

    return result.model_dump()


@app.get("/api/commentary")
async def stream_commentary(
    task_type: str,
    model_name: str,
    rule_reason: str,
    metrics: str,
    n_rows: int,
    n_cols: int,
    class_balance: str = "N/A",
    preprocessing_steps: str = "",
):
    """
    SSE endpoint. Streams LLM commentary about the pipeline results.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set.")

    # Use OpenAI SDK pointed at OpenRouter's OpenAI-compatible endpoint
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = f"""You are an expert ML engineer giving a concise, insightful commentary on an automated ML pipeline run.

Dataset: {n_rows} rows × {n_cols} columns
Task type: {task_type}
Class balance: {class_balance}
Preprocessing: {preprocessing_steps}
Selected model: {model_name}
Rule-based reason: {rule_reason}
Evaluation metrics: {metrics}

Write a 4-6 sentence commentary covering:
1. Why this model is appropriate for this data
2. What the metrics tell us (is it good, concerning, overfitting risk?)
3. One specific actionable recommendation to improve performance further
4. Any caveats or things to watch out for

Be direct, technical, and specific. Avoid generic statements. Use concrete observations from the metrics and dataset properties."""

    async def event_stream():
        try:
            stream = await client.chat.completions.create(
                model="anthropic/claude-3-5-sonnet",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    payload = json.dumps({"token": chunk.choices[0].delta.content})
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

