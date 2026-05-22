import pandas as pd
import numpy as np
from models.schemas import EDAReport


def detect_task_type(df: pd.DataFrame) -> tuple[str, str | None]:
    """
    Heuristic task-type detection.
    Returns (task_type, target_column).
    """
    # Common target column names (priority order)
    target_candidates = [
        "target", "label", "class", "y", "output", "result",
        "survived", "price", "salary", "churn", "fraud",
        "diagnosis", "disease", "outcome", "category", "species"
    ]

    target_col = None
    for cand in target_candidates:
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            target_col = matches[0]
            break

    # Fall back: last column
    if target_col is None:
        target_col = df.columns[-1]

    target = df[target_col].dropna()
    n_unique = target.nunique()
    dtype = target.dtype

    if pd.api.types.is_numeric_dtype(dtype):
        if n_unique <= 20:
            task_type = "classification"
        else:
            task_type = "regression"
    else:
        task_type = "classification"

    # Detect clustering: no obvious target in common names + all numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in target_candidates and len(numeric_cols) == len(df.columns):
        task_type = "clustering"
        target_col = None

    return task_type, target_col


def run_eda(df: pd.DataFrame) -> EDAReport:
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_pct = {
        col: round(df[col].isna().mean() * 100, 2)
        for col in df.columns
        if df[col].isna().any()
    }
    total_missing_pct = round(df.isna().mean().mean() * 100, 2)

    task_type, target_col = detect_task_type(df)

    n_classes = None
    class_balance = "N/A"
    if task_type == "classification" and target_col:
        n_classes = int(df[target_col].nunique())
        counts = df[target_col].value_counts(normalize=True)
        min_freq = counts.min()
        class_balance = "balanced" if min_freq >= 0.3 else "imbalanced"

    sample_preview = df.head(5).fillna("").astype(str).to_dict(orient="records")

    return EDAReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=df.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        missing_pct=missing_pct,
        total_missing_pct=total_missing_pct,
        task_type=task_type,
        target_column=target_col,
        n_classes=n_classes,
        class_balance=class_balance,
        sample_preview=sample_preview,
    )
