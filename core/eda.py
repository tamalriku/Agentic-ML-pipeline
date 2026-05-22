import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from models.schemas import EDAReport


def detect_task_type(df: pd.DataFrame) -> tuple[str, str | None]:
    """
    Heuristic task-type detection.
    Returns (task_type, target_column).
    """
    target_candidates = [
        "target", "label", "class", "y", "output", "result",
        "survived", "price", "salary", "churn", "fraud",
        "diagnosis", "disease", "outcome", "category", "species",
        "medv", "charges", "fare", "mpg", "housing_median_age",
        "median_house_value",
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
    col_lower = [c.lower() for c in df.columns]
    named_target = any(c in target_candidates for c in col_lower)
    if not named_target and len(numeric_cols) == len(df.columns):
        task_type = "clustering"
        target_col = None

    return task_type, target_col


def compute_linearity_score(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Approximates how 'linear' the relationship between features and target is.
    Returns a score in [0, 1]. Higher = more linear.
    Uses average absolute Pearson correlation of numeric features with target.
    """
    numeric_X = X.select_dtypes(include=[np.number]).copy()
    if numeric_X.empty or y is None:
        return 0.0
    correlations = []
    for col in numeric_X.columns:
        col_data = numeric_X[col].fillna(numeric_X[col].median())
        if col_data.std() > 0:
            r, _ = scipy_stats.pearsonr(col_data, y)
            correlations.append(abs(r))
    return float(np.mean(correlations)) if correlations else 0.0


def compute_sparsity(X: pd.DataFrame) -> float:
    """Fraction of zero-valued cells in numeric columns."""
    numeric_X = X.select_dtypes(include=[np.number])
    if numeric_X.empty:
        return 0.0
    total = numeric_X.size
    zeros = (numeric_X == 0).sum().sum()
    return float(zeros / total)


def compute_mean_skewness(X: pd.DataFrame) -> float:
    """Average absolute skewness across numeric features."""
    numeric_X = X.select_dtypes(include=[np.number])
    if numeric_X.empty:
        return 0.0
    skews = numeric_X.apply(lambda c: abs(c.dropna().skew()) if c.dropna().std() > 0 else 0)
    return float(skews.mean())


def compute_outlier_ratio(X: pd.DataFrame) -> float:
    """Fraction of values beyond 3 std deviations (z-score method)."""
    numeric_X = X.select_dtypes(include=[np.number]).dropna()
    if numeric_X.empty:
        return 0.0
    z = np.abs(scipy_stats.zscore(numeric_X, nan_policy="omit"))
    return float((z > 3).mean())


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

    # --- Rich stats for smart model selection ---
    feature_cols = [c for c in df.columns if c != target_col]
    X_df = df[feature_cols] if feature_cols else df

    # High-cardinality categoricals (e.g. free-text, IDs)
    high_cardinality_cols = [
        c for c in categorical_features
        if c in X_df.columns and df[c].nunique() > 20
    ]

    # Sparsity (useful for detecting bag-of-words style data → LinearSVC/NB)
    sparsity = compute_sparsity(X_df)

    # Mean absolute skewness across numeric features
    mean_skewness = compute_mean_skewness(X_df)

    # Outlier ratio
    outlier_ratio = compute_outlier_ratio(X_df)

    # Linearity signal vs target (only for supervised tasks)
    linearity_score = 0.0
    if task_type in ("classification", "regression") and target_col and target_col in df.columns:
        y_series = df[target_col]
        if task_type == "regression":
            linearity_score = compute_linearity_score(X_df, y_series)
        else:
            # For classification encode target numerically for correlation
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y_series.astype(str).fillna("NA")))
            linearity_score = compute_linearity_score(X_df, y_enc)

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
        # enriched stats
        sparsity=round(sparsity, 4),
        mean_skewness=round(mean_skewness, 4),
        outlier_ratio=round(outlier_ratio, 4),
        linearity_score=round(linearity_score, 4),
        high_cardinality_cols=high_cardinality_cols,
    )
