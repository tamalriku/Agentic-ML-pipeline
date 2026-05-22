import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from models.schemas import EDAReport, PreprocessingConfig


def build_preprocessing_pipeline(df: pd.DataFrame, eda: EDAReport):
    """
    Builds a ColumnTransformer preprocessing pipeline based on EDA results.
    Returns (preprocessor, X, y, config, label_encoder).
    """
    target = eda.target_column
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].copy() if target else df.copy()
    y = None
    le = None

    if target and target in df.columns:
        y = df[target].copy()
        if eda.task_type == "classification" and y.dtype == object:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target)

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # Decide scaler
    scaler_name = "StandardScaler"
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Decide encoder
    encoder_name = "OrdinalEncoder"
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if cat_cols:
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    steps_summary = []
    if numeric_cols:
        steps_summary.append(f"Numeric ({len(numeric_cols)} cols): median imputation → StandardScaler")
    if cat_cols:
        steps_summary.append(f"Categorical ({len(cat_cols)} cols): mode imputation → OrdinalEncoder")
    if eda.class_balance == "imbalanced":
        steps_summary.append("Note: class imbalance detected — consider SMOTE post-split")

    config = PreprocessingConfig(
        scaler=scaler_name,
        numeric_imputer="median",
        categorical_imputer="most_frequent",
        encoder=encoder_name,
        steps_summary=steps_summary,
    )

    return preprocessor, X, y, config, le
