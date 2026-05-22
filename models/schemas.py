from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class EDAReport(BaseModel):
    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    missing_pct: Dict[str, float]
    total_missing_pct: float
    task_type: str  # classification | regression | clustering
    target_column: Optional[str]
    n_classes: Optional[int]
    class_balance: Optional[str]  # balanced | imbalanced | N/A
    sample_preview: List[Dict[str, Any]]
    # Enriched dataset statistics for intelligent model selection
    sparsity: float = 0.0           # fraction of zero cells in numeric features
    mean_skewness: float = 0.0      # average |skewness| across numeric features
    outlier_ratio: float = 0.0      # fraction of values beyond 3-sigma
    linearity_score: float = 0.0    # avg |Pearson r| of features with target (0=nonlinear, 1=linear)
    high_cardinality_cols: List[str] = []  # categorical cols with >20 unique values


class PreprocessingConfig(BaseModel):
    scaler: str
    numeric_imputer: str
    categorical_imputer: str
    encoder: str
    steps_summary: List[str]


class ModelSelection(BaseModel):
    selected_model: str
    alternative_model: str
    rule_reason: str
    hyperparameter_grid: Dict[str, Any]


class EvaluationReport(BaseModel):
    model_name: str
    task_type: str
    best_params: Dict[str, Any]
    metrics: Dict[str, float]
    feature_importance: Optional[List[Dict[str, Any]]]
    cv_score: float
    cv_std: float


class PipelineResult(BaseModel):
    eda: EDAReport
    preprocessing: PreprocessingConfig
    selection: ModelSelection
    evaluation: EvaluationReport
