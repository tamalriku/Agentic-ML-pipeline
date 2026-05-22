from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from models.schemas import EDAReport, ModelSelection


def select_model(eda: EDAReport) -> tuple[object, object, ModelSelection]:
    """
    Rule-based model selection. Returns (primary_model, alternative_model, config).
    """
    task = eda.task_type
    n_rows = eda.n_rows
    n_cols = eda.n_cols
    n_classes = eda.n_classes or 2
    has_categoricals = len(eda.categorical_features) > 0
    imbalanced = eda.class_balance == "imbalanced"

    if task == "classification":
        if n_rows < 1000 and not has_categoricals:
            primary = SVC(probability=True, random_state=42)
            primary_name = "SVC"
            alt = LogisticRegression(max_iter=1000, random_state=42)
            alt_name = "LogisticRegression"
            reason = "Small dataset with numeric features — SVC generalizes well in low-data regimes."
            hp_grid = {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["rbf", "linear"],
            }
        elif n_rows >= 1000 and n_classes == 2:
            primary = GradientBoostingClassifier(random_state=42)
            primary_name = "GradientBoostingClassifier"
            alt = RandomForestClassifier(random_state=42)
            alt_name = "RandomForestClassifier"
            reason = "Binary classification with sufficient data — GBM captures complex patterns with strong performance."
            hp_grid = {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
            }
        else:
            primary = RandomForestClassifier(random_state=42)
            primary_name = "RandomForestClassifier"
            alt = GradientBoostingClassifier(random_state=42)
            alt_name = "GradientBoostingClassifier"
            reason = "Multiclass task — Random Forest handles multiple classes robustly out of the box."
            hp_grid = {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            }

    elif task == "regression":
        if n_cols <= 10 and not has_categoricals:
            primary = Ridge(random_state=42) if hasattr(Ridge, 'random_state') else Ridge()
            primary_name = "Ridge"
            alt = RandomForestRegressor(random_state=42)
            alt_name = "RandomForestRegressor"
            reason = "Low-dimensional numeric regression — Ridge provides stable, interpretable coefficients."
            hp_grid = {
                "regressor__alpha": [0.01, 0.1, 1.0, 10.0],
            }
        else:
            primary = GradientBoostingRegressor(random_state=42)
            primary_name = "GradientBoostingRegressor"
            alt = RandomForestRegressor(random_state=42)
            alt_name = "RandomForestRegressor"
            reason = "Complex regression with many features — GBM handles nonlinear relationships and mixed feature types."
            hp_grid = {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [3, 5],
                "regressor__learning_rate": [0.05, 0.1],
            }

    else:  # clustering
        primary = KMeans(random_state=42, n_init=10)
        primary_name = "KMeans"
        alt = DBSCAN()
        alt_name = "DBSCAN"
        reason = "Unsupervised task — KMeans is interpretable and efficient; DBSCAN handles arbitrary shapes."
        hp_grid = {
            "clusterer__n_clusters": [2, 3, 4, 5, 6],
        }

    config = ModelSelection(
        selected_model=primary_name,
        alternative_model=alt_name,
        rule_reason=reason,
        hyperparameter_grid=hp_grid,
    )

    return primary, alt, config
