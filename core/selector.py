"""
Smart model selector.

Decision logic uses the enriched EDAReport statistics to pick the
most appropriate algorithm for each dataset. The selector considers:

  Classification signals
  ─────────────────────
  • n_rows < 500            → logistic regression or SVC (low-data, needs regularisation)
  • linearity_score > 0.5   → logistic regression (features correlate linearly with target)
  • imbalanced + large      → gradient boosting (handles imbalance better than RF)
  • high sparsity (>0.5)    → LinearSVC (sparse data, text-like)
  • many categoricals        → gradient boosting (tree handles mixed types natively)
  • n_classes > 10          → random forest (stable for many classes)
  • default medium datasets → gradient boosting (binary) / random forest (multi)

  Regression signals
  ──────────────────
  • n_cols <= 5 and linearity > 0.5  → ridge regression
  • linearity > 0.4 and low skew     → elastic net
  • high outlier_ratio               → huber regressor (robust to outliers)
  • high mean_skewness               → gradient boosting (robust to skewed distributions)
  • default                          → gradient boosting

  Clustering signals
  ──────────────────
  • sparsity > 0.3   → MiniBatchKMeans (fast for sparse)
  • outlier > 0.05   → DBSCAN (noise-robust)
  • default          → KMeans
"""

from sklearn.linear_model import (
    LogisticRegression, Ridge, ElasticNet, HuberRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.calibration import CalibratedClassifierCV
from models.schemas import EDAReport, ModelSelection


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _classification_choice(eda: EDAReport):
    n       = eda.n_rows
    p       = eda.n_cols
    k       = eda.n_classes or 2
    imbal   = eda.class_balance == "imbalanced"
    has_cat = len(eda.categorical_features) > 0
    sparse  = eda.sparsity > 0.40
    linear  = eda.linearity_score
    skew    = eda.mean_skewness
    out_r   = eda.outlier_ratio

    # ── Sparse / text-like data ────────────────────────────────
    if sparse:
        inner = LinearSVC(max_iter=2000, random_state=42)
        primary = CalibratedClassifierCV(inner)
        primary_name = "LinearSVC (calibrated)"
        alt = LogisticRegression(max_iter=1000, solver="saga", random_state=42)
        alt_name = "LogisticRegression (saga)"
        reason = (
            f"Data is highly sparse ({eda.sparsity:.1%} zeros). "
            "LinearSVC is optimal for sparse high-dimensional spaces such as "
            "TF-IDF or one-hot encoded data."
        )
        hp_grid = {
            "classifier__base_estimator__C": [0.01, 0.1, 1.0],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Very small datasets (< 500 rows) ──────────────────────
    if n < 500:
        if linear > 0.35:
            primary = LogisticRegression(max_iter=1000, random_state=42)
            primary_name = "LogisticRegression"
            alt = SVC(probability=True, random_state=42)
            alt_name = "SVC"
            reason = (
                f"Small dataset ({n} rows) with moderate linearity signal "
                f"(avg |r|={linear:.2f}). Logistic Regression avoids overfitting "
                "with built-in L2 regularisation."
            )
            hp_grid = {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            }
        else:
            primary = SVC(probability=True, kernel="rbf", random_state=42)
            primary_name = "SVC (RBF)"
            alt = KNeighborsClassifier()
            alt_name = "KNeighborsClassifier"
            reason = (
                f"Small dataset ({n} rows) with nonlinear signal "
                f"(avg |r|={linear:.2f}). RBF-SVC maps to higher dimensions "
                "and generalises well with limited data."
            )
            hp_grid = {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", "auto"],
            }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Many classes (> 10) ───────────────────────────────────
    if k > 10:
        primary = RandomForestClassifier(n_estimators=200, random_state=42)
        primary_name = "RandomForestClassifier"
        alt = ExtraTreesClassifier(n_estimators=200, random_state=42)
        alt_name = "ExtraTreesClassifier"
        reason = (
            f"Many-class problem ({k} classes). Random Forest is stable for "
            "high-cardinality targets; each tree votes independently across "
            "all class boundaries without one-vs-rest decomposition."
        )
        hp_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 15, 30],
            "classifier__min_samples_leaf": [1, 2],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Strong linear signal + large data ─────────────────────
    if linear > 0.50 and not has_cat:
        primary = LogisticRegression(max_iter=1000, random_state=42)
        primary_name = "LogisticRegression"
        alt = GradientBoostingClassifier(random_state=42)
        alt_name = "GradientBoostingClassifier"
        reason = (
            f"Strong linear signal between features and target "
            f"(avg |Pearson r|={linear:.2f}). Logistic Regression is "
            "efficient, interpretable, and optimal when decision boundaries "
            "are approximately linear."
        )
        hp_grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Imbalanced binary / multiclass ───────────────────────
    if imbal:
        primary = GradientBoostingClassifier(random_state=42)
        primary_name = "GradientBoostingClassifier"
        alt = RandomForestClassifier(class_weight="balanced", random_state=42)
        alt_name = "RandomForestClassifier (balanced)"
        reason = (
            f"Class imbalance detected (min class freq < 30%). "
            "GradientBoosting iteratively focuses on misclassified minority "
            "samples; ensemble with class_weight='balanced' as fallback."
        )
        hp_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Large binary classification ───────────────────────────
    if k == 2 and n >= 1000:
        primary = GradientBoostingClassifier(random_state=42)
        primary_name = "GradientBoostingClassifier"
        alt = RandomForestClassifier(random_state=42)
        alt_name = "RandomForestClassifier"
        reason = (
            f"Binary classification with {n:,} rows. Gradient Boosting "
            "sequentially corrects residuals and typically outperforms "
            "Random Forest on medium-large tabular datasets."
        )
        hp_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Default: multiclass, mixed features, medium data ──────
    primary = RandomForestClassifier(random_state=42)
    primary_name = "RandomForestClassifier"
    alt = GradientBoostingClassifier(random_state=42)
    alt_name = "GradientBoostingClassifier"
    reason = (
        f"Multiclass classification ({k} classes, {n:,} rows, "
        f"{len(eda.categorical_features)} categorical features). "
        "Random Forest handles mixed feature types and multi-class "
        "natively without hyperparameter sensitivity."
    )
    hp_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }
    return primary, primary_name, alt, alt_name, reason, hp_grid


def _regression_choice(eda: EDAReport):
    n       = eda.n_rows
    p       = len(eda.numeric_features)
    linear  = eda.linearity_score
    skew    = eda.mean_skewness
    out_r   = eda.outlier_ratio
    has_cat = len(eda.categorical_features) > 0

    # ── High outlier ratio → robust regressor ─────────────────
    if out_r > 0.05:
        primary = HuberRegressor(max_iter=500)
        primary_name = "HuberRegressor"
        alt = GradientBoostingRegressor(random_state=42)
        alt_name = "GradientBoostingRegressor"
        reason = (
            f"High outlier ratio detected ({out_r:.1%} of values > 3σ). "
            "HuberRegressor is robust to outliers — it uses L2 loss for "
            "inliers and L1 for outliers, avoiding their undue influence."
        )
        hp_grid = {
            "regressor__epsilon": [1.1, 1.35, 1.5],
            "regressor__alpha": [0.0001, 0.001],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Low-dimensional + strong linear signal ────────────────
    if p <= 5 and linear > 0.50 and not has_cat:
        primary = Ridge()
        primary_name = "Ridge"
        alt = ElasticNet(max_iter=2000, random_state=42)
        alt_name = "ElasticNet"
        reason = (
            f"Low-dimensional ({p} numeric features) with strong linear "
            f"signal (avg |r|={linear:.2f}). Ridge provides stable, "
            "interpretable coefficients and avoids overfitting through L2 regularisation."
        )
        hp_grid = {
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Moderate linear signal, low skew → ElasticNet ─────────
    if linear > 0.40 and skew < 1.0 and not has_cat:
        primary = ElasticNet(max_iter=2000, random_state=42)
        primary_name = "ElasticNet"
        alt = Ridge()
        alt_name = "Ridge"
        reason = (
            f"Moderate linear signal (avg |r|={linear:.2f}) with low "
            f"feature skewness ({skew:.2f}). ElasticNet combines L1 "
            "sparsity and L2 stability — well-suited for correlated numeric features."
        )
        hp_grid = {
            "regressor__alpha": [0.01, 0.1, 1.0],
            "regressor__l1_ratio": [0.2, 0.5, 0.8],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── High skewness or mixed features → tree ensemble ───────
    primary = GradientBoostingRegressor(random_state=42)
    primary_name = "GradientBoostingRegressor"
    alt = RandomForestRegressor(random_state=42)
    alt_name = "RandomForestRegressor"
    reason = (
        f"Nonlinear regression task ({n:,} rows, avg |r|={linear:.2f}, "
        f"skewness={skew:.2f}). Gradient Boosting is robust to skewed "
        "distributions, mixed feature types, and nonlinear interactions."
    )
    hp_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [3, 5],
        "regressor__learning_rate": [0.05, 0.1],
    }
    return primary, primary_name, alt, alt_name, reason, hp_grid


def _clustering_choice(eda: EDAReport):
    n     = eda.n_rows
    out_r = eda.outlier_ratio
    sparse = eda.sparsity > 0.30

    # ── Outlier-heavy → DBSCAN ────────────────────────────────
    if out_r > 0.05:
        primary = DBSCAN(eps=0.5, min_samples=5)
        primary_name = "DBSCAN"
        alt = KMeans(n_clusters=4, random_state=42, n_init=10)
        alt_name = "KMeans"
        reason = (
            f"High outlier density ({out_r:.1%} of points > 3σ). "
            "DBSCAN naturally identifies noise points and discovers "
            "clusters of arbitrary shape without requiring k up front."
        )
        hp_grid = {
            "clusterer__eps": [0.3, 0.5, 0.8],
            "clusterer__min_samples": [3, 5, 10],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Large sparse data → MiniBatchKMeans ───────────────────
    if sparse or n > 20_000:
        primary = MiniBatchKMeans(random_state=42, n_init=10)
        primary_name = "MiniBatchKMeans"
        alt = KMeans(random_state=42, n_init=10)
        alt_name = "KMeans"
        reason = (
            f"Large ({n:,} rows) or sparse dataset. MiniBatchKMeans "
            "achieves near-identical results to full KMeans at a fraction "
            "of the memory and computation cost."
        )
        hp_grid = {
            "clusterer__n_clusters": [3, 4, 5, 6, 8],
        }
        return primary, primary_name, alt, alt_name, reason, hp_grid

    # ── Default KMeans ─────────────────────────────────────────
    primary = KMeans(random_state=42, n_init=10)
    primary_name = "KMeans"
    alt = DBSCAN()
    alt_name = "DBSCAN"
    reason = (
        f"Unsupervised clustering on {n:,} rows. KMeans is interpretable, "
        "scalable, and effective when clusters are roughly convex; "
        "DBSCAN offered as a density-based alternative."
    )
    hp_grid = {
        "clusterer__n_clusters": [2, 3, 4, 5, 6],
    }
    return primary, primary_name, alt, alt_name, reason, hp_grid


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────

def select_model(eda: EDAReport) -> tuple[object, object, ModelSelection]:
    """
    Data-driven model selection. Returns (primary_model, alternative_model, config).
    """
    task = eda.task_type

    if task == "classification":
        primary, p_name, alt, a_name, reason, hp_grid = _classification_choice(eda)
    elif task == "regression":
        primary, p_name, alt, a_name, reason, hp_grid = _regression_choice(eda)
    else:
        primary, p_name, alt, a_name, reason, hp_grid = _clustering_choice(eda)

    config = ModelSelection(
        selected_model=p_name,
        alternative_model=a_name,
        rule_reason=reason,
        hyperparameter_grid=hp_grid,
    )

    return primary, alt, config
