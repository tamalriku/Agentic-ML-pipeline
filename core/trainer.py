import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer


def train_model(
    preprocessor: ColumnTransformer,
    model,
    X,
    y,
    task_type: str,
    hp_grid: dict,
    model_name: str,
) -> tuple:
    """
    Builds full sklearn Pipeline, runs GridSearchCV, returns
    (best_estimator, X_test, y_test, best_params, cv_score, cv_std).

    Handles any step name embedded in hp_grid keys automatically so
    wrapped models (e.g. CalibratedClassifierCV) work without manual fixes.
    """
    if task_type == "clustering":
        step_name = "clusterer"
    elif task_type == "regression":
        step_name = "regressor"
    else:
        step_name = "classifier"

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        (step_name, model),
    ])

    if task_type == "clustering":
        pipe.fit(X)
        return pipe, None, None, {}, 0.0, 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    scoring = "f1_weighted" if task_type == "classification" else "r2"
    cv_folds = min(5, len(y_train) // 10) if len(y_train) < 50 else 5
    cv_folds = max(cv_folds, 2)

    # Re-key the hp_grid so every key is prefixed with the correct step_name.
    # The selector already prefixes keys (e.g. "classifier__C"), but we
    # normalise here in case a key comes with a different prefix or no prefix.
    normalised_grid: dict = {}
    for k, v in hp_grid.items():
        parts = k.split("__", 1)
        if len(parts) == 2 and parts[0] in ("classifier", "regressor", "clusterer"):
            # Replace whatever step name the selector used with the actual step name
            normalised_grid[f"{step_name}__{parts[1]}"] = v
        elif "__" not in k:
            normalised_grid[f"{step_name}__{k}"] = v
        else:
            # Pass through as-is (nested like classifier__base_estimator__C)
            normalised_grid[k] = v

    # Limit grid size for speed on large datasets
    trimmed_grid = {k: v[:2] for k, v in normalised_grid.items()}

    # Fall back to a no-op single-point grid if hp_grid was empty
    if not trimmed_grid:
        trimmed_grid = {}

    try:
        search = GridSearchCV(
            pipe,
            param_grid=trimmed_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
    except Exception:
        # If grid search fails (e.g. bad param combo), fall back to plain fit
        pipe.fit(X_train, y_train)
        return pipe, X_test, y_test, {}, 0.0, 0.0

    best_params = {
        k.replace(f"{step_name}__", "", 1): v
        for k, v in search.best_params_.items()
    }
    cv_score = float(search.best_score_)
    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

    return search.best_estimator_, X_test, y_test, best_params, cv_score, cv_std
