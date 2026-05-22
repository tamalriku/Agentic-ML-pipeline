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

    # Limit grid size for speed on small datasets
    trimmed_grid = {k: v[:2] for k, v in hp_grid.items()}

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

    best_params = {
        k.replace(f"{step_name}__", ""): v
        for k, v in search.best_params_.items()
    }
    cv_score = float(search.best_score_)
    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

    return search.best_estimator_, X_test, y_test, best_params, cv_score, cv_std
