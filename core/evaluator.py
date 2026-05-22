import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from models.schemas import EDAReport, ModelSelection, EvaluationReport


def evaluate_model(
    pipeline,
    X_test,
    y_test,
    eda: EDAReport,
    selection: ModelSelection,
    best_params: dict,
    cv_score: float,
    cv_std: float,
    X_full: pd.DataFrame,
) -> EvaluationReport:
    task = eda.task_type

    if task == "clustering":
        return EvaluationReport(
            model_name=selection.selected_model,
            task_type=task,
            best_params=best_params,
            metrics={"note": 0.0},
            feature_importance=None,
            cv_score=0.0,
            cv_std=0.0,
        )

    y_pred = pipeline.predict(X_test)
    metrics = {}

    if task == "classification":
        metrics["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
        metrics["f1_weighted"] = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        try:
            if eda.n_classes == 2:
                proba = pipeline.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(y_test, proba), 4)
            else:
                proba = pipeline.predict_proba(X_test)
                metrics["roc_auc"] = round(roc_auc_score(y_test, proba, multi_class="ovr"), 4)
        except Exception:
            pass

    elif task == "regression":
        metrics["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
        metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        metrics["r2"] = round(r2_score(y_test, y_pred), 4)

    # Feature importance
    feature_importance = None
    try:
        step_name = "classifier" if task == "classification" else "regressor"
        final_model = pipeline.named_steps[step_name]

        feature_names = X_full.columns.tolist()

        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_
            # Map back through preprocessor if possible
            fi = sorted(
                [{"feature": f, "score": round(float(s), 4)} for f, s in zip(feature_names, importances[:len(feature_names)])],
                key=lambda x: x["score"], reverse=True
            )[:10]
            feature_importance = fi
        elif hasattr(final_model, "coef_"):
            coef = np.abs(final_model.coef_).flatten()
            fi = sorted(
                [{"feature": f, "score": round(float(s), 4)} for f, s in zip(feature_names, coef[:len(feature_names)])],
                key=lambda x: x["score"], reverse=True
            )[:10]
            feature_importance = fi
    except Exception:
        pass

    return EvaluationReport(
        model_name=selection.selected_model,
        task_type=task,
        best_params=best_params,
        metrics=metrics,
        feature_importance=feature_importance,
        cv_score=round(cv_score, 4),
        cv_std=round(cv_std, 4),
    )
