"""
core/models/evaluator.py
Compute evaluation metrics and feature importance — no Streamlit here.

Public API
----------
get_classification_metrics(y_true, y_pred, y_prob) -> dict
get_regression_metrics(y_true, y_pred)             -> dict
get_feature_importance(model, feature_names)       -> pd.DataFrame | None
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)


def get_classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average: str = "weighted",
) -> dict:
    """Return a dict of classification metrics."""
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
                )
        except Exception:
            pass

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
    return metrics


def get_regression_metrics(y_true, y_pred) -> dict:
    """Return a dict of regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """
    Extract feature importances from a fitted model.

    Returns a DataFrame sorted by importance descending,
    or None if the model does not expose importances.
    """
    importance = None

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importance is None:
        return None

    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df
