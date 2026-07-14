"""
core/models/trainer.py
Model training helpers — no Streamlit here.

Public API
----------
train(model, X_train, y_train)                  -> fitted model
cross_validate_model(model, X, y, cv, scoring)  -> dict of cv results
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from config.settings import RANDOM_STATE, DEFAULT_CV_FOLDS


def train(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit model and return it (mutates in place, also returns for chaining)."""
    model.fit(X_train, y_train)
    return model


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = DEFAULT_CV_FOLDS,
    scoring: str | list | None = None,
    task_type: str = "classification",
) -> dict:
    """
    Run k-fold cross-validation.

    Returns dict with mean ± std for each metric.
    """
    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        default_scoring = ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        default_scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

    used_scoring = scoring or default_scoring

    try:
        cv_results = cross_validate(
            model, X, y, cv=splitter, scoring=used_scoring, return_train_score=False
        )
    except Exception:
        # Fallback to single score if multi-metric fails
        cv_results = cross_validate(model, X, y, cv=splitter, return_train_score=False)

    summary = {}
    for key, values in cv_results.items():
        if key.startswith("test_"):
            metric = key[5:]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values.tolist(),
            }
    return summary
