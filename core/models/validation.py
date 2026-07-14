"""
core/models/validation.py
=========================
Unified validation strategy factory.

Supports:
  - Train / Test Split
  - K-Fold Cross Validation
  - Stratified K-Fold  (classification only)
  - Leave-One-Out CV (LOOCV)
  - Leave-One-Group-Out (requires group column)

Public API
----------
get_cv_strategy(method, config, X, y, groups=None)
    -> cv splitter OR (X_train, X_test, y_train, y_test) for simple splits

run_validation(method, config, model, X, y, scoring, groups=None)
    -> dict with scores, mean, std
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from config.settings import RANDOM_STATE

warnings.filterwarnings("ignore")

# ── Validation method constants (used in UI) ─────────────────────────────────
METHOD_TRAIN_TEST   = "Train / Test Split"
METHOD_KFOLD        = "K-Fold Cross Validation"
METHOD_STRATIFIED   = "Stratified K-Fold"
METHOD_LOOCV        = "Leave-One-Out CV (LOOCV)"
METHOD_LOGO         = "Leave-One-Group-Out"

ALL_METHODS = [
    METHOD_TRAIN_TEST,
    METHOD_KFOLD,
    METHOD_STRATIFIED,
    METHOD_LOOCV,
    METHOD_LOGO,
]

CLASSIFICATION_METHODS = ALL_METHODS  # all methods available for classification
REGRESSION_METHODS = [
    METHOD_TRAIN_TEST,
    METHOD_KFOLD,
    METHOD_LOOCV,
    METHOD_LOGO,
]


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def get_cv_strategy(
    method: str,
    config: dict,
    task_type: str = "classification",
):
    """
    Return an sklearn CV splitter for the given method and config.

    For METHOD_TRAIN_TEST this returns None (splitting is done separately
    via split_for_training).

    Parameters
    ----------
    method    : one of the METHOD_* constants
    config    : dict of method-specific settings (n_splits, shuffle, etc.)
    task_type : "classification" | "regression"

    Returns
    -------
    sklearn CV splitter or None (for train/test split)
    """
    n_splits     = int(config.get("n_splits", 5))
    shuffle      = bool(config.get("shuffle", True))
    random_state = int(config.get("random_state", RANDOM_STATE))

    if method == METHOD_TRAIN_TEST:
        return None  # caller uses split_for_training()

    if method == METHOD_KFOLD:
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)

    if method == METHOD_STRATIFIED:
        if task_type != "classification":
            raise ValueError("Stratified K-Fold is only valid for classification tasks.")
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)

    if method == METHOD_LOOCV:
        return LeaveOneOut()

    if method == METHOD_LOGO:
        return LeaveOneGroupOut()

    raise ValueError(f"Unknown validation method: {method!r}")


def split_for_training(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
    stratify: bool = False,
):
    """
    Simple train/test split used when method == METHOD_TRAIN_TEST.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    stratify_arr = y if stratify else None
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_arr,
    )


def run_validation(
    method: str,
    config: dict,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    task_type: str = "classification",
    groups: Optional[pd.Series] = None,
) -> dict:
    """
    Run the full validation and return a results dict.

    For METHOD_TRAIN_TEST: fits on train, scores on test.
    For CV methods: runs cross_val_score.

    Returns
    -------
    {
        "method"   : str,
        "scores"   : list[float],
        "mean"     : float,
        "std"      : float,
        "scoring"  : str,
    }
    """
    random_state = int(config.get("random_state", RANDOM_STATE))

    if method == METHOD_TRAIN_TEST:
        test_size = float(config.get("test_size", 0.2))
        shuffle   = bool(config.get("shuffle", True))
        stratify  = bool(config.get("stratify", False)) and task_type == "classification"
        X_tr, X_te, y_tr, y_te = split_for_training(
            X, y, test_size=test_size, random_state=random_state,
            shuffle=shuffle, stratify=stratify,
        )
        model.fit(X_tr, y_tr)
        from sklearn.metrics import get_scorer
        scorer = get_scorer(scoring)
        score  = scorer(model, X_te, y_te)
        scores = [score]

    else:
        cv = get_cv_strategy(method, config, task_type)
        grp = groups.values if groups is not None and method == METHOD_LOGO else None
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, groups=grp)
        scores = scores.tolist()

    return {
        "method":  method,
        "scores":  scores,
        "mean":    float(np.mean(scores)),
        "std":     float(np.std(scores)),
        "scoring": scoring,
    }
