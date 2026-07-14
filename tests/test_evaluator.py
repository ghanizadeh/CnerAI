"""tests/test_evaluator.py"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from core.models.evaluator import (
    get_classification_metrics,
    get_regression_metrics,
    get_feature_importance,
)


@pytest.fixture
def clf_results():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=50)
    y_pred = rng.integers(0, 2, size=50)
    y_prob = rng.random((50, 2))
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_prob


@pytest.fixture
def reg_results():
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(50)
    y_pred = y_true + rng.standard_normal(50) * 0.1
    return y_true, y_pred


@pytest.fixture
def fitted_clf():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, [f"f{i}" for i in range(5)]


@pytest.fixture
def fitted_reg():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, [f"f{i}" for i in range(5)]


# ── Classification metrics ────────────────────────────────────────────

def test_clf_metrics_keys(clf_results):
    y_true, y_pred, y_prob = clf_results
    m = get_classification_metrics(y_true, y_pred, y_prob)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in m, f"Missing key: {key}"


def test_clf_accuracy_range(clf_results):
    y_true, y_pred, _ = clf_results
    m = get_classification_metrics(y_true, y_pred)
    assert 0.0 <= m["accuracy"] <= 1.0


def test_clf_confusion_matrix_present(clf_results):
    y_true, y_pred, _ = clf_results
    m = get_classification_metrics(y_true, y_pred)
    assert "confusion_matrix" in m
    assert len(m["confusion_matrix"]) == 2  # binary


# ── Regression metrics ────────────────────────────────────────────────

def test_reg_metrics_keys(reg_results):
    y_true, y_pred = reg_results
    m = get_regression_metrics(y_true, y_pred)
    for key in ("r2", "mae", "mse", "rmse"):
        assert key in m


def test_rmse_equals_sqrt_mse(reg_results):
    y_true, y_pred = reg_results
    m = get_regression_metrics(y_true, y_pred)
    assert abs(m["rmse"] - np.sqrt(m["mse"])) < 1e-9


def test_r2_near_one_for_good_preds():
    y_true = np.arange(50, dtype=float)
    y_pred = y_true + np.random.default_rng(0).standard_normal(50) * 0.01
    m = get_regression_metrics(y_true, y_pred)
    assert m["r2"] > 0.99


# ── Feature importance ────────────────────────────────────────────────

def test_feature_importance_clf(fitted_clf):
    model, names = fitted_clf
    df = get_feature_importance(model, names)
    assert df is not None
    assert list(df.columns) == ["Feature", "Importance"]
    assert len(df) == len(names)
    # Should be sorted descending
    assert df["Importance"].is_monotonic_decreasing


def test_feature_importance_reg(fitted_reg):
    model, names = fitted_reg
    df = get_feature_importance(model, names)
    assert df is not None
    assert df["Importance"].sum() > 0


def test_feature_importance_none_for_unsupported():
    """Model with no coef_ or feature_importances_ returns None."""
    from sklearn.neighbors import KNeighborsClassifier
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    model = KNeighborsClassifier().fit(X, y)
    result = get_feature_importance(model, ["a", "b", "c", "d"])
    assert result is None
