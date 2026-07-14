"""tests/test_trainer.py"""
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from core.models.registry import get_model_names, get_model_instance, get_default_params
from core.models.trainer import train, cross_validate_model


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


# ── Registry ──────────────────────────────────────────────────────────

def test_registry_returns_clf_names():
    names = get_model_names("classification")
    assert len(names) > 0
    assert "LogisticRegression" in names


def test_registry_returns_reg_names():
    names = get_model_names("regression")
    assert "LinearRegression" in names


def test_get_default_params():
    params = get_default_params("classification", "LogisticRegression")
    assert "max_iter" in params


def test_get_model_instance():
    model = get_model_instance("classification", "LogisticRegression")
    assert hasattr(model, "fit")


def test_get_model_instance_custom_params():
    model = get_model_instance("classification", "LogisticRegression", {"max_iter": 500})
    assert model.max_iter == 500


# ── Train ─────────────────────────────────────────────────────────────

def test_train_classification(clf_data):
    X, y = clf_data
    model = get_model_instance("classification", "LogisticRegression")
    fitted = train(model, X, y)
    assert hasattr(fitted, "predict")
    preds = fitted.predict(X)
    assert len(preds) == len(y)


def test_train_regression(reg_data):
    X, y = reg_data
    model = get_model_instance("regression", "LinearRegression")
    fitted = train(model, X, y)
    preds = fitted.predict(X)
    assert len(preds) == len(y)


# ── Cross-validate ────────────────────────────────────────────────────

def test_cross_validate_classification(clf_data):
    X, y = clf_data
    model = get_model_instance("classification", "LogisticRegression")
    results = cross_validate_model(model, X, y, cv=3, task_type="classification")
    assert len(results) > 0
    first = next(iter(results.values()))
    assert "mean" in first and "std" in first


def test_cross_validate_regression(reg_data):
    X, y = reg_data
    model = get_model_instance("regression", "LinearRegression")
    results = cross_validate_model(model, X, y, cv=3, task_type="regression")
    assert len(results) > 0
