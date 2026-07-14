"""tests/test_preprocessor.py"""
import numpy as np
import pandas as pd
import pytest

from core.data.preprocessor import (
    impute_missing, detect_outliers, remove_outliers,
    encode_categoricals, scale_features, split_data,
    categorical_summary, categorical_warnings,
)


@pytest.fixture
def num_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 100.0],  # 100 is an outlier
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def cat_df():
    return pd.DataFrame({
        "color": ["red", "blue", "red", "green", "blue"],
        "size":  ["S", "M", "L", "S", "M"],
        "target": [1, 0, 1, 0, 1],
    })


# ── Impute ────────────────────────────────────────────────────────────

def test_impute_drop(num_df):
    result = impute_missing(num_df, strategy="drop")
    assert result.isnull().sum().sum() == 0
    assert len(result) == 4  # one row dropped


def test_impute_mean(num_df):
    result = impute_missing(num_df, strategy="mean", columns=["a"])
    assert result["a"].isnull().sum() == 0


def test_impute_median(num_df):
    result = impute_missing(num_df, strategy="median", columns=["a"])
    assert result["a"].isnull().sum() == 0


# ── Outliers ──────────────────────────────────────────────────────────

def test_detect_outliers_iqr(num_df):
    df = num_df.dropna()
    summary = detect_outliers(df, ["a"], method="IQR")
    assert summary.loc[summary["Feature"] == "a", "Outliers"].values[0] >= 1


def test_remove_outliers_iqr(num_df):
    df = num_df.dropna()
    cleaned, n_removed = remove_outliers(df, ["a"], method="IQR")
    assert n_removed >= 1
    assert len(cleaned) < len(df)


# ── Encoding ──────────────────────────────────────────────────────────

def test_encode_onehot(cat_df):
    X = cat_df[["color", "size"]]
    X_enc, _ = encode_categoricals(X, strategy="onehot")
    assert X_enc.shape[1] > 2  # expanded columns


def test_encode_label(cat_df):
    X = cat_df[["color", "size"]]
    X_enc, enc_map = encode_categoricals(X, strategy="label")
    assert X_enc["color"].dtype in [np.int64, np.int32, int]
    assert "color" in enc_map


# ── Scaling ───────────────────────────────────────────────────────────

def test_scale_standard(num_df):
    df = num_df.dropna()
    X = df[["a", "b"]]
    split = split_data(X, df["target"], test_size=0.4)
    X_tr_s, X_te_s, scaler = scale_features(split["X_train"], split["X_test"], "standard")
    # Training mean should be ~0 after standard scaling
    assert abs(X_tr_s.mean().mean()) < 1e-9


# ── Split ─────────────────────────────────────────────────────────────

def test_split_data_sizes(num_df):
    df = num_df.dropna()
    X = df[["a", "b"]]
    y = df["target"]
    result = split_data(X, y, test_size=0.4)
    assert "X_train" in result and "X_test" in result
    assert len(result["X_train"]) + len(result["X_test"]) == len(X)


# ── Categorical helpers ───────────────────────────────────────────────

def test_categorical_summary(cat_df):
    summary = categorical_summary(cat_df)
    assert "color" in summary.index
    assert "n_unique" in summary.columns


def test_categorical_warnings_high_cardinality():
    df = pd.DataFrame({"col": [str(i) for i in range(25)]})
    warnings = categorical_warnings(df)
    assert not warnings.empty
