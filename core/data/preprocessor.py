"""
core/data/preprocessor.py
Pure Python preprocessing utilities — no Streamlit here.

Public API
----------
impute_missing(df, strategy, columns)  -> pd.DataFrame
remove_outliers_iqr(df, columns, factor) -> pd.DataFrame
remove_outliers_zscore(df, columns, threshold) -> pd.DataFrame
detect_outliers(df, columns, method, **kw) -> pd.DataFrame   # summary table
encode_categoricals(X, columns, strategy) -> (pd.DataFrame, dict)
scale_features(X_train, X_test, strategy) -> (X_train_s, X_test_s, scaler)
split_data(X, y, test_size, random_state, stratify) -> dict

EDA helpers (reusable across pages)
-------------------------------------
extended_describe(df)        -> pd.DataFrame
categorical_summary(df)      -> pd.DataFrame
categorical_warnings(df)     -> pd.DataFrame
categorical_imbalance(df)    -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder
)
from config.settings import RANDOM_STATE, DEFAULT_TEST_SIZE


# ── Missing values ────────────────────────────────────────────────────

def impute_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Handle missing values in *columns* (default = all numeric columns).

    Parameters
    ----------
    strategy : "drop" | "mean" | "median" | "mode" | "constant"
    columns  : columns to impute; None → all numeric columns
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include="number").columns.tolist()

    if strategy == "drop":
        return df.dropna()
    for col in cols:
        if df[col].isna().any():
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
    return df


# ── Outlier detection & removal ───────────────────────────────────────

def detect_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "IQR",
    iqr_factor: float = 1.5,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """
    Return a summary DataFrame: Feature | Outliers | Percent(%).
    """
    rows = []
    for col in columns:
        x = df[col].dropna()
        if len(x) == 0:
            rows.append({"Feature": col, "Outliers": 0, "Percent (%)": 0.0})
            continue
        if method == "IQR":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            mask = (x < q1 - iqr_factor * iqr) | (x > q3 + iqr_factor * iqr)
        else:  # Z-score
            std = x.std()
            if std == 0:
                mask = pd.Series(False, index=x.index)
            else:
                z = (x - x.mean()) / std
                mask = z.abs() > z_thresh
        n = int(mask.sum())
        rows.append({"Feature": col, "Outliers": n, "Percent (%)": round(100 * n / len(x), 2)})
    return pd.DataFrame(rows)


def _build_outlier_mask(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    iqr_factor: float = 1.5,
    z_thresh: float = 3.0,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        x = df[col]
        if method == "IQR":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            mask &= x.between(q1 - iqr_factor * iqr, q3 + iqr_factor * iqr)
        else:
            z = (x - x.mean()) / x.std()
            mask &= z.abs() <= z_thresh
    return mask


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "IQR",
    iqr_factor: float = 1.5,
    z_thresh: float = 3.0,
) -> tuple[pd.DataFrame, int]:
    """
    Remove outlier rows.  Returns (cleaned_df, n_removed).
    """
    mask = _build_outlier_mask(df, columns, method, iqr_factor, z_thresh)
    cleaned = df[mask]
    return cleaned, int((~mask).sum())


# ── Encoding ──────────────────────────────────────────────────────────

def encode_categoricals(
    X: pd.DataFrame,
    columns: list[str] | None = None,
    strategy: str = "onehot",
) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns.

    Parameters
    ----------
    strategy : "onehot" | "label" | "ordinal"

    Returns
    -------
    (encoded_X, encoder_map)  — encoder_map stored for inverse transform
    """
    X = X.copy()
    cols = columns or X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoder_map: dict = {}

    if strategy == "onehot":
        X = pd.get_dummies(X, columns=cols, drop_first=False)
    elif strategy == "label":
        for col in cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoder_map[col] = le
    elif strategy == "ordinal":
        oe = OrdinalEncoder()
        X[cols] = oe.fit_transform(X[cols].astype(str))
        encoder_map["__ordinal__"] = oe

    return X, encoder_map


# ── Scaling ───────────────────────────────────────────────────────────

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Fit scaler on X_train, transform both sets.

    Returns (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    if strategy not in _SCALERS:
        raise ValueError(f"Unknown scaler '{strategy}'. Choose from {list(_SCALERS)}")
    scaler = _SCALERS[strategy]()
    cols = X_train.columns
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
    return X_train_s, X_test_s, scaler


# ── Train / test split ────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = False,
) -> dict:
    """
    Split X, y into train / test sets.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ── EDA helpers (reusable on any page) ───────────────────────────────

def extended_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Extended summary: counts + null info + pandas describe."""
    total = len(df)
    summary = pd.DataFrame(
        {
            "total_count": total,
            "not_null_count": df.notnull().sum(),
            "null_count": df.isnull().sum(),
            "null_%": (df.isnull().mean() * 100).round(2),
            "data_type": df.dtypes.astype(str),
            "n_unique": df.nunique(),
        }
    )
    desc = df.describe(include="all").T
    return summary.join(desc)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary table for categorical / object columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return pd.DataFrame()
    rows = []
    for col in cat_cols:
        rows.append(
            {
                "Column": col,
                "n_unique": df[col].nunique(),
                "top_value": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "top_freq": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
                "null_count": df[col].isnull().sum(),
            }
        )
    return pd.DataFrame(rows).set_index("Column")


def categorical_warnings(df: pd.DataFrame) -> pd.DataFrame:
    """Return a warning table for high cardinality / imbalance issues."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    rows = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_freq_pct = df[col].value_counts(normalize=True).iloc[0] * 100 if n_unique else 0
        issues = []
        if n_unique > 20:
            issues.append(f"High cardinality ({n_unique} unique)")
        if top_freq_pct > 90:
            issues.append(f"Dominant class ({top_freq_pct:.1f} %)")
        if issues:
            rows.append({"Column": col, "Warning": "; ".join(issues)})
    return pd.DataFrame(rows)


def categorical_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Return value-counts (%) for all categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    frames = []
    for col in cat_cols:
        vc = df[col].value_counts(normalize=True).mul(100).round(2).reset_index()
        vc.columns = ["Value", "Percent (%)"]
        vc.insert(0, "Column", col)
        frames.append(vc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()