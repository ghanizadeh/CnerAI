# core/data/feature_engineering.py
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
import pandas as pd


@dataclass
class FeatureEngineeringConfig:
    enabled: bool = False

    # numeric transforms
    add_log: bool = False
    add_sqrt: bool = False
    add_square: bool = False
    add_cube: bool = False
    add_inverse: bool = False

    # pairwise engineering
    add_interactions: bool = False
    add_ratios: bool = False

    # binning
    add_quantile_bins: bool = False
    n_bins: int = 5

    # safety / size controls
    max_base_features: int = 8
    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(default_factory=list)
    drop_original_binned: bool = False

# core/data/feature_engineering.py
def _safe_numeric_columns(
    df: pd.DataFrame,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    exclude_columns = exclude_columns or []
    cols = include_columns or df.select_dtypes(include="number").columns.tolist()
    cols = [c for c in cols if c in df.columns and c not in exclude_columns]
    return cols


def _limit_columns(cols: list[str], max_base_features: int) -> list[str]:
    return cols[:max_base_features]


def add_math_features(
    df: pd.DataFrame,
    columns: list[str],
    *,
    add_log: bool = False,
    add_sqrt: bool = False,
    add_square: bool = False,
    add_cube: bool = False,
    add_inverse: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        x = pd.to_numeric(out[col], errors="coerce")

        if add_log:
            # only valid for strictly positive values
            if (x > 0).all():
                out[f"log_{col}"] = np.log(x)

        if add_sqrt:
            # use abs to avoid invalid values
            out[f"sqrt_{col}"] = np.sqrt(np.abs(x))

        if add_square:
            out[f"{col}_sq"] = x ** 2

        if add_cube:
            out[f"{col}_cube"] = x ** 3

        if add_inverse:
            out[f"inv_{col}"] = 1.0 / (x.replace(0, np.nan))

    return out


def add_interaction_features(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    out = df.copy()

    for c1, c2 in combinations(columns, 2):
        out[f"{c1}__x__{c2}"] = out[c1] * out[c2]

    return out


def add_ratio_features(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    out = df.copy()

    for c1, c2 in combinations(columns, 2):
        denom = out[c2].replace(0, np.nan)
        out[f"{c1}__div__{c2}"] = out[c1] / denom

    return out


def add_quantile_binning(
    df: pd.DataFrame,
    columns: list[str],
    n_bins: int = 5,
    drop_original: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        try:
            out[f"{col}_bin"] = pd.qcut(
                out[col],
                q=n_bins,
                duplicates="drop",
                labels=False,
            )
            if drop_original:
                out = out.drop(columns=[col])
        except Exception:
            # skip columns that cannot be binned
            continue

    return out

# core/data/feature_engineering.py
def apply_feature_engineering(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply optional feature engineering and return:
    (engineered_df, metadata)
    """
    if not config.enabled:
        return df.copy(), {
            "enabled": False,
            "original_columns": list(df.columns),
            "new_columns": [],
            "n_added": 0,
        }

    out = df.copy()

    numeric_cols = _safe_numeric_columns(
        out,
        include_columns=config.include_columns,
        exclude_columns=config.exclude_columns,
    )
    base_cols = _limit_columns(numeric_cols, config.max_base_features)

    original_cols = list(out.columns)

    # 1) math transforms
    out = add_math_features(
        out,
        base_cols,
        add_log=config.add_log,
        add_sqrt=config.add_sqrt,
        add_square=config.add_square,
        add_cube=config.add_cube,
        add_inverse=config.add_inverse,
    )

    # 2) interactions
    if config.add_interactions:
        out = add_interaction_features(out, base_cols)

    # 3) ratios
    if config.add_ratios:
        out = add_ratio_features(out, base_cols)

    # 4) quantile bins
    if config.add_quantile_bins:
        out = add_quantile_binning(
            out,
            base_cols,
            n_bins=config.n_bins,
            drop_original=config.drop_original_binned,
        )

    new_cols = [c for c in out.columns if c not in original_cols]

    meta = {
        "enabled": True,
        "base_columns_used": base_cols,
        "original_columns": original_cols,
        "new_columns": new_cols,
        "n_added": len(new_cols),
    }

    return out, meta