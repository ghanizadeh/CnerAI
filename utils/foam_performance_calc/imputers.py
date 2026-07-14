"""
imputers.py
-----------
Missing-value imputation for foam volumes and texture labels.

Foam volume imputation
~~~~~~~~~~~~~~~~~~~~~~
Row-wise strategy: for each sample (row), fit all registered curve models
to the observed (non-NaN) foam values, then use the best-fitting model to
predict missing days.  The model selection can be forced to a specific model
or left as 'best' (highest R²).

Texture imputation
~~~~~~~~~~~~~~~~~~
For each missing texture on a given day, look backward and forward to the
nearest days that have a known texture with a registered weight.  The
imputed texture is the label whose weight is closest to the average of the
two boundary weights.  If only one boundary is found the texture is left
as-is (no extrapolation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cleaners import clean_texture_label
from .constants import (
    BEST_MODEL_COLUMN,
    MODEL_BEST,
    MODEL_NONE,
    R2_COLUMNS,
    TW_NORMALIZED_TEXTURE,
    TW_WEIGHT,
)
from .model_fitters import FITTER_REGISTRY, FitResult
from .schemas import ColumnMapping, PipelineConfig


# ---------------------------------------------------------------------------
# Foam volume imputation
# ---------------------------------------------------------------------------

def fill_missing_foam_rowwise(
    df: pd.DataFrame,
    column_mapping: ColumnMapping,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Impute missing foam volume values on a per-row basis.

    For each row:
      1. Extract the vector of observed (non-NaN) foam values and their day
         indices.
      2. Fit every model in ``FITTER_REGISTRY`` on the observed subset.
      3. Select the model with the highest R² (or use the forced model).
      4. If the selected model passes the R² threshold, fill NaN cells with
         the model's predictions at those day indices.

    R² scores and the winning model name are recorded in new columns
    (``R2_linear``, ``R2_exp``, ``R2_poly``, ``R2_rf``, ``Best_Model``).

    Parameters
    ----------
    df             : pd.DataFrame – valid samples (reset index).
    column_mapping : ColumnMapping
    config         : PipelineConfig

    Returns
    -------
    pd.DataFrame
        Copy of *df* with missing foam values filled and R² diagnostics added.
    """
    df = df.copy()
    foam_map = column_mapping.foam_map
    num_days = config.num_days
    t_full = np.arange(num_days + 1, dtype=float)

    r2_records: dict[str, list[float]] = {key: [] for key in FITTER_REGISTRY}
    best_model_names: list[str] = []

    for idx in df.index:
        # Build the full y-vector (NaN for days without a column)
        y_full = np.array(
            [
                df.at[idx, foam_map[d]] if d in foam_map else np.nan
                for d in range(num_days + 1)
            ],
            dtype=float,
        )

        observed_mask = np.isfinite(y_full)
        t_obs = t_full[observed_mask]
        y_obs = y_full[observed_mask]

        if len(y_obs) < 2:
            for key in FITTER_REGISTRY:
                r2_records[key].append(np.nan)
            best_model_names.append(MODEL_NONE)
            continue

        # Fit all models
        results: dict[str, FitResult] = {
            key: fitter(t_obs, y_obs)
            for key, fitter in FITTER_REGISTRY.items()
        }

        for key in FITTER_REGISTRY:
            r2_records[key].append(results[key].r2)

        # Select the model to use
        selected_key = _select_model(results, config.selected_model, config.r2_threshold)

        if selected_key == MODEL_NONE:
            best_model_names.append(MODEL_NONE)
            continue

        best_model_names.append(selected_key)
        best_result = results[selected_key]

        # Fill NaN foam cells using the selected model's predictions
        _apply_predictions(df, idx, foam_map, num_days, t_full, best_result, selected_key)

    # Write diagnostics
    for key in FITTER_REGISTRY:
        df[f"R2_{key}"] = r2_records[key]
    df[BEST_MODEL_COLUMN] = best_model_names

    return df


def _select_model(
    results: dict[str, FitResult],
    selected_model: str,
    r2_threshold: float,
) -> str:
    """
    Return the key of the model to use, or MODEL_NONE if no model qualifies.
    """
    if selected_model == MODEL_BEST:
        key = max(
            results,
            key=lambda k: results[k].r2 if np.isfinite(results[k].r2) else -np.inf,
        )
    else:
        key = selected_model

    r2 = results[key].r2
    if not np.isfinite(r2) or r2 < r2_threshold:
        return MODEL_NONE

    return key


def _apply_predictions(
    df: pd.DataFrame,
    row_idx: int,
    foam_map: dict[int, str],
    num_days: int,
    t_full: np.ndarray,
    result: FitResult,
    model_key: str,
) -> None:
    """
    Write model predictions into NaN cells of a single row (in-place).
    """
    for d in range(num_days + 1):
        if d not in foam_map:
            continue
        col = foam_map[d]
        if not np.isfinite(df.at[row_idx, col]):
            predicted = _predict_single(d, result, model_key)
            if predicted is not None:
                df.at[row_idx, col] = predicted


def _predict_single(day: int, result: FitResult, model_key: str) -> Optional[float]:
    """
    Generate a scalar prediction for *day* from the fitted result.
    """
    model = result.model
    if model_key == "linear":
        return float(model.predict([[day]])[0])
    if model_key == "exp":
        A, B = model
        return float(A * np.exp(B * day))
    if model_key == "poly":
        return float(np.polyval(model, day))
    if model_key == "rf":
        return float(model.predict([[day]])[0])
    return None


# ---------------------------------------------------------------------------
# Texture imputation
# ---------------------------------------------------------------------------

def fill_missing_textures(
    df: pd.DataFrame,
    texture_weights_df: pd.DataFrame,
    column_mapping: ColumnMapping,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Impute missing or unrecognised texture labels using neighbour interpolation.

    Strategy:
      - Build a ``{cleaned_label: weight}`` lookup from *texture_weights_df*.
      - For each missing texture on day *d*, search backward and forward for
        the nearest days with a valid, known label.
      - Impute as the label whose weight is closest to the average of the two
        boundary weights.
      - If only one boundary is found, the cell is left unchanged.

    Parameters
    ----------
    df                  : pd.DataFrame – valid samples after foam imputation.
    texture_weights_df  : pd.DataFrame – must have columns
                          ``Normalized_Texture`` and ``Weight``.
    column_mapping      : ColumnMapping
    config              : PipelineConfig

    Returns
    -------
    pd.DataFrame
        Copy of *df* with texture columns imputed where possible.
    """
    df = df.copy().reset_index(drop=True)
    texture_map = column_mapping.texture_map
    num_days = config.num_days

    # Build weight lookup (cleaned label → float weight)
    tw = texture_weights_df.copy()
    tw[TW_NORMALIZED_TEXTURE] = tw[TW_NORMALIZED_TEXTURE].apply(clean_texture_label)
    weight_lookup: dict[str, float] = dict(zip(tw[TW_NORMALIZED_TEXTURE], tw[TW_WEIGHT]))

    all_labels = list(weight_lookup.keys())
    all_weights = list(weight_lookup.values())

    # Clean texture columns in the data frame
    for col in df.columns:
        if "Texture" in col:
            df[col] = df[col].apply(clean_texture_label)

    for day in range(num_days + 1):
        if day not in texture_map:
            continue
        tex_col = texture_map[day]
        if tex_col not in df.columns:
            continue

        for i in df.index:
            current = df.at[i, tex_col]
            if current is not None and current in weight_lookup:
                continue   # already valid

            prev_weight = _find_neighbour_weight(df, i, day, texture_map, weight_lookup, direction="back")
            next_weight = _find_neighbour_weight(df, i, day, texture_map, weight_lookup, direction="forward", num_days=num_days)

            if prev_weight is not None and next_weight is not None:
                avg_w = (prev_weight + next_weight) / 2.0
                df.at[i, tex_col] = _nearest_label(avg_w, all_labels, all_weights)

    return df


def _find_neighbour_weight(
    df: pd.DataFrame,
    row_idx: int,
    day: int,
    texture_map: dict[int, str],
    weight_lookup: dict[str, float],
    direction: str,
    num_days: int = 0,
) -> Optional[float]:
    """
    Walk backward or forward from *day* to find the nearest valid texture weight.
    """
    if direction == "back":
        search_range = range(day - 1, -1, -1)
    else:
        search_range = range(day + 1, num_days + 1)

    for d in search_range:
        if d not in texture_map:
            continue
        label = df.at[row_idx, texture_map[d]]
        if label in weight_lookup:
            return weight_lookup[label]

    return None


def _nearest_label(
    target_weight: float,
    labels: list[str],
    weights: list[float],
) -> str:
    """Return the label whose weight is closest to *target_weight*."""
    diffs = [abs(w - target_weight) for w in weights]
    return labels[int(np.argmin(diffs))]
