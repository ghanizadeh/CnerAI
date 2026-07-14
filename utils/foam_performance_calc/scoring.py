"""
scoring.py
----------
Column renaming and performance index computation.

Responsibilities
~~~~~~~~~~~~~~~~
1. ``rename_to_day_format``  – rename raw foam/texture columns to the
   standardised ``Day d - Foam (cc)`` / ``Day d - Foam Texture`` scheme.

2. ``compute_performance``   – given the renamed DataFrame plus a texture
   weight lookup, calculate per-row ``Score_Volume``, ``Score_Texture``,
   and ``Performance_Index``.

3. ``merge_scores_to_full``  – merge the three score columns back onto the
   original (full) DataFrame so non-valid rows get NaN scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaners import clean_texture_label
from .constants import (
    COL_PERFORMANCE_INDEX,
    COL_SCORE_TEXTURE,
    COL_SCORE_VOLUME,
    DAY_FOAM_TEMPLATE,
    DAY_TEXTURE_TEMPLATE,
    SCORE_COLUMNS,
    TW_NORMALIZED_TEXTURE,
    TW_WEIGHT,
)
from .schemas import ColumnMapping, PipelineConfig


# ---------------------------------------------------------------------------
# Column renaming
# ---------------------------------------------------------------------------

def rename_to_day_format(
    df: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> pd.DataFrame:
    """
    Rename raw foam and texture columns to the ``Day d - …`` convention.

    Parameters
    ----------
    df             : pd.DataFrame – post-imputation DataFrame.
    column_mapping : ColumnMapping

    Returns
    -------
    pd.DataFrame
        Copy with renamed columns.
    """
    rename_map: dict[str, str] = {}

    for day, col in column_mapping.foam_map.items():
        rename_map[col] = DAY_FOAM_TEMPLATE.format(day=day)

    for day, col in column_mapping.texture_map.items():
        rename_map[col] = DAY_TEXTURE_TEMPLATE.format(day=day)

    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_performance(
    df: pd.DataFrame,
    texture_weights_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute ``Score_Volume``, ``Score_Texture``, and ``Performance_Index``.

    The DataFrame is expected to have already been renamed via
    ``rename_to_day_format`` so columns follow the ``Day d - Foam (cc)`` /
    ``Day d - Foam Texture`` pattern.

    Parameters
    ----------
    df                 : pd.DataFrame – renamed post-imputation data.
    texture_weights_df : pd.DataFrame – weight lookup table.
    config             : PipelineConfig

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - scored DataFrame with three new columns appended.
        - sorted list of texture labels that were not found in the weight file.
    """
    df = df.copy()

    # Coerce volume columns to numeric
    for col in df.columns:
        if "(cc)" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build texture weight lookup
    tw = texture_weights_df.copy()
    tw[TW_NORMALIZED_TEXTURE] = tw[TW_NORMALIZED_TEXTURE].apply(clean_texture_label)
    weight_lookup: dict[str, float] = dict(zip(tw[TW_NORMALIZED_TEXTURE], tw[TW_WEIGHT]))

    # Clean texture columns
    for col in df.columns:
        if "Texture" in col:
            df[col] = df[col].apply(clean_texture_label)

    score_volumes: list[float] = []
    score_textures: list[float] = []
    performance_indices: list[float] = []
    missing_textures: set[str] = set()

    for _, row in df.iterrows():
        sv, st = _score_row(row, config, weight_lookup, missing_textures)
        score_volumes.append(sv)
        score_textures.append(st)
        performance_indices.append(sv + st)

    df[COL_SCORE_VOLUME] = score_volumes
    df[COL_SCORE_TEXTURE] = score_textures
    df[COL_PERFORMANCE_INDEX] = performance_indices

    return df, sorted(missing_textures)


def _score_row(
    row: pd.Series,
    config: PipelineConfig,
    weight_lookup: dict[str, float],
    missing_textures: set[str],
) -> tuple[float, float]:
    """
    Compute ``(score_volume, score_texture)`` for a single row.
    """
    score_volume = 0.0
    score_texture = 0.0

    for day in range(config.num_days + 1):
        day_weight = config.weight_for(day)
        vol_col = DAY_FOAM_TEMPLATE.format(day=day)
        tex_col = DAY_TEXTURE_TEMPLATE.format(day=day)

        # Volume contribution
        volume = row.get(vol_col, np.nan)
        if pd.notna(volume):
            try:
                score_volume += day_weight * float(volume)
            except (ValueError, TypeError):
                pass

        # Texture contribution
        texture = row.get(tex_col)
        if pd.notna(texture) and texture is not None:
            if texture in weight_lookup:
                score_texture += day_weight * weight_lookup[texture]
            else:
                missing_textures.add(str(texture))

    return score_volume, score_texture


# ---------------------------------------------------------------------------
# Merge scores onto the full (original) DataFrame
# ---------------------------------------------------------------------------

def merge_scores_to_full(
    original_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    original_valid_indices: list[int],
) -> pd.DataFrame:
    """
    Attach score columns from *scored_df* back to every row of *original_df*.

    Rows that were filtered out as invalid receive ``NaN`` in the score columns.

    Parameters
    ----------
    original_df            : pd.DataFrame – full original dataset (all rows).
    scored_df              : pd.DataFrame – scored valid subset (reset index).
    original_valid_indices : list[int] – original index labels for valid rows.

    Returns
    -------
    pd.DataFrame
        Copy of *original_df* with ``Score_Volume``, ``Score_Texture``, and
        ``Performance_Index`` columns appended.
    """
    full_output = original_df.copy()
    full_output[SCORE_COLUMNS] = np.nan
    full_output.loc[original_valid_indices, SCORE_COLUMNS] = (
        scored_df[SCORE_COLUMNS].values
    )
    return full_output
