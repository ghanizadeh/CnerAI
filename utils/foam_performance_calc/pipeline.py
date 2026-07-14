"""
pipeline.py
-----------
Orchestration layer: connects all processing steps into a single callable
that the Streamlit page invokes with a config object and raw file inputs.

The pipeline executes these stages in order:

  1.  Read & parse input CSVs          (io_utils)
  2.  Deduplicate column names          (io_utils)
  3.  Map foam / texture columns        (column_mapping)
  4.  Clean foam volume columns         (cleaners)
  5.  Validate & filter samples         (validators)
  6.  Impute missing foam volumes       (imputers)
  7.  Impute missing texture labels     (imputers)
  8.  Rename columns to Day-d format    (scoring)
  9.  Compute performance scores        (scoring)
  10. Merge scores onto original df     (scoring)

A ``PipelineResult`` is returned so the UI layer can display each
intermediate stage without re-running the computation.
"""

from __future__ import annotations

from typing import Union
import io

import numpy as np
import pandas as pd

from .cleaners import clean_foam_columns
from .column_mapping import map_day_columns
from .imputers import fill_missing_foam_rowwise, fill_missing_textures
from .io_utils import load_data_safe
from .scoring import compute_performance, merge_scores_to_full, rename_to_day_format
from .schemas import ColumnMapping, PipelineConfig, PipelineResult
from .validators import filter_valid_samples
from .constants import R2_COLUMNS


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
from .io_utils import load_data_safe

def run_pipeline(
    foam_source,
    texture_weights_source,
    config,
    foam_sheet=None,
    texture_sheet=None,
):
    """
    Execute the full foam performance calculation pipeline.

    Parameters
    ----------
    foam_source            : File path or Streamlit UploadedFile for foam data.
    texture_weights_source : File path or Streamlit UploadedFile for texture weights.
    config                 : ``PipelineConfig`` with user-supplied settings.

    Returns
    -------
    PipelineResult
        Structured object containing all intermediate and final DataFrames,
        diagnostics, and metadata needed by the UI layer.

    Raises
    ------
    ValueError
        If either CSV cannot be parsed or no valid samples are found.
    """
    # ------------------------------------------------------------------
    # Stage 1–2: Read and normalise headers
    # ------------------------------------------------------------------
    foam_df = load_data_safe(foam_source, sheet_name=foam_sheet)
    texture_weights_df = load_data_safe(texture_weights_source, sheet_name=texture_sheet)

    # ------------------------------------------------------------------
    # Stage 3: Detect column mapping
    # ------------------------------------------------------------------
    column_mapping: ColumnMapping = map_day_columns(foam_df)

    # ------------------------------------------------------------------
    # Stage 4: Clean foam volume columns
    # ------------------------------------------------------------------
    foam_df = clean_foam_columns(foam_df, column_mapping.foam_map)

    # Keep a clean copy of the full original DataFrame for the final merge
    original_df = foam_df.copy()

    # ------------------------------------------------------------------
    # Stage 5: Filter valid samples
    # ------------------------------------------------------------------
    valid_df, original_valid_indices = filter_valid_samples(foam_df, column_mapping, config.num_days)

    if valid_df.empty:
        raise ValueError("No valid samples were found for the selected number of days.")

    valid_df_raw = valid_df.copy()

    # Reset index so row-wise operations work with a clean 0-based index
    valid_df = valid_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stage 6: Impute missing foam volumes
    # ------------------------------------------------------------------
    imputed_foam_df = fill_missing_foam_rowwise(valid_df, column_mapping, config)

    # Compute average best-model R²
    avg_best_r2 = _compute_avg_best_r2(imputed_foam_df)

    # ------------------------------------------------------------------
    # Stage 7: Impute missing textures
    # ------------------------------------------------------------------
    imputed_texture_df = fill_missing_textures(
        imputed_foam_df, texture_weights_df, column_mapping, config
    )

    # ------------------------------------------------------------------
    # Stage 8: Rename columns to Day-d format
    # ------------------------------------------------------------------
    renamed_df = rename_to_day_format(imputed_texture_df, column_mapping)

    # ------------------------------------------------------------------
    # Stage 9: Compute performance scores
    # ------------------------------------------------------------------
    scored_df, missing_texture_labels = compute_performance(
        renamed_df, texture_weights_df, config
    )

    # ------------------------------------------------------------------
    # Stage 10: Merge scores onto the full original DataFrame
    # ------------------------------------------------------------------
    full_output_df = merge_scores_to_full(original_df, scored_df, original_valid_indices)

    return PipelineResult(
        original_df=original_df,
        valid_df_raw=valid_df_raw,
        imputed_foam_df=imputed_foam_df,
        imputed_texture_df=imputed_texture_df,
        renamed_df=renamed_df,
        scored_df=scored_df,
        full_output_df=full_output_df,
        original_valid_indices=original_valid_indices,
        missing_texture_labels=missing_texture_labels,
        column_mapping=column_mapping,
        avg_best_r2=avg_best_r2,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_avg_best_r2(df: pd.DataFrame) -> float | None:
    """Return the mean of the per-row best R² score, or None if unavailable."""
    present_r2_cols = [c for c in R2_COLUMNS if c in df.columns]
    if not present_r2_cols:
        return None
    best_per_row = df[present_r2_cols].max(axis=1)
    mean_val = np.nanmean(best_per_row)
    return float(mean_val) if np.isfinite(mean_val) else None
