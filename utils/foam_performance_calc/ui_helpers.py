"""
ui_helpers.py
-------------
Reusable Streamlit display helpers.

All functions accept data objects (DataFrames, dicts, etc.) and render them
using Streamlit.  No business logic lives here — this module is purely
concerned with presentation.
"""

from __future__ import annotations

from typing import Optional
from unittest import result

import numpy as np
import pandas as pd
import streamlit as st

from .schemas import ColumnMapping, PipelineResult
from .constants import R2_COLUMNS, BEST_MODEL_COLUMN


# ---------------------------------------------------------------------------
# Column mapping display
# ---------------------------------------------------------------------------

def show_column_mapping(column_mapping: ColumnMapping) -> None:
    """
    Render the detected foam and texture column mapping as a compact table.
    """
    with st.expander("🗺️ Detected Column Mapping", expanded=False):
        rows = []
        all_days = sorted(
            set(column_mapping.foam_map) | set(column_mapping.texture_map)
        )
        for day in all_days:
            rows.append(
                {
                    "Day": day,
                    "Foam (cc) column": column_mapping.foam_map.get(day, "—"),
                    "Texture column": column_mapping.texture_map.get(day, "—"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Step previews
# ---------------------------------------------------------------------------

def show_raw_preview(df: pd.DataFrame, label: str = "Raw data preview", n: int = 10) -> None:
    """Render the first *n* rows of a DataFrame inside a labelled expander."""
    with st.expander(f"👁️ {label}", expanded=False):
        st.dataframe(df.head(n), use_container_width=True)


def show_valid_samples_summary(
    original_df: pd.DataFrame,
    valid_indices: list[int],
) -> None:
    """
    Show how many rows passed validation vs. the total.
    """
    total = len(original_df)
    valid = len(valid_indices)
    invalid = total - valid
    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows", total)
    col2.metric("Valid samples", valid)
    col3.metric("Filtered out", invalid)


def show_imputation_summary(imputed_df: pd.DataFrame) -> None:
    """
    Show the average best-model R² and model distribution after foam imputation.
    """
    present_r2_cols = [c for c in R2_COLUMNS if c in imputed_df.columns]
    if not present_r2_cols:
        return

    best_per_row = imputed_df[present_r2_cols].max(axis=1)
    avg_r2 = np.nanmean(best_per_row)
    if np.isfinite(avg_r2):
        st.success(f"Average best-model R² across valid samples: **{avg_r2:.3f}**")

    if BEST_MODEL_COLUMN in imputed_df.columns:
        with st.expander("📊 Model selection distribution", expanded=False):
            counts = imputed_df[BEST_MODEL_COLUMN].value_counts().reset_index()
            counts.columns = ["Model", "Count"]
            st.dataframe(counts, hide_index=True)


# ---------------------------------------------------------------------------
# Final results
# ---------------------------------------------------------------------------

def show_scored_results(scored_df: pd.DataFrame) -> None:
    """Render the final scored DataFrame."""
    st.dataframe(scored_df, use_container_width=True)


def show_missing_textures_warning(missing: list[str]) -> None:
    """Warn the user about texture labels not found in the weight file."""
    if missing:
        st.warning(
            f"⚠️ {len(missing)} texture label(s) were not found in the texture "
            f"weights file and could not contribute to Score_Texture:"
        )
        st.write(missing)
    else:
        st.success("✅ All texture labels matched the weight file successfully!")


# ---------------------------------------------------------------------------
# Download button
# ---------------------------------------------------------------------------

def show_download_button(df: pd.DataFrame):
    """
    Converts the DataFrame to CSV and renders the Streamlit download button.
    """
    # Use 'utf-8-sig' to ensure Excel opens the CSV without 'Â' characters
    csv_data = df.to_csv(index=False, encoding="utf-8-sig")
    
    st.download_button(
        label="📥 Download Full Results (CSV)",
        data=csv_data,
        file_name="foam_performance_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Full pipeline result renderer
# ---------------------------------------------------------------------------

def render_pipeline_results(result: PipelineResult) -> None:
    """
    Render all pipeline stages in the Streamlit page after a successful run.

    This is the single call the page makes after ``run_pipeline`` succeeds.
    """
    # ---- Step 1 ----
    st.subheader("Step 1 – Filter Valid Samples")
    show_raw_preview(result.original_df, label="Raw input data (first 10 rows)")
    show_column_mapping(result.column_mapping)
    show_valid_samples_summary(result.original_df, result.original_valid_indices)
    # show_raw_preview(
    #     result.valid_df_raw,
    #     label="Valid samples (raw, before imputation)",
    # )

    # ---- Step 2 ----
    st.subheader("Step 2 – Impute Missing Foam Volumes")
    show_imputation_summary(result.imputed_foam_df)
    show_raw_preview(result.imputed_foam_df, label="Preview after foam imputation")

    # ---- Step 3 ----
    st.subheader("Step 3 – Impute Missing Textures")
    show_raw_preview(result.imputed_texture_df, label="Preview after texture imputation")

    # ---- Step 4 ----
    st.subheader("Step 4 – Rename Columns to Day-Based Format")
    show_raw_preview(result.renamed_df, label="Preview after renaming")

    # ---- Step 5 ----
    st.subheader("Step 5 – Compute Performance Index")
    st.subheader("📊 Final Results with Performance Index")
    show_scored_results(result.scored_df)
    show_missing_textures_warning(result.missing_texture_labels)
 

    st.divider()
    #csv_data = result.full_output_df.to_csv(index=False, encoding="utf-8-sig")
    show_download_button(result.full_output_df)
