"""
schemas.py
----------
Typed data containers (dataclasses) used as structured hand-offs between
pipeline stages.  Keeping mappings and config in typed objects instead of
loose dicts makes the code self-documenting and IDE-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .constants import (
    DEFAULT_NUM_DAYS,
    DEFAULT_MODEL,
    DEFAULT_R2_THRESHOLD,
    DEFAULT_DAY_WEIGHT,
    MODEL_CHOICES,
)


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

@dataclass
class ColumnMapping:
    """
    Maps integer day numbers to their corresponding raw DataFrame column names.

    Attributes
    ----------
    foam_map     : {day: column_name} for Foam (cc) columns.
    texture_map  : {day: column_name} for Texture columns.
    """
    foam_map: dict[int, str] = field(default_factory=dict)
    texture_map: dict[int, str] = field(default_factory=dict)

    def foam_columns_ordered(self) -> list[str]:
        """Return foam column names sorted by day."""
        return [self.foam_map[d] for d in sorted(self.foam_map)]

    def texture_columns_ordered(self) -> list[str]:
        """Return texture column names sorted by day."""
        return [self.texture_map[d] for d in sorted(self.texture_map)]


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    User-supplied configuration for the full calculation pipeline.

    Attributes
    ----------
    num_days       : Last day index (inclusive).  Days 0 … num_days are used.
    selected_model : Imputation model key ('best', 'linear', 'exp', 'poly', 'rf').
    r2_threshold   : Minimum R² required to apply imputation for a sample row.
    day_weights    : Per-day weight multipliers used in scoring.
    """
    num_days: int = DEFAULT_NUM_DAYS
    selected_model: str = DEFAULT_MODEL
    r2_threshold: float = DEFAULT_R2_THRESHOLD
    day_weights: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.selected_model not in MODEL_CHOICES:
            raise ValueError(
                f"selected_model must be one of {MODEL_CHOICES}, "
                f"got '{self.selected_model}'."
            )
        # Fill any missing day weights with the default
        for d in range(self.num_days + 1):
            self.day_weights.setdefault(d, DEFAULT_DAY_WEIGHT)

    def weight_for(self, day: int) -> float:
        """Return the weight for *day*, falling back to the default."""
        return self.day_weights.get(day, DEFAULT_DAY_WEIGHT)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Structured output returned by the orchestration layer.

    Attributes
    ----------
    original_df          : The raw input DataFrame (all rows, after header normalisation).
    valid_df_raw         : Valid samples before any imputation (reset index).
    imputed_foam_df      : Valid samples after foam volume imputation.
    imputed_texture_df   : Valid samples after texture imputation.
    renamed_df           : Day-labelled columns ('Day d - Foam (cc)' etc.).
    scored_df            : Renamed df with Score_Volume, Score_Texture, Performance_Index.
    full_output_df       : original_df with score columns merged in (NaN for invalid rows).
    original_valid_indices : Original row indices (in original_df) that were valid.
    missing_texture_labels : Texture labels not found in the weight file.
    column_mapping       : Detected foam / texture column mapping.
    avg_best_r2          : Mean best-model R² across valid samples (None if unavailable).
    """
    original_df: pd.DataFrame
    valid_df_raw: pd.DataFrame
    imputed_foam_df: pd.DataFrame
    imputed_texture_df: pd.DataFrame
    renamed_df: pd.DataFrame
    scored_df: pd.DataFrame
    full_output_df: pd.DataFrame
    original_valid_indices: list[int]
    missing_texture_labels: list[str]
    column_mapping: ColumnMapping
    avg_best_r2: Optional[float] = None
