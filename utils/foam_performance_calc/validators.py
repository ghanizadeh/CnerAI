"""
validators.py
-------------
Sample-level validity rules and row-filtering logic.

A sample (row) is considered *valid* for a given ``num_days`` if:
  1. At least two foam measurement days exist within the selected range.
  2. The first and last day values are both non-NaN.
  3. No two consecutive days within the range are both NaN.

Filtering returns the valid subset while preserving the original index so
that scores can be merged back onto the full dataset later.
"""

from __future__ import annotations

import pandas as pd

from .schemas import ColumnMapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_valid_sample(
    row: pd.Series,
    column_mapping: ColumnMapping,
    num_days: int,
) -> bool:
    """
    Determine whether a single row passes all validity rules.

    Parameters
    ----------
    row            : pd.Series  – one row of the foam DataFrame.
    column_mapping : ColumnMapping – detected foam/texture column map.
    num_days       : int – last day (inclusive) of the analysis window.

    Returns
    -------
    bool
    """
    foam_map = column_mapping.foam_map
    active_days = [d for d in range(num_days + 1) if d in foam_map]

    # Rule 1 – need at least two measurement days
    if len(active_days) < 2:
        return False

    first_day = active_days[0]
    last_day = active_days[-1]

    # Rule 2 – first and last day values must be present
    if pd.isna(row.get(foam_map[first_day])) or pd.isna(row.get(foam_map[last_day])):
        return False

    # Rule 3 – no two consecutive missing values
    for i in range(len(active_days) - 1):
        d1, d2 = active_days[i], active_days[i + 1]
        if pd.isna(row.get(foam_map[d1])) and pd.isna(row.get(foam_map[d2])):
            return False

    return True


def filter_valid_samples(
    df: pd.DataFrame,
    column_mapping: ColumnMapping,
    num_days: int,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Return only the rows that pass ``is_valid_sample``.

    The original DataFrame index is preserved in the returned subset so that
    scores can later be merged back onto the full dataset using ``.loc``.

    Parameters
    ----------
    df             : pd.DataFrame – full cleaned foam DataFrame.
    column_mapping : ColumnMapping
    num_days       : int

    Returns
    -------
    tuple[pd.DataFrame, list[int]]
        - valid_df : subset of *df* (original index preserved).
        - original_valid_indices : list of original index labels for valid rows.
    """
    mask = df.apply(
        lambda row: is_valid_sample(row, column_mapping, num_days),
        axis=1,
    )
    valid_df = df[mask].copy()
    original_valid_indices = valid_df.index.tolist()
    return valid_df, original_valid_indices
