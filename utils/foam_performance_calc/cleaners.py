"""
cleaners.py
-----------
Data-cleaning utilities for foam volume columns and texture label strings.

All functions are pure (no side effects) and return copies of their inputs.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Texture label cleaning
# ---------------------------------------------------------------------------

def clean_texture_label(val: object) -> str | None:
    """
    Normalise a single texture label to a clean, comparable string.

    Transformations applied:
      - Cast to string and lower-case.
      - Strip leading/trailing whitespace and non-breaking spaces (``\\xa0``).
      - Replace any character that is not a letter, digit, or space with a space.
      - Collapse multiple consecutive spaces to a single space.
      - Return ``None`` for empty or ``"nan"`` results.

    Parameters
    ----------
    val : object
        Raw texture cell value (string, float NaN, None, etc.).

    Returns
    -------
    str or None
        Normalised label, or ``None`` if the input was missing/blank.
    """
    if pd.isna(val):
        return None
    text = str(val).lower().strip().replace("\xa0", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text and text != "nan" else None


def clean_texture_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ``clean_texture_label`` to every column whose name contains 'Texture'.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy of *df* with cleaned texture columns.
    """
    df = df.copy()
    for col in df.columns:
        if "Texture" in col:
            df[col] = df[col].apply(clean_texture_label)
    return df


# ---------------------------------------------------------------------------
# Foam volume cleaning
# ---------------------------------------------------------------------------

def clean_foam_column(series: pd.Series) -> pd.Series:
    """
    Coerce a foam-volume column to numeric, handling common Excel artefacts.

    Steps:
      1. Cast to string.
      2. Remove non-breaking spaces (``\\xa0``).
      3. Strip whitespace.
      4. Replace empty strings with ``NaN``.
      5. Convert to float via ``pd.to_numeric`` (unrecognised values → ``NaN``).

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    pd.Series
        Float series (same index as input).
    """
    return (
        series
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
        .replace("", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )


def clean_foam_columns(df: pd.DataFrame, foam_map: dict[int, str]) -> pd.DataFrame:
    """
    Apply ``clean_foam_column`` to every column referenced in *foam_map*.

    Parameters
    ----------
    df       : pd.DataFrame
    foam_map : {day: column_name}

    Returns
    -------
    pd.DataFrame
        Copy of *df* with all foam columns coerced to float.
    """
    df = df.copy()
    for col in foam_map.values():
        if col in df.columns:
            df[col] = clean_foam_column(df[col])
    return df
