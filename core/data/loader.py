"""
core/data/loader.py
Pure Python data-loading utilities — no Streamlit here.

Public API
----------
load_csv(file_obj)          -> pd.DataFrame
load_excel(file_obj, sheet) -> pd.DataFrame
list_excel_sheets(file_obj) -> list[str]
validate_df(df)             -> list[str]   # list of warning strings
extended_describe(df)       -> pd.DataFrame
"""

from __future__ import annotations
import pandas as pd
from config.settings import MAX_UPLOAD_MB, MAX_ROWS_PREVIEW


# ── Loaders ───────────────────────────────────────────────────────────

def load_csv(file_obj) -> pd.DataFrame:
    """
    Read a CSV file object into a DataFrame.

    Tries UTF-8 first, then falls back through common encodings used by
    Excel and other tools (Latin-1 covers Windows-1252 as a superset).
    This prevents the 'UnicodeDecodeError — try saving as UTF-8' crash.
    """
    _ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    raw_bytes = file_obj.read()          # read once into memory

    for enc in _ENCODINGS:
        try:
            import io
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue

    # Last resort: replace undecodable bytes rather than crashing
    import io
    return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1", errors="replace")


def list_excel_sheets(file_obj) -> list[str]:
    """Return sheet names without loading data."""
    file_obj.seek(0)
    return pd.ExcelFile(file_obj).sheet_names


def load_excel(file_obj, sheet_name: str) -> pd.DataFrame:
    """Read one sheet from an Excel file object."""
    file_obj.seek(0)
    return pd.read_excel(file_obj, sheet_name=sheet_name)


# ── Validation ────────────────────────────────────────────────────────

def validate_df(df: pd.DataFrame) -> list[str]:
    """
    Run basic sanity checks on a freshly loaded DataFrame.

    Returns a list of human-readable warning strings (empty = all good).
    """
    warnings: list[str] = []

    if df.empty:
        warnings.append("Dataset is empty.")
        return warnings  # nothing else to check

    # Duplicate columns
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        warnings.append(f"Duplicate column names detected: {dupes}")

    # High missing-value columns (> 50 %)
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.5].index.tolist()
    if high_missing:
        warnings.append(
            f"Columns with >50 % missing values: {high_missing}"
        )

    # Constant columns (zero variance)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        warnings.append(f"Constant (zero-variance) columns: {const_cols}")

    # Very large dataset notice
    if len(df) > MAX_ROWS_PREVIEW:
        warnings.append(
            f"Large dataset ({len(df):,} rows). "
            f"Preview limited to {MAX_ROWS_PREVIEW:,} rows."
        )

    return warnings


# ── Summary ───────────────────────────────────────────────────────────

def extended_describe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a combined summary table: custom counts + pandas describe.

    Works for both numeric and categorical columns.
    """
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