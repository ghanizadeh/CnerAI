"""
io_utils.py
-----------
Handles all file I/O concerns:
  - Reading CSV files uploaded via Streamlit (or from a file path).
  - Making duplicate column names unique (pandas deduplication suffix style).
  - Safe file parsing with graceful error handling.

Nothing in this module knows about foam business logic.
"""

from __future__ import annotations

from typing import Union
import io
import pandas as pd

def load_table(file, sheet_name=None):

    # -------------------------
    # Read file
    # -------------------------
    if file.name.endswith(".csv"):

        try:
            df = pd.read_csv(file, encoding="utf-8")

        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="cp1252")

    elif file.name.endswith(".xlsx"):

        df = pd.read_excel(
            file,
            sheet_name=sheet_name,
            engine="openpyxl"
        )

    else:
        raise ValueError("Unsupported file format")

    # -------------------------
    # Remove Excel hidden chars
    # -------------------------
    df = df.replace(r"\xa0", "", regex=True)
    df = df.replace(r"Â", "", regex=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from text columns
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()

    # Empty string -> NaN
    df = df.replace("", pd.NA)

    return df

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data_safe(source: Union[str, io.IOBase], sheet_name: str = None) -> pd.DataFrame:
    """
    Read CSV or Excel safely, handling encoding and cleaning hidden characters.
    """
    if source is None:
        return pd.DataFrame()

    # 1. Read the file based on extension
    file_name = getattr(source, "name", "")
    
    try:
        if file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")
        else:
            # Try UTF-8, fallback to cp1252 for Excel-style CSVs
            try:
                if hasattr(source, "seek"): source.seek(0)
                df = pd.read_csv(source, encoding="utf-8")
            except (UnicodeDecodeError, pd.errors.ParserError):
                if hasattr(source, "seek"): source.seek(0)
                df = pd.read_csv(source, encoding="cp1252")
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    # 2. Clean 'Â' and Non-breaking spaces (\xa0)
    # We replace them with a standard space or empty string to prevent encoding errors
    import re

    def clean_text(text):
        if isinstance(text, str):

            text = (
                text.replace("\xa0", " ")
                    .replace("Â", "")
                    .replace("*", "")
            )

            # Remove extra spaces
            text = re.sub(r"\s+", " ", text)

            return text.strip()

        return text

    # Clean Column Headers (Crucial for the "Â" issue)
    df.columns = [clean_text(str(col)) for col in df.columns]

    # Clean Data Cells
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].apply(clean_text)

    # 3. Final cleaning: Empty strings to NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    
    # 4. Deduplicate Column Names
    df = deduplicate_columns(df)

    return df


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename duplicate column headers so that every column name is unique.

    Pandas normally silently allows duplicate column names, which causes
    unpredictable behaviour when selecting by name.  This function applies
    the same ``<name>.<n>`` suffix convention that pandas uses internally
    for ``read_csv`` when ``mangle_dupe_cols`` is active, but we do it
    explicitly so the mapping is deterministic and inspectable.

    The *first* occurrence of a duplicated name keeps the bare name.
    Subsequent occurrences become ``<name>.1``, ``<name>.2``, etc.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns may contain duplicates.

    Returns
    -------
    pd.DataFrame
        New DataFrame with unique column names (same data, same order).
    """
    new_cols = []
    seen = {}
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    df.columns = new_cols
    return df
