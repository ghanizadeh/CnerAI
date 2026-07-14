"""
column_mapping.py
-----------------
Detects and maps foam-volume and texture columns in a DataFrame to a
sequential day-index (0, 1, 2, …).

Background
~~~~~~~~~~
The raw CSV can arrive in several naming formats because it was originally
exported from a tool that stored repeated measurements as duplicate column
names, which pandas then disambiguates with a ``.N`` suffix:

  Raw CSV headers (before deduplication):
      Foam (cc)   Texture   Foam (cc)   Texture   Foam (cc)   Texture …

  After ``io_utils.deduplicate_columns``:
      Foam (cc)   Texture   Foam (cc).1  Texture.1  Foam (cc).2  Texture.2 …

  Desired day mapping:
      Day 0 → Foam (cc)      Day 0 → Texture
      Day 1 → Foam (cc).1    Day 1 → Texture.1
      Day 2 → Foam (cc).2    Day 2 → Texture.2
      …

Key insight: the ``.N`` suffix is a **pandas deduplication artefact**, NOT
a day number.  The correct day assignment is simply the sequential order of
all matching columns (bare first, then .1, .2, …).

Additionally, future datasets might already use 'Day 0 - Foam (cc)' style
headers, which is also handled.
"""

from __future__ import annotations

import re

import pandas as pd

from .constants import FOAM_CC_KEYWORD, TEXTURE_KEYWORD, SCORE_KEYWORD
from .schemas import ColumnMapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_day_columns(df: pd.DataFrame) -> ColumnMapping:
    """
    Detect foam-volume and texture columns and assign them sequential day indices.

    Supports two header formats:
      1. Legacy / deduplicated:
           ``Foam (cc)``, ``Foam (cc).1``, ``Foam (cc).2``, …
           ``Texture``,   ``Texture.1``,   ``Texture.2``,   …
      2. Pre-labelled day format:
           ``Day 0 - Foam (cc)``, ``Day 1 - Foam (cc)``, …

    When both formats are mixed the function handles them correctly by sorting
    all matched columns into their natural sequential order.

    Columns whose normalised name contains ``score`` are explicitly excluded
    to avoid confusing ``Score_Texture (all days)`` with a texture measurement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with deduplicated column names.

    Returns
    -------
    ColumnMapping
        Dataclass containing ``foam_map`` and ``texture_map``
        (both ``{day_int: column_name}``).
    """
    foam_entries: list[tuple[int, str]] = []   # (sort_key, column_name)
    texture_entries: list[tuple[int, str]] = []

    for col in df.columns:
        normalised = col.lower().strip().replace(" ", "")

        # ------------------------------------------------------------------
        # Skip score/helper columns that happen to contain keywords
        # ------------------------------------------------------------------
        if SCORE_KEYWORD in normalised:
            continue

        # ------------------------------------------------------------------
        # Pre-labelled format: "Day N - Foam (cc)" / "Day N - Foam Texture"
        # ------------------------------------------------------------------
        day_prefix_match = re.match(r"day(\d+)-", normalised)
        if day_prefix_match:
            day = int(day_prefix_match.group(1))
            if FOAM_CC_KEYWORD in normalised:
                foam_entries.append((day * 10, col))   # *10 keeps ordering stable
            elif TEXTURE_KEYWORD in normalised:
                texture_entries.append((day * 10, col))
            continue

        # ------------------------------------------------------------------
        # Legacy deduplicated format: bare or ".<suffix>"
        # ------------------------------------------------------------------
        if FOAM_CC_KEYWORD in normalised:
            sort_key = _dedup_sort_key(normalised)
            foam_entries.append((sort_key, col))

        elif TEXTURE_KEYWORD in normalised:
            sort_key = _dedup_sort_key(normalised)
            texture_entries.append((sort_key, col))

    # Sort and assign sequential day numbers
    foam_entries.sort(key=lambda x: x[0])
    texture_entries.sort(key=lambda x: x[0])

    foam_map = {day: col for day, (_, col) in enumerate(foam_entries)}
    texture_map = {day: col for day, (_, col) in enumerate(texture_entries)}

    return ColumnMapping(foam_map=foam_map, texture_map=texture_map)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dedup_sort_key(normalised_col: str) -> int:
    """
    Return a stable integer sort key for a deduplicated column name.

    - Bare column (no dot suffix)  → 0
    - ``<name>.1``                 → 1
    - ``<name>.2``                 → 2
    - …

    This preserves the original column order so day assignment is correct.
    """
    if "." in normalised_col:
        try:
            return int(normalised_col.rsplit(".", 1)[-1]) + 1
        except ValueError:
            return 9999   # push unrecognised suffixes to the end
    return 0
