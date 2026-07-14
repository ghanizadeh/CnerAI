"""
Centralised session-state management.

All pages import from here — never touch st.session_state["ml"] directly.

Schema
------
ml
├── data
│   ├── raw           pd.DataFrame | None
│   ├── cleaned       pd.DataFrame | None
│   ├── X             pd.DataFrame | None
│   ├── y             pd.Series    | None
│   ├── feature_names list[str]
│   ├── target_name   str | None
│   ├── file_name     str | None   (uploaded file's name — survives page navigation)
│   ├── sheet_name    str | None   (selected Excel sheet, if any)
│   ├── formulation_id  pd.Series | None
│   │                 (hash of the chemical columns used in Preprocessing —
│   │                  rows with identical chemistry share an ID, indexed like
│   │                  data.X. Lets Train's Leave-One-Group-Out CV group by
│   │                  formulation instead of by row, avoiding leakage when the
│   │                  same chemistry is re-measured at different dilution/
│   │                  temperature/brine.)
│   ├── oil_identity_id    pd.Series | None
│   │                 (hash of the Oil-group columns — rows with the same oil
│   │                  share an ID. Lets Train's LOGO test generalization to an
│   │                  oil never seen in training, not just a new formulation
│   │                  of a familiar one.)
│   └── brine_identity_id  pd.Series | None
│                     (same idea as oil_identity_id, keyed on the Brine-group
│                      columns — Divalent/Monovalent.)
├── preprocessing
│   ├── missing_strategy   str       ("drop" | "mean" | "median")
│   ├── outlier_method     str | None
│   ├── outlier_removed    bool
│   ├── scaler             object | None
│   ├── encoder            object | None
│   ├── rows_before        int | None
│   ├── rows_after         int | None
│   └── applied            bool
├── split
│   ├── method      str   ("train_test" | "cross_val")
│   ├── train_size  float
│   ├── X_train / X_val / X_test   pd.DataFrame | None
│   └── y_train / y_val / y_test   pd.Series    | None
├── model
│   ├── task_type   str | None   ("classification" | "regression")
│   ├── name        str | None
│   ├── object      fitted model | None
│   └── params      dict
└── results
    ├── metrics      dict
    ├── predictions  pd.Series | None
    ├── residuals    pd.Series | None
    └── trained_models  list[dict]   # history for comparison
"""

from __future__ import annotations
from typing import Any
import streamlit as st


_DEFAULT_STATE: dict = {
    "data": {
        "raw": None,
        "cleaned": None,
        "X": None,
        "y": None,
        "feature_names": [],
        "target_name": None,
        "file_name": None,
        "sheet_name": None,
        "formulation_id": None,
        "oil_identity_id": None,
        "brine_identity_id": None,
    },
    "preprocessing": {
        "missing_strategy": "drop",
        "outlier_method": None,
        "outlier_removed": False,
        "scaler": None,
        "encoder": None,
        "rows_before": None,
        "rows_after": None,
        "applied": False,
    },
    "split": {
        "method": "train_test",
        "train_size": 0.8,
        "X_train": None,
        "X_val": None,
        "X_test": None,
        "y_train": None,
        "y_val": None,
        "y_test": None,
    },
    "model": {
        "task_type": None,
        "name": None,
        "object": None,
        "params": {},
    },
    "results": {
        "metrics": {},
        "predictions": None,
        "residuals": None,
        "trained_models": [],
    },
}


def init_state() -> None:
    """Initialise session state on first load (idempotent)."""
    if "ml" not in st.session_state:
        import copy
        st.session_state.ml = copy.deepcopy(_DEFAULT_STATE)


def get_state() -> dict:
    """Return the full ml state dict."""
    init_state()
    return st.session_state.ml


def set_state(key_path: str, value: Any) -> None:
    """
    Set a nested key using dot notation.

    Examples
    --------
    set_state("data.raw", df)
    set_state("model.task_type", "classification")
    """
    init_state()
    keys = key_path.split(".")
    node = st.session_state.ml
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def get_value(key_path: str, default: Any = None) -> Any:
    """
    Get a nested value using dot notation.

    Examples
    --------
    df = get_value("data.raw")
    """
    init_state()
    keys = key_path.split(".")
    node = st.session_state.ml
    try:
        for k in keys:
            node = node[k]
        return node
    except (KeyError, TypeError):
        return default


def clear_state() -> None:
    """Wipe and reinitialise the entire pipeline state."""
    if "ml" in st.session_state:
        del st.session_state["ml"]
    # also clear working_data scratch pad
    if "working_data" in st.session_state:
        del st.session_state["working_data"]
    init_state()


def pipeline_status() -> dict[str, bool]:
    """Convenience: return booleans for dashboard status indicators."""
    return {
        "data_loaded": get_value("data.raw") is not None,
        "preprocessed": get_value("data.X") is not None,
        "model_trained": get_value("model.object") is not None,
    }
