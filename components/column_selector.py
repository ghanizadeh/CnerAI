"""
components/column_selector.py
Reusable Streamlit widget: feature & target column selector.

Usage
-----
from components.column_selector import render_column_selector

features, target = render_column_selector(df)

# With pre-selected values (e.g. restored from session state):
features, target = render_column_selector(
    df,
    default_features=st.session_state.ml["data"]["feature_names"],
    default_target=st.session_state.ml["data"]["target_name"],
)
"""

from __future__ import annotations
import streamlit as st
import pandas as pd


def render_column_selector(
    df: pd.DataFrame,
    default_features: list[str] | None = None,
    default_target: str | None = None,
    key_prefix: str = "col_sel",
) -> tuple[list[str], str | None]:
    """
    Render feature multiselect + target selectbox.

    Parameters
    ----------
    df              : DataFrame whose columns are shown as options.
    default_features: Pre-selected feature columns (restored from state).
    default_target  : Pre-selected target column.
    key_prefix      : Unique prefix to avoid widget key collisions when
                      this component is used on multiple pages.

    Returns
    -------
    (selected_features, selected_target)
    selected_target is None when the user has not yet chosen.
    """
    cols = df.columns.tolist()
    feat_key = f"{key_prefix}_features"
    tgt_key  = f"{key_prefix}_target"

    # ── Seed features widget ──────────────────────────────────────────
    # Priority: existing valid selection for this prefix > ml-state default > empty.
    # Re-seed from ml state whenever current widget value is empty/stale,
    # so that navigating between pages inherits saved selections.
    current_feat_val = st.session_state.get(feat_key, [])
    valid_current    = [c for c in current_feat_val if c in cols]

    if valid_current:
        # Widget already has a live, valid value for this page — preserve it
        seed_features = valid_current
    else:
        # Nothing yet for this prefix → seed from ml state / caller default
        seed_features = [c for c in (default_features or []) if c in cols]

    st.session_state[feat_key] = seed_features

    # ── Feature multiselect ────────────────────────────────────────────
    selected_features: list[str] = st.multiselect(
        "🟢 Feature columns",
        options=cols,
        key=feat_key,
        help="Select one or more columns to use as model inputs.",
    )

    # ── Target selectbox ───────────────────────────────────────────────
    remaining = [c for c in cols if c not in selected_features]

    if not remaining:
        st.warning("⚠️ All columns are selected as features — no column left for target.")
        return selected_features, None

    # Seed target: existing valid value > ml-state default > first remaining
    current_tgt_val = st.session_state.get(tgt_key)
    if current_tgt_val in remaining:
        seed_target = current_tgt_val
    elif default_target in remaining:
        seed_target = default_target
    else:
        seed_target = remaining[0]

    st.session_state[tgt_key] = seed_target

    selected_target: str = st.selectbox(
        "🟠 Target column",
        options=remaining,
        key=tgt_key,
        help="The column your model will learn to predict.",
    )

    # ── Inline feedback ───────────────────────────────────────────────
    if selected_features and selected_target:
        target_dtype = df[selected_target].dtype
        n_unique = df[selected_target].nunique()
        if pd.api.types.is_numeric_dtype(target_dtype) and n_unique > 5:
            task_hint = "🔢 **Regression** task detected (continuous target)"
        else:
            task_hint = f"🏷️ **Classification** task detected ({n_unique} unique classes)"
        st.caption(task_hint)

    return selected_features, selected_target
