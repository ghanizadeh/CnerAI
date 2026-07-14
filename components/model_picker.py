"""
components/model_picker.py
Reusable Streamlit widget: model + hyperparameter picker.

Usage
-----
from components.model_picker import render_model_picker

model_name, params = render_model_picker(task_type="classification")
"""

from __future__ import annotations
import streamlit as st
from core.models.registry import get_model_names, get_default_params


def render_model_picker(
    task_type: str,
    key_prefix: str = "model_picker",
) -> tuple[str, dict]:
    """
    Render a selectbox for model choice + editable default params.

    Parameters
    ----------
    task_type  : "classification" or "regression"
    key_prefix : Unique prefix to avoid widget key collisions.

    Returns
    -------
    (model_name, params_dict)
    """
    model_names = get_model_names(task_type)
    if not model_names:
        st.error(f"No models found for task type: {task_type}")
        return "", {}

    # ── Model selectbox ────────────────────────────────────────────────
    model_name: str = st.selectbox(
        "🤖 Select model",
        options=model_names,
        key=f"{key_prefix}_name",
    )

    # ── Hyperparameter editor ──────────────────────────────────────────
    defaults = get_default_params(task_type, model_name)
    params: dict = {}

    if defaults:
        with st.expander("⚙️ Hyperparameters", expanded=False):
            st.caption("Edit values below — changes apply on the next training run.")
            for param, default_val in defaults.items():
                widget_key = f"{key_prefix}_{model_name}_{param}"
                if isinstance(default_val, bool):
                    params[param] = st.checkbox(param, value=default_val, key=widget_key)
                elif isinstance(default_val, int):
                    params[param] = st.number_input(
                        param, value=default_val, step=1, key=widget_key
                    )
                elif isinstance(default_val, float):
                    params[param] = st.number_input(
                        param, value=default_val, step=0.01, format="%.4f", key=widget_key
                    )
                elif isinstance(default_val, str):
                    params[param] = st.text_input(param, value=default_val, key=widget_key)
                else:
                    # fallback: show as read-only info
                    st.text(f"{param}: {default_val}")
                    params[param] = default_val
    else:
        st.caption("ℹ️ This model has no configurable hyperparameters.")

    return model_name, params
