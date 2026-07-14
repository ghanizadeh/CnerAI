"""
pages_content/page_train.py
============================
Model Training — Model Selection + Validation Method + Training Output.

Train/Test split has been REMOVED from Preprocessing and lives here.
Users choose from 5 validation strategies before training.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from state.session import init_state, get_value, set_state
from components.model_picker import render_model_picker
from core.models.registry import get_model_instance
from core.models.trainer import train
from core.models.evaluator import (
    get_classification_metrics,
    get_regression_metrics,
    get_feature_importance,
)
from core.models.validation import (
    METHOD_TRAIN_TEST,
    METHOD_KFOLD,
    METHOD_STRATIFIED,
    METHOD_LOOCV,
    METHOD_LOGO,
    CLASSIFICATION_METHODS,
    REGRESSION_METHODS,
    get_cv_strategy,
    split_for_training,
    run_validation,
)
from components.metrics_card import render_metrics_card

# Sentinels shown in the LOGO group-column dropdown for the auto-built group
# IDs (state/session.py: data.formulation_id / oil_identity_id /
# brine_identity_id) — each tests generalization along a different axis:
# formulation (chemistry), a genuinely new oil, or a genuinely new brine.
FORMULATION_ID_OPTION = "🧪 Formulation ID (auto — chemical columns from Preprocessing)"
OIL_IDENTITY_OPTION   = "🛢️ Oil Identity (auto — oil property columns from Preprocessing)"
BRINE_IDENTITY_OPTION = "🌊 Brine Identity (auto — brine property columns from Preprocessing)"
_AUTO_GROUP_OPTIONS = {
    FORMULATION_ID_OPTION: "data.formulation_id",
    OIL_IDENTITY_OPTION:   "data.oil_identity_id",
    BRINE_IDENTITY_OPTION: "data.brine_identity_id",
}


# ---------------------------------------------------------------------------
# Validation UI
# ---------------------------------------------------------------------------

def _render_validation_section(task_type: str) -> tuple:
    """
    Render the Validation Method section.
    Returns (method: str, config: dict).
    """
    st.subheader("🔬 Validation Method")

    available = (
        CLASSIFICATION_METHODS if task_type == "classification"
        else REGRESSION_METHODS
    )

    method = st.selectbox(
        "Select validation strategy",
        available,
        index=0,
        key="validation_method",
        help=(
            "**Train / Test Split** – classic hold-out.  \n"
            "**K-Fold CV** – rotate the test window k times.  \n"
            "**Stratified K-Fold** – K-Fold preserving class proportions (classification only).  \n"
            "**LOOCV** – leave one sample out each time; expensive on large data.  \n"
            "**Leave-One-Group-Out** – hold out one group at a time; requires a group column."
        ),
    )

    config = {}

    if method == METHOD_TRAIN_TEST:
        with st.expander("Split options", expanded=True):
            config["test_size"] = st.slider(
                "Test set size", 0.10, 0.40, 0.20, 0.05, key="val_test_size"
            )
            config["random_state"] = st.number_input(
                "Random state", value=42, step=1, key="val_rs"
            )
            config["shuffle"] = st.checkbox("Shuffle before split", value=True, key="val_shuffle")
            if task_type == "classification":
                config["stratify"] = st.checkbox(
                    "Stratify (preserve class proportions)", value=False, key="val_stratify"
                )
            else:
                config["stratify"] = False

    elif method in (METHOD_KFOLD, METHOD_STRATIFIED):
        with st.expander("K-Fold options", expanded=True):
            config["n_splits"] = st.selectbox(
                "Number of folds (k)", [3, 5, 10], index=1, key="val_k"
            )
            config["shuffle"] = st.checkbox("Shuffle", value=True, key="val_kf_shuffle")
            config["random_state"] = st.number_input(
                "Random state", value=42, step=1, key="val_kf_rs"
            )

    elif method == METHOD_LOOCV:
        df_raw = get_value("data.raw")
        n_rows = len(df_raw) if df_raw is not None else 0
        if n_rows > 500:
            st.warning(
                f"LOOCV will run **{n_rows:,} iterations** — this can be very slow. "
                "Consider K-Fold instead."
            )
        else:
            st.info("LOOCV runs one iteration per sample — no parameters needed.")

    elif method == METHOD_LOGO:
        df_raw = get_value("data.raw")
        auto_options = [opt for opt, key in _AUTO_GROUP_OPTIONS.items()
                        if get_value(key) is not None]

        options = ["(none)"] + auto_options
        if df_raw is not None:
            options += df_raw.columns.tolist()

        if len(options) == 1:
            st.warning("Load data first to select a group column.")
            config["group_col"] = None
        else:
            default_idx = options.index(FORMULATION_ID_OPTION) if FORMULATION_ID_OPTION in options else 0
            group_col = st.selectbox(
                "Group column", options, index=default_idx, key="val_group_col",
                help=(
                    "Rows sharing the same chemical formulation (e.g. same "
                    "chemistry re-measured at a different dilution ratio, "
                    "temperature, or brine) should stay in the same fold — "
                    "otherwise CV leaks the answer across train/test. "
                    f"'{FORMULATION_ID_OPTION}' does this automatically using "
                    "the chemical columns selected in Preprocessing.\n\n"
                    f"'{OIL_IDENTITY_OPTION}' / '{BRINE_IDENTITY_OPTION}' test "
                    "generalization to an oil/brine never seen in training, "
                    "not just a new formulation of a familiar one."
                ),
            )
            config["group_col"] = None if group_col == "(none)" else group_col
            if config["group_col"] is None:
                st.warning("A group column is required for Leave-One-Group-Out CV.")
            elif group_col not in auto_options and auto_options:
                st.info(
                    f"Using raw column **{group_col}**. One of the auto-built "
                    "group options above (Formulation ID / Oil Identity / "
                    "Brine Identity) usually gives a more honest estimate."
                )

    return method, config


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    init_state()

    st.title("🤖 Model Training")
    st.divider()

    X = get_value("data.X")
    y = get_value("data.y")
    task_type = get_value("model.task_type")

    if X is None or y is None:
        st.warning("No preprocessed data found. Go to **Preprocessing** first.")
        st.stop()

    if task_type is None:
        st.warning("Task type not set. Re-run **Preprocessing** and click 'Apply'.")
        st.stop()

    st.info(
        f"**Task:** {task_type.capitalize()} &nbsp;|&nbsp; "
        f"**Samples:** {len(X):,} &nbsp;|&nbsp; **Features:** {X.shape[1]}"
    )

    # 1. Model Selection
    st.subheader("🤖 Model Selection")
    model_name, params = render_model_picker(task_type, key_prefix="train")

    st.divider()

    # 2. Validation Method
    method, val_config = _render_validation_section(task_type)

    st.divider()

    # 3. Train Button
    if not st.button("🚀 Train Model", type="primary", use_container_width=True):
        st.stop()

    scoring = "f1_weighted" if task_type == "classification" else "r2"

    # Resolve groups for LOGO
    groups = None
    if method == METHOD_LOGO:
        group_col = val_config.get("group_col")
        if group_col is None:
            st.error("Select a group column to use Leave-One-Group-Out CV.")
            st.stop()
        try:
            if group_col in _AUTO_GROUP_OPTIONS:
                group_series = get_value(_AUTO_GROUP_OPTIONS[group_col])
                if group_series is not None:
                    groups = group_series.loc[X.index]
            else:
                df_raw = get_value("data.raw")
                if df_raw is not None and group_col in df_raw.columns:
                    groups = df_raw.loc[X.index, group_col]
        except KeyError:
            groups = None
        if groups is None:
            st.error(
                "Could not resolve the selected group column against the "
                "current data. Re-run Preprocessing or pick a different column."
            )
            st.stop()

    with st.spinner(f"Training **{model_name}** with **{method}**…"):
        t0 = time.time()

        if method == METHOD_TRAIN_TEST:
            X_train, X_test, y_train, y_test = split_for_training(
                X, y,
                test_size=val_config.get("test_size", 0.2),
                random_state=int(val_config.get("random_state", 42)),
                shuffle=bool(val_config.get("shuffle", True)),
                stratify=bool(val_config.get("stratify", False)),
            )
            model = get_model_instance(task_type, model_name, params)
            model = train(model, X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics = (
                get_classification_metrics(y_test, y_pred, y_prob)
                if task_type == "classification"
                else get_regression_metrics(y_test, y_pred)
            )
            val_results = None

        else:
            model_for_cv = get_model_instance(task_type, model_name, params)
            val_results = run_validation(
                method=method,
                config=val_config,
                model=model_for_cv,
                X=X,
                y=y,
                scoring=scoring,
                task_type=task_type,
                groups=groups,
            )
            # Refit on ALL data — this is the model saved for prediction/explainability
            model = get_model_instance(task_type, model_name, params)
            model = train(model, X, y)
            # CV already gave us unbiased fold scores — use those as the primary metrics.
            # Do NOT evaluate the refitted model on a slice of its own training data.
            metrics = {
                scoring: round(val_results["mean"], 4),
                f"{scoring}_std": round(val_results["std"], 4),
            }
            # Keep a split available so Evaluate / Explainability pages have
            # X_test for SHAP, residual plots, etc. These rows are NOT used
            # to compute the metrics above.
            X_train, X_test, y_train, y_test = split_for_training(X, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        elapsed = time.time() - t0

    st.success(f"✅ **{model_name}** trained in {elapsed:.2f}s")

    # 4. Training Output
    st.subheader("📈 Training Output")

    if val_results is not None:
        st.markdown(f"#### {method} — `{scoring}`")
        scores_df = pd.DataFrame({
            "Fold": [f"#{i+1}" for i in range(len(val_results["scores"]))],
            scoring: [round(s, 4) for s in val_results["scores"]],
        })
        st.dataframe(scores_df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Mean", f"{val_results['mean']:.4f}")
        c2.metric("Std", f"±{val_results['std']:.4f}")
        st.divider()

    label = "Test-Set Metrics" if method == METHOD_TRAIN_TEST else f"CV Summary ({method})"
    st.markdown(f"#### {label} Metrics")
    render_metrics_card(metrics, task_type=task_type)

    feature_names = (
        get_value("data.processed_feature_names")
        or get_value("data.feature_names")
        or list(X.columns)
    )
    importance_df = get_feature_importance(model, feature_names)
    if importance_df is not None:
        st.markdown("#### 🏆 Feature Importances (Top 15)")
        from core.viz.evaluation import draw_feature_importance
        from core.viz.style import fig_to_st
        fig = draw_feature_importance(importance_df, top_n=15)
        fig_to_st(fig)

    # 5. Persist
    set_state("model.name",          model_name)
    set_state("model.object",        model)
    set_state("model.params",        params)
    set_state("results.metrics",     metrics)
    set_state("results.predictions", y_pred)
    set_state("split.X_train",  X_train)
    set_state("split.X_test",   X_test)
    set_state("split.y_train",  y_train)
    set_state("split.y_test",   y_test)
    set_state("split.train_size", 1 - val_config.get("test_size", 0.2))

    trained_models = get_value("results.trained_models") or []
    trained_models.append({
        "name":        model_name,
        "params":      params,
        "validation":  method,
        "metrics":     {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "val_results": val_results,
    })
    set_state("results.trained_models", trained_models)

    st.info("👉 Proceed to **📈 Evaluate**, **🧠 Explainability**, or **🟢 Safe Region & Optimizer**.")
