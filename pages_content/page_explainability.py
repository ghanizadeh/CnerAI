"""
pages_content/page_explainability.py
=====================================
🧠 Explainability — tabs: SHAP | Auto Plots | PDP | Surrogate Rules

Original-scale policy
---------------------
All feature values shown to the user come from X_train_orig (unscaled).
SHAP values are computed on X_train_scaled (model input) but the x-axis
of every dependence / beeswarm plot uses X_train_orig so concentrations
are physically meaningful even when scaling was applied.
"""
from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap as _shap
import streamlit as st

from config.settings import RANDOM_STATE
from core.models.explainability import (
    MAX_SHAP_SAMPLES,
    SURROGATE_DEPTH_RANGE,
    SURROGATE_MAX_DEPTH,
    extract_rules,
    get_shap_values,
    plot_pdp_1d,
    plot_pdp_2d,
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_dependence_2d,
    shap_importance_df,
)
from core.viz.style import fig_to_st
from state.session import get_value, init_state


# ── colour map matching preprocessing groups ──────────────────────────────────
_GRP_CLR = {
    "Nanoparticle": "#1565C0", "Anionic": "#E53935",
    "Nonionic": "#FB8C00",     "Zwitterionic": "#8E24AA",
    "Surfactant": "#FF8F00",   "Polymer": "#6A1B9A",
    "Citric/Buffer": "#2E7D32","Acid/Chelant": "#C62828",
    "Antiscalant": "#00695C",  "Brine": "#4527A0",
    "Oil": "#BF360C",          "Process": "#546E7A",
    "Interaction": "#795548",  "Custom": "#37474F",
}


def _build_X_train_orig(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Return the unscaled version of X_train.

    Strategy (in order of preference):
      1. data.X_original exists → slice matching rows by index
      2. No scaler was saved → X_train is already unscaled → return copy
    """
    X_orig_full = get_value("data.X_original")
    if X_orig_full is None:
        # No scaler applied — X_train is already in original scale
        return X_train.copy()

    # Align to training rows
    try:
        out = X_orig_full.loc[X_train.index].copy()
    except KeyError:
        # Index mismatch fallback — positional slice
        out = X_orig_full.iloc[: len(X_train)].copy()

    # Guarantee column names match (handles any reordering edge case)
    out = out.reindex(columns=X_train.columns)
    return out


def _plot_2d_shap_orig(
    shap_vals: np.ndarray,
    X_orig: pd.DataFrame,
    feature: str,
    color_feature: str | None,
    n_bins: int = 10,
    title_suffix: str = "",
) -> plt.Figure:
    """
    2D SHAP bar+scatter in original scale.
    X_orig must already be unscaled — x-axis shows real concentrations.
    """
    feat_names = list(X_orig.columns)
    if feature not in feat_names:
        raise ValueError(f"Feature '{feature}' not in X_orig")

    feat_idx = feat_names.index(feature)
    x_vals = pd.to_numeric(X_orig[feature], errors="coerce").values
    s_vals = shap_vals[:, feat_idx]

    valid = ~(np.isnan(x_vals) | np.isnan(s_vals))
    x_v, s_v = x_vals[valid], s_vals[valid]

    # Interaction color
    if color_feature and color_feature in feat_names:
        c_series = pd.to_numeric(X_orig[color_feature], errors="coerce")
        if not pd.api.types.is_numeric_dtype(c_series):
            c_series = c_series.astype("category").cat.codes.astype(float)
        c_v = c_series.values[valid]
        c_v = np.where(np.isnan(c_v), 0.0, c_v)
        c_label = color_feature
    else:
        c_v = s_v
        c_label = f"SHAP({feature})"

    # Bin means
    if x_v.max() > x_v.min():
        bins = np.linspace(x_v.min(), x_v.max(), n_bins + 1)
        bin_idx = np.clip(np.digitize(x_v, bins, right=True), 1, n_bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_means = np.array([
            s_v[bin_idx == b].mean() if (bin_idx == b).any() else 0.0
            for b in range(1, n_bins + 1)
        ])
        bar_width = (bins[1] - bins[0]) * 0.7
    else:
        bin_centers, bin_means, bar_width = np.array([x_v.mean()]), np.array([s_v.mean()]), 0.1

    c_min, c_max = float(c_v.min()), float(c_v.max())
    if c_min == c_max:
        c_max = c_min + 1.0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(bin_centers, bin_means, width=bar_width,
           color="lightgrey", edgecolor="white", zorder=1, label="Mean SHAP / bin")
    sc = ax.scatter(x_v, s_v, c=c_v, cmap="coolwarm", s=30, alpha=0.8,
                    zorder=2, vmin=c_min, vmax=c_max)
    ax.axhline(0, color="grey", lw=0.9, ls="--", zorder=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(c_label, rotation=270, labelpad=14)
    mid = (c_min + c_max) / 2
    cbar.set_ticks([c_min, mid, c_max])
    cbar.set_ticklabels([f"{c_min:.3g}", f"{mid:.3g}", f"{c_max:.3g}"])
    ax.set_xlabel(f"{feature}  (original scale)")
    ax.set_ylabel(f"SHAP value for\n{feature}")
    ax.set_title(f"SHAP Dependence — {feature}{title_suffix}")
    fig.tight_layout()
    return fig


def _auto_plots_tab(
    shap_vals: np.ndarray,
    X_orig: pd.DataFrame,
    feature_names: list[str],
    feat_to_group: dict,
    task_type: str,
    class_names: list[str],
    class_idx: int,
    n_top: int = 10,
    n_bins: int = 10,
):
    """
    Auto-generate 2D SHAP dependence plots for top-N features.
    - Regression : n_top plots, color = most correlated other feature
    - Classification : n_top plots per class (5 each by default)
    """
    st.markdown(
        "Automatically plots 2D SHAP dependence for the **top features by mean |SHAP|**. "
        "Grey bars = mean SHAP per feature-value bin. "
        "Dots coloured by the most correlated other feature. "
        "**X-axis is always in original (unscaled) units.**"
    )

    sv_df = pd.DataFrame(shap_vals, columns=feature_names)
    mean_abs = sv_df.abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.head(n_top).index.tolist()

    def _best_color(feat: str) -> str:
        """Pick the other top feature most correlated with SHAP values of feat."""
        feat_idx = feature_names.index(feat)
        s = shap_vals[:, feat_idx]
        best_r, best_f = 0.0, None
        for other in top_feats:
            if other == feat:
                continue
            x_other = pd.to_numeric(X_orig[other], errors="coerce").values
            valid = ~(np.isnan(x_other) | np.isnan(s))
            if valid.sum() < 5:
                continue
            r = abs(np.corrcoef(x_other[valid], s[valid])[0, 1])
            if not np.isnan(r) and r > best_r:
                best_r, best_f = r, other
        return best_f or (top_feats[1] if len(top_feats) > 1 else top_feats[0])

    if task_type == "regression":
        st.markdown(f"### Top {n_top} features — Regression")
        n_cols = 2
        rows = math.ceil(n_top / n_cols)
        for row in range(rows):
            cols_ui = st.columns(n_cols)
            for col_i in range(n_cols):
                idx = row * n_cols + col_i
                if idx >= len(top_feats):
                    break
                feat = top_feats[idx]
                color_feat = _best_color(feat)
                grp = feat_to_group.get(feat, "")
                grp_color = _GRP_CLR.get(grp, "#607D8B")
                with cols_ui[col_i]:
                    st.markdown(
                        f"<span style='font-size:12px;font-weight:700;"
                        f"color:{grp_color}'>#{idx+1} {feat} [{grp}]</span>",
                        unsafe_allow_html=True,
                    )
                    try:
                        fig = _plot_2d_shap_orig(
                            shap_vals, X_orig, feat, color_feat, n_bins
                        )
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception as exc:
                        st.warning(f"Plot failed: {exc}")

    else:  # classification — per class
        n_per_class = min(n_top, 5)
        for cls_i, cls_name in enumerate(class_names):
            st.markdown(f"### Class: **{cls_name}** — Top {n_per_class} features")
            # Recompute SHAP for this class
            shap_cls_key = f"shap_auto_cls{cls_i}"
            if shap_cls_key not in st.session_state:
                X_train_sc = get_value("split.X_train")
                if X_train_sc is not None:
                    sample_n = min(MAX_SHAP_SAMPLES, len(X_train_sc))
                    Xs = X_train_sc.sample(sample_n, random_state=RANDOM_STATE) \
                         if len(X_train_sc) > sample_n else X_train_sc.copy()
                    try:
                        st.session_state[shap_cls_key] = get_shap_values(
                            get_value("model.object"), Xs, cls_i
                        )
                    except Exception:
                        st.session_state[shap_cls_key] = shap_vals  # fallback

            sv_cls = st.session_state.get(shap_cls_key, shap_vals)
            sv_cls_df = pd.DataFrame(sv_cls, columns=feature_names)
            top_cls = sv_cls_df.abs().mean().sort_values(ascending=False).head(n_per_class).index.tolist()

            n_cols2 = 2
            rows2 = math.ceil(n_per_class / n_cols2)
            for row in range(rows2):
                cols_ui = st.columns(n_cols2)
                for col_i in range(n_cols2):
                    idx = row * n_cols2 + col_i
                    if idx >= len(top_cls):
                        break
                    feat = top_cls[idx]
                    # best color: most correlated other feature from this class's top
                    s_cls = sv_cls[:, feature_names.index(feat)]
                    best_r, color_feat = 0.0, top_cls[1] if len(top_cls) > 1 else top_cls[0]
                    for other in top_cls:
                        if other == feat:
                            continue
                        xo = pd.to_numeric(X_orig[other], errors="coerce").values
                        valid = ~(np.isnan(xo) | np.isnan(s_cls))
                        if valid.sum() < 5:
                            continue
                        r = abs(np.corrcoef(xo[valid], s_cls[valid])[0, 1])
                        if not np.isnan(r) and r > best_r:
                            best_r, color_feat = r, other
                    grp = feat_to_group.get(feat, "")
                    grp_color = _GRP_CLR.get(grp, "#607D8B")
                    with cols_ui[col_i]:
                        st.markdown(
                            f"<span style='font-size:12px;font-weight:700;"
                            f"color:{grp_color}'>#{idx+1} {feat} [{grp}]</span>",
                            unsafe_allow_html=True,
                        )
                        try:
                            fig = _plot_2d_shap_orig(
                                sv_cls, X_orig, feat, color_feat, n_bins,
                                title_suffix=f" | class={cls_name}",
                            )
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        except Exception as exc:
                            st.warning(f"Plot failed: {exc}")
            st.divider()


# ─────────────────────────────────────────────────────────────────────────────

def render():
    init_state()

    st.title("🧠 Explainability")
    st.divider()

    # ── Guards ────────────────────────────────────────────────────────────
    model     = get_value("model.object")
    X_train   = get_value("split.X_train")   # scaled — feeds the model
    y_train   = get_value("split.y_train")
    task_type = get_value("model.task_type")

    if model is None or X_train is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Train** first.")
        st.stop()

    # ── Original-scale training data ──────────────────────────────────────
    # This is the ONLY source of unscaled values used throughout this page.
    # We never call inverse_transform — we use the saved unscaled DataFrame.
    X_train_orig = _build_X_train_orig(X_train)

    feature_names = list(X_train.columns)   # must match SHAP dims

    class_names: list[str] = []
    if task_type == "classification":
        y = get_value("data.y")
        if y is not None:
            class_names = [str(c) for c in sorted(y.unique())]

    # Feature → group mapping (set by preprocessing page)
    feat_to_group: dict = get_value("preprocessing.feat_to_group") or {}

    # ── Class selector ────────────────────────────────────────────────────
    class_idx = 0
    if task_type == "classification" and class_names:
        class_idx = st.selectbox(
            "Class for SHAP / PDP",
            options=list(range(len(class_names))),
            format_func=lambda i: f"{i} – {class_names[i]}",
        )

    # ── SHAP subsample ────────────────────────────────────────────────────
    sample_n = min(MAX_SHAP_SAMPLES, len(X_train))
    # Scaled data for SHAP computation (model input)
    X_shap_scaled = (
        X_train.sample(sample_n, random_state=RANDOM_STATE)
        if len(X_train) > sample_n
        else X_train.copy()
    )
    # Unscaled data — same rows — for all plot x-axes
    X_shap_orig = X_train_orig.loc[X_shap_scaled.index].copy()

    # ── Compute SHAP values ───────────────────────────────────────────────
    shap_key = f"shap_{id(model)}_{class_idx}"
    if shap_key not in st.session_state:
        with st.spinner("Computing SHAP values…"):
            try:
                st.session_state[shap_key] = get_shap_values(
                    model, X_shap_scaled, class_idx   # model always sees scaled input
                )
            except Exception as exc:
                st.warning(f"SHAP computation failed: {exc}")
                st.session_state[shap_key] = None

    shap_vals = st.session_state[shap_key]

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_shap, tab_auto, tab_pdp, tab_rules = st.tabs([
        "🐝 SHAP Analysis",
        "📊 Explainability Plots",
        "📉 PDP",
        "📜 Surrogate Rules",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — SHAP Analysis (manual, interactive)
    # ══════════════════════════════════════════════════════════════════════
    with tab_shap:
        if shap_vals is None:
            st.warning("SHAP values could not be computed.")
        else:
            # Beeswarm — uses X_shap_orig so x-axis ticks are original values
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Beeswarm (global impact) — original scale**")
                try:
                    # shap.summary_plot uses X_sample for the colour bar values
                    # Passing X_shap_orig means colour = original concentration
                    st.pyplot(plot_shap_beeswarm(shap_vals, X_shap_orig))
                except Exception as exc:
                    st.warning(f"Beeswarm failed: {exc}")

            with col2:
                st.markdown("**Mean |SHAP| bar**")
                imp_df = shap_importance_df(shap_vals, feature_names)
                colors = [_GRP_CLR.get(feat_to_group.get(f, ""), "#607D8B")
                          for f in imp_df["Feature"]]
                fig_bar = px.bar(
                    imp_df.sort_values("MeanAbsSHAP", ascending=True),
                    x="MeanAbsSHAP", y="Feature", orientation="h",
                    color="Feature",
                    color_discrete_sequence=colors,
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("**SHAP Dependence — LOWESS trend (original scale)**")
            dep_feat = st.selectbox("Feature", feature_names, key="shap_dep")
            try:
                # plot_shap_dependence uses X_sample for x-axis → pass orig
                fig = plot_shap_dependence(shap_vals, X_shap_orig, dep_feat)
                fig_to_st(fig)
            except Exception as exc:
                st.warning(f"Dependence plot failed: {exc}")

            st.markdown("**SHAP Dependence — interaction colouring (original scale)**")
            dep_feat_int = st.selectbox("X-axis feature", feature_names, key="shap_dep_int")
            int_opts = ["Auto"] + [f for f in feature_names if f != dep_feat_int]
            int_feat = st.selectbox("Colour by", int_opts, key="shap_dep_color")
            try:
                fig_int, ax_int = plt.subplots(figsize=(9, 5))
                _shap.dependence_plot(
                    dep_feat_int,
                    shap_vals,
                    X_shap_orig,        # ← original scale x-axis
                    interaction_index=None if int_feat == "Auto" else int_feat,
                    ax=ax_int,
                    show=False,
                )
                fig_int.tight_layout()
                st.pyplot(fig_int)
                plt.close(fig_int)
            except Exception as exc:
                st.warning(f"Interaction dependence failed: {exc}")

            st.markdown("**2D SHAP Dependence — bins + scatter (original scale)**")
            c1, c2 = st.columns(2)
            feat_2d = c1.selectbox("X-axis feature", feature_names, key="shap_2d_x")
            int_opts_2d = ["Auto"] + [f for f in feature_names if f != feat_2d]
            int_feat_2d = c2.selectbox("Colour by", int_opts_2d, key="shap_2d_color")
            n_bins_2d = st.slider("X bins", 5, 20, 10, key="shap_2d_bins")
            try:
                fig_2d = _plot_2d_shap_orig(
                    shap_vals, X_shap_orig,
                    feat_2d,
                    None if int_feat_2d == "Auto" else int_feat_2d,
                    n_bins_2d,
                )
                st.pyplot(fig_2d)
                plt.close(fig_2d)
            except Exception as exc:
                st.error(f"2D SHAP failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Auto Explainability Plots
    # ══════════════════════════════════════════════════════════════════════
    with tab_auto:
        st.subheader("📊 Automatic Explainability Plots")
        if shap_vals is None:
            st.warning("SHAP values unavailable.")
        else:
            ac1, ac2 = st.columns(2)
            n_top_auto = ac1.slider("Top N features", 4, 20, 10, key="auto_n_top")
            n_bins_auto = ac2.slider("X bins per plot", 5, 20, 10, key="auto_bins")

            with st.spinner("Generating plots…"):
                _auto_plots_tab(
                    shap_vals=shap_vals,
                    X_orig=X_shap_orig,
                    feature_names=feature_names,
                    feat_to_group=feat_to_group,
                    task_type=task_type,
                    class_names=class_names,
                    class_idx=class_idx,
                    n_top=n_top_auto,
                    n_bins=n_bins_auto,
                )

    # ══════════════════════════════════════════════════════════════════════
    # Tab 3 — PDP
    # ══════════════════════════════════════════════════════════════════════
    # with tab_pdp:
    #     st.subheader("📉 Partial Dependence Plots")
    #     st.caption(
    #         "PDP grid values are computed in model-input space (scaled). "
    #         "Feature axis labels reflect scaled values for PDP; "
    #         "use SHAP dependence plots above for original-scale inspection."
    #     )
    #     pdp_feat = st.selectbox("Feature for 1D PDP", feature_names, key="pdp_1d")
    #     try:
    #         fig = plot_pdp_1d(
    #             model, X_train, pdp_feat,
    #             class_idx if task_type == "classification" else None,
    #             task_type,
    #         )
    #         fig_to_st(fig)
    #     except Exception as exc:
    #         st.warning(f"1D PDP failed: {exc}")

    #     if len(feature_names) >= 2:
    #         st.markdown("**2D PDP (interaction surface)**")
    #         p1, p2 = st.columns(2)
    #         f1 = p1.selectbox("Feature 1", feature_names, key="pdp_2d_f1")
    #         f2 = p2.selectbox("Feature 2", [f for f in feature_names if f != f1], key="pdp_2d_f2")
    #         try:
    #             fig = plot_pdp_2d(
    #                 model, X_train, f1, f2,
    #                 class_idx if task_type == "classification" else None,
    #                 task_type,
    #             )
    #             fig_to_st(fig)
    #         except Exception as exc:
    #             st.warning(f"2D PDP failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 4 — Surrogate Rules
    # ══════════════════════════════════════════════════════════════════════
    with tab_rules:
        st.subheader("📜 Surrogate Rule Extraction")
        st.caption(
            "A shallow decision tree trained on **unscaled** data — "
            "thresholds show real concentrations (e.g. AOS ≤ 1.50), "
            "not scaled values."
        )
        depth = st.slider(
            "Rule tree max depth",
            SURROGATE_DEPTH_RANGE[0], SURROGATE_DEPTH_RANGE[1],
            SURROGATE_MAX_DEPTH, key="rule_depth",
        )
        try:
            _, rules_text = extract_rules(
                X_train_orig, y_train, feature_names,
                class_names, task_type, depth,
            )
            st.code(rules_text)
        except Exception as exc:
            st.warning(f"Rule extraction failed: {exc}")