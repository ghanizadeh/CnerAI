"""
pages_content/page_eda.py
Exploratory Data Analysis — fully decoupled from preprocessing.

Tabs
----
  1  Preview
  2  Data Filtering          ← shared render_data_filters (same as preprocessing)
  3  Feature Engineering     ← foam groups + auto-interactions for EDA only
  4  Summary
  5  Scatter + Histograms
  6  3D Scatter
  7  Box Plots
  8  Correlation
  9  Categorical
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from state.session import init_state, get_value
from components.column_selector import render_column_selector
from core.data.preprocessor import (
    extended_describe, categorical_summary,
    categorical_warnings, categorical_imbalance,
)
from core.viz.eda import (
    draw_correlation_heatmap, draw_boxplots,
    draw_histograms, draw_scatter,
    draw_pairwise_scatter_with_hist,
)
from core.viz.style import fig_to_st
from config.settings import MAX_HEATMAP_FEATURES
from utils.data_filter import render_data_filters
from core.data.foam_feature_engineering import (
    render_feature_engineering_ui,
    GRP_CLR, CHEM_GROUPS,
)




# ─────────────────────────────────────────────────────────────────────────────
#  EDA Feature Engineering tab — delegates to shared render_feature_engineering_ui
# ─────────────────────────────────────────────────────────────────────────────

def _render_fe_tab(df_sub: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Render the EDA Feature Engineering tab using the same shared UI as the
    Preprocessing page (key_prefix="eda" keeps widget keys isolated).
    Returns df_sub with engineered columns merged in (EDA-only, not saved).
    """
    st.markdown(
        "Configure chemical groups below. Click **▶ Apply to EDA dataset** "
        "to add the engineered features to the current EDA session — "
        "**no changes are saved to preprocessing or the model**."
    )

    df_out, feature_cols, fg = render_feature_engineering_ui(
        df_sub, key_prefix="eda", header=False,
    )

    st.divider()
    if st.button(
        "▶ Apply to EDA dataset",
        type="primary",
        use_container_width=True,
        key="eda_fe_apply",
        help="Adds engineered columns to the EDA dataset for this session only. "
             "Does NOT affect preprocessing or model training.",
    ):
        df_merged = df_sub.copy().reset_index(drop=True)
        for col in feature_cols:
            if col in df_out.columns and col not in df_merged.columns:
                df_merged[col] = df_out[col].values
        st.session_state["eda_enriched_df"] = df_merged
        st.session_state["eda_fe_applied"]  = True
        st.success(
            f"✅ Added **{len(feature_cols)}** engineered features to EDA dataset. "
            "All other EDA tabs now show these features too."
        )
        st.rerun()

    # Return enriched df if already applied and shape matches, else original
    if st.session_state.get("eda_fe_applied") and "eda_enriched_df" in st.session_state:
        enriched = st.session_state["eda_enriched_df"]
        if len(enriched) == len(df_sub):
            st.caption(
                f"🟢 Engineered features are active — EDA dataset has "
                f"**{enriched.shape[1]}** columns."
            )
            if st.button("↩ Remove engineered features", key="eda_fe_reset"):
                st.session_state.pop("eda_enriched_df", None)
                st.session_state["eda_fe_applied"] = False
                st.rerun()
            return enriched
        else:
            st.session_state.pop("eda_enriched_df", None)
            st.session_state["eda_fe_applied"] = False

    return df_sub


# ─────────────────────────────────────────────────────────────────────────────
#  Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    init_state()

    st.title("📊 Exploratory Data Analysis")
    st.divider()

    df = get_value("data.raw")
    if df is None:
        st.warning("⚠️ No dataset loaded. Go to **📂 Data** first.")
        st.stop()

    # ── Column selector ───────────────────────────────────────────────────
    st.subheader("🎯 Select Columns to Explore")
    with st.expander("Column selector", expanded=True):
        features, target = render_column_selector(
            df,
            default_features=get_value("data.feature_names"),
            default_target=get_value("data.target_name"),
            key_prefix="eda",
        )

    if not features or target is None:
        st.info("Select at least one feature and a target column above.")
        st.stop()

    selected_cols = list(dict.fromkeys(features + [target]))
    df_base = df[selected_cols].copy()

    # ── EDA tabs ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔎 EDA Dashboard")

    (tab_prev, tab_filt, tab_fe, tab_sum,
     tab_scat, tab_3d, tab_box, tab_corr, tab_cat) = st.tabs([
        "🔎 Preview",
        "🔍 Data Filtering",
        "⚗️ Feature Engineering",
        "📝 Summary",
        "📊 Scatter + Histograms",
        "🧊 3D Scatter",
        "📦 Box Plots",
        "🔗 Correlation",
        "⚠️ Categorical",
    ])

    # ── Tab 1: Preview ────────────────────────────────────────────────────
    with tab_prev:
        st.write(f"**Shape:** {df_base.shape[0]:,} rows × {df_base.shape[1]} cols")
        st.dataframe(df_base, use_container_width=True)

    # ── Tab 2: Data Filtering ─────────────────────────────────────────────
    with tab_filt:
        st.subheader("🔍 Filter Rows for EDA")
        st.caption(
            "Filters here affect all EDA tabs below — they do **not** touch "
            "the preprocessing pipeline or model data."
        )
        # Use the shared filter widget (eda-namespaced keys)
        df_sub = render_data_filters(df_base, key_prefix="eda")

        # Store filtered df in session so other tabs can use it
        st.session_state["eda_filtered_df"] = df_sub

        # Reset FE enrichment when filter changes (shape mismatch guard)
        if st.session_state.get("eda_fe_applied"):
            enriched = st.session_state.get("eda_enriched_df")
            if enriched is not None and len(enriched) != len(df_sub):
                st.session_state.pop("eda_enriched_df", None)
                st.session_state["eda_fe_applied"] = False
                st.info("ℹ️ Filter changed — engineered features reset. Re-apply in the Feature Engineering tab.")

    # Resolve working df for all downstream tabs
    # Priority: FE-enriched → filtered → base
    if st.session_state.get("eda_fe_applied") and "eda_enriched_df" in st.session_state:
        df_work = st.session_state["eda_enriched_df"]
    elif "eda_filtered_df" in st.session_state:
        df_work = st.session_state["eda_filtered_df"]
    else:
        df_work = df_base

    # Resolve feature list for df_work
    work_features = [c for c in df_work.columns if c != target]
    num_cols      = [c for c in work_features if pd.api.types.is_numeric_dtype(df_work[c])]

    # ── Tab 3: Feature Engineering ────────────────────────────────────────
    with tab_fe:
        st.subheader("⚗️ Feature Engineering — EDA Exploration")
        # Pass the currently filtered (not enriched) df so the FE tab
        # always starts from raw features, not stacked engineered ones
        filt_df = st.session_state.get("eda_filtered_df", df_base)
        df_work = _render_fe_tab(filt_df, target)

    # ── Tab 4: Summary ────────────────────────────────────────────────────
    with tab_sum:
        st.markdown("#### 🔢 Numeric Summary")
        st.dataframe(extended_describe(df_work), use_container_width=True)
        cat_df = categorical_summary(df_work)
        if not cat_df.empty:
            st.markdown("#### 🏷️ Categorical Summary")
            st.dataframe(cat_df, use_container_width=True)

    # ── Tab 5: Scatter + Histograms ───────────────────────────────────────
    with tab_scat:
        num_cols_work = [c for c in work_features
                         if c in df_work.columns
                         and pd.api.types.is_numeric_dtype(df_work[c])]

        if not num_cols_work:
            st.warning("No numeric feature columns available.")
        else:
            st.markdown("#### 🔄 Scatter + Marginal Histograms")
            y_s = df_work[target]
            n_classes = y_s.dropna().nunique()

            if pd.api.types.is_numeric_dtype(y_s) and n_classes > 5:
                st.success("Detected: Regression")
                show_trend = st.checkbox("Show R² trendline", value=True, key="eda_trend")
                # ---- Select top correlated features ----
                corr_df = df_work[num_cols_work + [target]].copy()

                # Correlation with target
                corr_vals = (
                    corr_df.corr(numeric_only=True)[target]
                    .drop(target, errors="ignore")
                    .abs()
                    .sort_values(ascending=False)
                )

                # Top 16 most correlated features
                top_features = corr_vals.head(16).index.tolist()

                # Data used for plotting
                plot_df = df_work[top_features + [target]]

                figs = draw_pairwise_scatter_with_hist(
                    plot_df,
                    target,
                    show_trend
                )
            elif n_classes <= 3:
                y_enc, mapping = pd.factorize(y_s)
                df_plot = df_work.copy()
                df_plot[target] = y_enc
                st.info(f"Detected: Classification ({n_classes} classes) — "
                        f"mapping: {dict(enumerate(mapping))}")
                # ---- Select top correlated features ----
                corr_df = df_plot[num_cols_work + [target]].copy()

                corr_vals = (
                    corr_df.corr(numeric_only=True)[target]
                    .drop(target, errors="ignore")
                    .abs()
                    .sort_values(ascending=False)
                )

                top_features = corr_vals.head(16).index.tolist()

                plot_df = df_plot[top_features + [target]]

                figs = draw_pairwise_scatter_with_hist(
                    plot_df,
                    target,
                    False
                )
            else:
                st.warning(f"⚠️ Target has {n_classes} classes — scatter disabled for >3.")
                figs = []

            if figs:
                for row_start in range(0, len(figs), 2):
                    cols_ui = st.columns(2)
                    for ci, fig in enumerate(figs[row_start: row_start+2]):
                        with cols_ui[ci]:
                            fig_to_st(fig)
            else:
                st.info("Not enough data to render plots.")

            # st.divider()
            # st.markdown("#### 📊 Histograms")
            # bins = st.slider("Histogram bins", 5, 100, 30, key="eda_bins")
            # fig_h = draw_histograms(df_work, num_cols_work, bins=bins)
            # fig_to_st(fig_h)

            # st.divider()
            # st.markdown("#### 🟢 All Features vs Target")
            # plot_all_one = st.checkbox("One combined figure", value=True, key="eda_all_one")
            # if not plot_all_one:
            #     for feat in num_cols_work:
            #         fig_s = draw_scatter(df_work, feat, target, hue_col=None)
            #         fig_to_st(fig_s)
            # else:
            #     n     = len(num_cols_work)
            #     ncols = 3
            #     nrows = math.ceil(n / ncols)
            #     fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
            #     axes = axes.flatten()
            #     for i, feat in enumerate(num_cols_work):
            #         pdf = df_work[[feat, target]].dropna()
            #         xv  = pd.to_numeric(pdf[feat],   errors="coerce")
            #         yv  = pd.to_numeric(pdf[target],  errors="coerce")
            #         ok  = xv.notna() & yv.notna()
            #         axes[i].scatter(xv[ok], yv[ok], alpha=0.6, s=18)
            #         axes[i].set_xlabel(feat, fontsize=8)
            #         axes[i].set_ylabel(target, fontsize=8)
            #         axes[i].set_title(f"{feat} vs {target}", fontsize=9)
            #     for j in range(i+1, len(axes)):
            #         fig.delaxes(axes[j])
            #     plt.tight_layout()
            #     fig_to_st(fig)

    # ── Tab 6: 3D Scatter ─────────────────────────────────────────────────
    with tab_3d:
        num_cols_3d = [c for c in df.columns
                       if pd.api.types.is_numeric_dtype(df[c])]

        enable_3d = st.checkbox("Enable Interactive Scatter", value=True, key="eda_3d_enable")
        if enable_3d:
            st.markdown("#### 🧊 2D Interactive Scatter")
            if len(num_cols_3d) >= 2:
                ca, cb, cc = st.columns(3)
                xi = ca.selectbox("X", num_cols_3d, index=0, key="i2d_x")
                yi = cb.selectbox("Y", num_cols_3d, index=min(1,len(num_cols_3d)-1), key="i2d_y")
                co_opts = ["None"] + df.columns.tolist()
                co_def  = co_opts.index(target) if target in co_opts else 0
                ci = cc.selectbox("Colour by", co_opts, index=co_def, key="i2d_c")
                if ci == "None": ci = None
                hov_opts = [c for c in df.columns if c not in [xi, yi]]
                hov_cols = st.multiselect("Hover columns", hov_opts,
                                           default=[target] if target in hov_opts else hov_opts[:2],
                                           key="i2d_hov")
                keep = list(dict.fromkeys([xi, yi]
                             + ([ci] if ci else [])
                             + [c for c in hov_cols if c not in [xi, yi, ci]]))
                pdf = df[keep].copy()
                if ci and not pd.api.types.is_numeric_dtype(pdf[ci]):
                    pdf[ci] = pdf[ci].astype(str)
                fig2d = px.scatter(pdf, x=xi, y=yi, color=ci,
                                    hover_data={c: True for c in hov_cols},
                                    color_continuous_scale="Viridis"
                                    if ci and pd.api.types.is_numeric_dtype(pdf[ci]) else None,
                                    opacity=0.75, height=500)
                fig2d.update_traces(marker=dict(size=6))
                fig2d.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2d, use_container_width=True)
            else:
                st.warning("Need ≥ 2 numeric columns.")

            st.divider()
            st.markdown("#### 🧊 3D Interactive Scatter")
            if len(num_cols_3d) >= 3:
                ca, cb, cc, cd = st.columns(4)
                xi = ca.selectbox("X", num_cols_3d, index=0, key="i3d_x")
                yi = cb.selectbox("Y", num_cols_3d, index=min(1,len(num_cols_3d)-1), key="i3d_y")
                zi = cc.selectbox("Z", num_cols_3d, index=min(2,len(num_cols_3d)-1), key="i3d_z")
                co_opts = ["None"] + df.columns.tolist()
                co_def  = co_opts.index(target) if target in co_opts else 0
                ci = cd.selectbox("Colour by", co_opts, index=co_def, key="i3d_c")
                if ci == "None": ci = None
                hov_opts = [c for c in df.columns if c not in [xi, yi, zi]]
                hov_cols = st.multiselect("Hover columns", hov_opts,
                                           default=[target] if target in hov_opts else hov_opts[:2],
                                           key="i3d_hov")
                keep = list(dict.fromkeys([xi, yi, zi]
                             + ([ci] if ci else [])
                             + [c for c in hov_cols if c not in [xi, yi, zi, ci]]))
                pdf = df[keep].copy()
                if ci and not pd.api.types.is_numeric_dtype(pdf[ci]):
                    pdf[ci] = pdf[ci].astype(str)
                fig3d = px.scatter_3d(pdf, x=xi, y=yi, z=zi, color=ci,
                                       hover_data={c: True for c in hov_cols},
                                       color_continuous_scale="Viridis"
                                       if ci and pd.api.types.is_numeric_dtype(pdf[ci]) else None,
                                       opacity=0.75, height=650)
                fig3d.update_traces(marker=dict(size=4))
                fig3d.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.warning("Need ≥ 3 numeric columns.")

    # ── Tab 7: Box Plots ──────────────────────────────────────────────────
    with tab_box:
        st.markdown("#### 📦 Boxplots")
        num_w = [c for c in work_features
                 if c in df_work.columns and pd.api.types.is_numeric_dtype(df_work[c])]
        use_std = st.checkbox("Standardise for visualization", value=False, key="eda_box_std")
        df_plot = df_work.copy()
        if use_std and num_w:
            df_plot[num_w] = StandardScaler().fit_transform(df_work[num_w])
        if num_w:
            fig_bp = draw_boxplots(df_plot, num_w)
            fig_to_st(fig_bp)
        else:
            st.info("No numeric feature columns.")

    # ── Tab 8: Correlation ────────────────────────────────────────────────
    with tab_corr:
        st.markdown("### 🟣 Correlation Matrix")
        num_w = [c for c in work_features
                 if c in df_work.columns and pd.api.types.is_numeric_dtype(df_work[c])]
        if not num_w:
            st.warning("No numeric features.")
        elif len(num_w) > MAX_HEATMAP_FEATURES:
            st.warning(f"Too many features ({len(num_w)}) — showing table only.")
            st.dataframe(df_work[num_w].corr(), use_container_width=True)
        else:
            st.dataframe(df_work[num_w + [target]].corr(), use_container_width=True)
            fig_hm, corr, rec, high_corr = draw_correlation_heatmap(
                df_work, columns=num_w + [target], target=target
            )
            fig_to_st(fig_hm)
            st.markdown("### ⭐ Recommended Features")
            if rec is not None:
                st.dataframe(rec, use_container_width=True)
            st.markdown("### ⚠️ Highly Correlated Feature Pairs")
            if not high_corr.empty:
                st.dataframe(high_corr, use_container_width=True)
            else:
                st.success("No severe multicollinearity detected.")

    # ── Tab 9: Categorical ────────────────────────────────────────────────
    with tab_cat:
        warn_df = categorical_warnings(df_work)
        if warn_df.empty:
            st.success("✅ No major categorical issues detected.")
        else:
            st.dataframe(warn_df, use_container_width=True)
        with st.expander("📊 Class Imbalance Details"):
            imb_df = categorical_imbalance(df_work)
            if imb_df.empty:
                st.info("No categorical columns found.")
            else:
                st.dataframe(imb_df, use_container_width=True)

    st.info("👉 Proceed to **⚙️ Preprocessing** when ready.")