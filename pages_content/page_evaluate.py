"""
pages_content/page_evaluate.py
Metrics, Plots, Explainability, Model Comparison, Explainability Plots.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from state.session import init_state, get_value
from components.metrics_card import render_metrics_card
from core.viz.evaluation import (
    draw_confusion_matrix, draw_roc_curve,
    draw_residuals, draw_pred_vs_actual,
)
from core.viz.style import fig_to_st
from config.settings import RANDOM_STATE


# ── colour map for groups ─────────────────────────────────────────────────────
_GRP_CLR = {
    "Nanoparticle": "#1565C0", "Anionic": "#E53935",
    "Nonionic": "#FB8C00",     "Zwitterionic": "#8E24AA",
    "Surfactant": "#FF8F00",   "Polymer": "#6A1B9A",
    "Citric/Buffer": "#2E7D32","Acid/Chelant": "#C62828",
    "Antiscalant": "#00695C",  "Brine": "#4527A0",
    "Oil": "#BF360C",          "Process": "#546E7A",
    "Interaction": "#795548",  "Custom": "#37474F",
}


# ── original-scale helper ─────────────────────────────────────────────────────
def _get_orig(X_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Return the unscaled version of X_scaled by matching rows from data.X_original.
    Falls back to X_scaled when no scaler was used (they are identical).
    """
    if X_scaled is None:
        return None
    X_full_orig = get_value("data.X_original")
    if X_full_orig is None:
        return X_scaled.copy()
    try:
        out = X_full_orig.loc[X_scaled.index].copy()
    except KeyError:
        out = X_full_orig.iloc[: len(X_scaled)].copy()
    return out.reindex(columns=X_scaled.columns)


# ── ROC curve that handles string labels ─────────────────────────────────────
def _draw_roc_safe(y_true, y_prob, class_names=None):
    """
    Wrapper around draw_roc_curve that handles string class labels
    (e.g. 'High'/'Low') by encoding them to integers first.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_prob = np.array(y_prob)
    classes = sorted(np.unique(y_true), key=str)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(7, 5))

    if n_classes == 2:
        # Binary — encode labels to 0/1
        le = LabelEncoder()
        y_enc = le.fit_transform(y_true)
        probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        fpr, tpr, _ = roc_curve(y_enc, probs)
        roc_auc = auc(fpr, tpr)
        pos_label = class_names[1] if class_names else str(le.classes_[1])
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f} (pos={pos_label})")
    else:
        # Multiclass — one-vs-rest
        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            label = (class_names[i] if class_names and i < len(class_names)
                     else str(cls))
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ── auto 2D SHAP plot (original scale) ───────────────────────────────────────
def _plot_2d_shap(shap_vals, X_orig, feature, color_feature=None,
                  n_bins=10, title_suffix=""):
    feat_names = list(X_orig.columns)
    if feature not in feat_names:
        raise ValueError(f"'{feature}' not in X_orig")
    feat_idx = feat_names.index(feature)
    x_vals = pd.to_numeric(X_orig[feature], errors="coerce").values
    s_vals = shap_vals[:, feat_idx]
    valid  = ~(np.isnan(x_vals) | np.isnan(s_vals))
    x_v, s_v = x_vals[valid], s_vals[valid]

    if color_feature and color_feature in feat_names:
        c_raw = pd.to_numeric(X_orig[color_feature], errors="coerce")
        if not pd.api.types.is_numeric_dtype(c_raw):
            c_raw = c_raw.astype("category").cat.codes.astype(float)
        c_v = c_raw.values[valid]
        c_v = np.where(np.isnan(c_v), 0.0, c_v)
        c_lbl = color_feature
    else:
        c_v, c_lbl = s_v, f"SHAP({feature})"

    # ── bin means (bar chart) ─────────────────────────────────────────────────
    if x_v.max() > x_v.min():
        bins  = np.linspace(x_v.min(), x_v.max(), n_bins + 1)
        bidx  = np.clip(np.digitize(x_v, bins, right=True), 1, n_bins)
        bctrs = 0.5 * (bins[:-1] + bins[1:])
        bmeans = np.array([s_v[bidx == b].mean() if (bidx == b).any() else 0.0
                           for b in range(1, n_bins + 1)])
        bw = (bins[1] - bins[0]) * 0.7
    else:
        bctrs, bmeans, bw = np.array([x_v.mean()]), np.array([s_v.mean()]), 0.1

    c_min, c_max = float(c_v.min()), float(c_v.max())
    if c_min == c_max:
        c_max = c_min + 1.0

    # ── layout: histogram on top, main plot below ─────────────────────────────
    fig = plt.figure(figsize=(8, 5.2))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.06)
    ax_hist = fig.add_subplot(gs[0])
    ax      = fig.add_subplot(gs[1], sharex=ax_hist)

    # ── marginal histogram ────────────────────────────────────────────────────
    ax_hist.hist(x_v, bins=40, color="#3498db", alpha=0.5, edgecolor="none")
    ax_hist.set_ylabel("count", fontsize=7)
    ax_hist.tick_params(labelbottom=False, labelsize=7)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.axhline(0, color="none")  # keeps y-axis tidy

    # ── bars + scatter ────────────────────────────────────────────────────────
    ax.bar(bctrs, bmeans, width=bw, color="lightgrey", edgecolor="white",
           zorder=1, label="Mean SHAP / bin")
    sc = ax.scatter(x_v, s_v, c=c_v, cmap="coolwarm", s=28, alpha=0.8,
                    zorder=2, vmin=c_min, vmax=c_max)
    ax.axhline(0, color="grey", lw=0.9, ls="--", zorder=0)

    # ── trend line + confidence band ──────────────────────────────────────────
    sort_idx = np.argsort(x_v)
    x_s, y_s = x_v[sort_idx], s_v[sort_idx]

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lo      = lowess(y_s, x_s, frac=0.4, return_sorted=True)
        x_sm, y_sm = lo[:, 0], lo[:, 1]

        # Heteroscedastic band: rolling residual std
        y_interp  = np.interp(x_s, x_sm, y_sm)
        residuals = y_s - y_interp
        win       = max(10, len(residuals) // 20)
        roll_std  = (pd.Series(residuals)
                       .rolling(win, center=True, min_periods=3)
                       .std()
                       .fillna(residuals.std())
                       .values)

        ax.fill_between(x_s, y_interp - roll_std, y_interp + roll_std,
                        alpha=0.18, color="#e74c3c", zorder=3, label="±1 σ band")
        ax.plot(x_sm, y_sm, color="#e74c3c", lw=2.2, zorder=4, label="LOWESS trend")

    except ImportError:
        # Fallback: polynomial trend
        deg   = min(3, max(1, len(x_s) // 50))
        coefs = np.polyfit(x_s, y_s, deg)
        x_lin = np.linspace(x_s.min(), x_s.max(), 300)
        y_lin = np.polyval(coefs, x_lin)

        # Bootstrap confidence band (no extra deps)
        rng       = np.random.default_rng(42)
        n         = len(x_s)
        boot_curves = []
        for _ in range(200):
            idx_b = rng.integers(0, n, n)
            try:
                c_b = np.polyfit(x_s[idx_b], y_s[idx_b], deg)
                boot_curves.append(np.polyval(c_b, x_lin))
            except np.linalg.LinAlgError:
                pass
        if boot_curves:
            boot_arr = np.vstack(boot_curves)
            ax.fill_between(x_lin,
                            np.percentile(boot_arr, 16, axis=0),
                            np.percentile(boot_arr, 84, axis=0),
                            alpha=0.18, color="#e74c3c", zorder=3,
                            label="68 % CI band")
        ax.plot(x_lin, y_lin, color="#e74c3c", lw=2.2, zorder=4,
                label=f"Poly-{deg} trend")

    # ── colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(c_lbl, rotation=270, labelpad=14)
    mid = (c_min + c_max) / 2
    cb.set_ticks([c_min, mid, c_max])
    cb.set_ticklabels([f"{c_min:.3g}", f"{mid:.3g}", f"{c_max:.3g}"])

    # ── labels + legend ───────────────────────────────────────────────────────
    ax.set_xlabel(f"{feature}  (original scale)")
    ax.set_ylabel(f"SHAP value for\n{feature}")
    ax.set_title(f"SHAP Dependence — {feature}{title_suffix}", pad=2)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    fig.tight_layout()
    return fig

def render():
    init_state()

    st.title("📈 Model Evaluation")
    st.divider()

    # ── Guards ────────────────────────────────────────────────────────────
    model      = get_value("model.object")
    model_name = get_value("model.name")
    task_type  = get_value("model.task_type")
    metrics    = get_value("results.metrics")
    X_test     = get_value("split.X_test")
    X_train    = get_value("split.X_train")   # scaled
    y_test     = get_value("split.y_test")
    y_train    = get_value("split.y_train")

    if model is None or X_test is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Train** first.")
        st.stop()

    # Unscaled versions for display
    X_train_orig = _get_orig(X_train)
    X_test_orig  = _get_orig(X_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    feature_names = (
        get_value("data.processed_feature_names")
        or get_value("data.feature_names")
        or list(X_test.columns)
    )
    feat_to_group: dict = get_value("preprocessing.feat_to_group") or {}

    class_names: list[str] = []
    if task_type == "classification":
        y = get_value("data.y")
        if y is not None:
            class_names = [str(c) for c in sorted(y.unique())]

    st.info(f"**Model:** {model_name} &nbsp;|&nbsp; **Task:** {task_type.capitalize()}")

    # ── Tabs ──────────────────────────────────────────────────────────────
    (tab_metrics, tab_plots, tab_explain,
     tab_compare, tab_auto) = st.tabs([
        "📊 Metrics",
        "📉 Plots",
        "🧠 Explainability",
        "⚖️ Model Comparison",
        "📊 Explainability Plots",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # Tab 1 — Metrics
    # ══════════════════════════════════════════════════════════════════════
    with tab_metrics:
        st.subheader("Test-Set Metrics")
        render_metrics_card(metrics, task_type=task_type)
        if task_type == "classification":
            report = metrics.get("classification_report", {})
            if report:
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # Tab 2 — Plots
    # ══════════════════════════════════════════════════════════════════════
    with tab_plots:
        if task_type == "classification":
            st.subheader("Confusion Matrix")
            classes = list(np.unique(y_test))
            fig = draw_confusion_matrix(y_test, y_pred, labels=classes)
            fig_to_st(fig)

            if y_prob is not None:
                st.subheader("ROC Curve")
                try:
                    # Use safe wrapper that handles string labels (High/Low/Mid)
                    fig = _draw_roc_safe(y_test, y_prob, class_names or None)
                    fig_to_st(fig)
                except Exception as exc:
                    st.warning(f"ROC curve failed: {exc}")
            else:
                st.info("Model does not support probability estimates — ROC curve unavailable.")
        else:
            st.subheader("Predicted vs Actual")
            fig = draw_pred_vs_actual(y_test, y_pred)
            fig_to_st(fig)
            st.subheader("Residuals")
            fig = draw_residuals(y_test, y_pred)
            fig_to_st(fig)

    # ══════════════════════════════════════════════════════════════════════
    # Tab 3 — Explainability (interactive SHAP + PDP + surrogate rules)
    # ══════════════════════════════════════════════════════════════════════
    with tab_explain:
        if X_train is None:
            st.warning("Training data not found. Re-train the model first.")
            st.stop()

        try:
            from core.models.explainability import (
                get_shap_values, shap_importance_df,
                plot_shap_beeswarm, plot_shap_dependence,
                plot_pdp_1d, plot_pdp_2d, extract_rules,
                MAX_SHAP_SAMPLES, SURROGATE_MAX_DEPTH, SURROGATE_DEPTH_RANGE,
            )
            import shap as _shap

            class_idx = 0
            if task_type == "classification" and class_names:
                class_idx = st.selectbox(
                    "Class for SHAP / PDP",
                    options=list(range(len(class_names))),
                    format_func=lambda i: f"{i} – {class_names[i]}",
                    key="eval_shap_class",
                )

            # Scaled sample for SHAP computation; orig for plot axes
            sample_n = min(MAX_SHAP_SAMPLES, len(X_train))
            X_shap_sc = (X_train.sample(sample_n, random_state=RANDOM_STATE)
                         if len(X_train) > sample_n else X_train.copy())
            X_shap_orig = X_train_orig.loc[X_shap_sc.index].copy()

            shap_key = f"eval_shap_{id(model)}_{class_idx}"
            if shap_key not in st.session_state:
                with st.spinner("Computing SHAP values…"):
                    try:
                        st.session_state[shap_key] = get_shap_values(
                            model, X_shap_sc, class_idx)
                    except Exception as exc:
                        st.warning(f"SHAP failed: {exc}")
                        st.session_state[shap_key] = None

            shap_vals = st.session_state[shap_key]

            if shap_vals is not None:
                # Beeswarm
                st.subheader("🐝 SHAP Beeswarm (original scale)")
                col1, col2 = st.columns([2, 1])
                with col1:
                    try:
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
                        color="Feature", color_discrete_sequence=colors,
                    )
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Dependence
                st.markdown("**SHAP Dependence (original scale)**")
                dep_feat = st.selectbox("Feature", feature_names, key="eval_dep")
                try:
                    fig = plot_shap_dependence(shap_vals, X_shap_orig, dep_feat)
                    fig_to_st(fig)
                except Exception as exc:
                    st.warning(f"Dependence plot failed: {exc}")

                # Interaction dependence
                # Interaction dependence
                st.markdown("**SHAP Dependence — interaction colouring**")
                d1, d2 = st.columns(2)
                dep_int  = d1.selectbox("X-axis feature", feature_names, key="eval_dep_int")
                int_opts = ["Auto"] + [f for f in feature_names if f != dep_int]
                int_feat = d2.selectbox("Colour by", int_opts, key="eval_dep_color")
                try:
                    fig_int = _plot_2d_shap(
                        shap_vals   = shap_vals,
                        X_orig      = X_shap_orig,
                        feature     = dep_int,
                        color_feature = None if int_feat == "Auto" else int_feat,
                    )
                    st.pyplot(fig_int)
                    plt.close(fig_int)
                except Exception as exc:
                    st.warning(f"Interaction dependence failed: {exc}")

            st.divider()

            # PDP
            # st.subheader("📉 Partial Dependence Plots")
            # pdp_feat = st.selectbox("Feature for PDP", feature_names, key="eval_pdp")
            # try:
            #     fig = plot_pdp_1d(model, X_train, pdp_feat,
            #                       class_idx if task_type == "classification" else None,
            #                       task_type)
            #     fig_to_st(fig)
            # except Exception as exc:
            #     st.warning(f"PDP failed: {exc}")

            # st.divider()

            # Surrogate rules — unscaled data so thresholds are real concentrations
            st.subheader("📜 Surrogate Rules")
            depth = st.slider("Rule tree depth",
                              SURROGATE_DEPTH_RANGE[0], SURROGATE_DEPTH_RANGE[1],
                              SURROGATE_MAX_DEPTH, key="eval_depth")
            try:
                _, rules_text = extract_rules(
                    X_train_orig, y_train, feature_names,
                    class_names, task_type, depth,
                )
                st.code(rules_text)
                st.caption(
                    "Trained on **unscaled** data — thresholds show real concentrations."
                )
            except Exception as exc:
                st.warning(f"Rule extraction failed: {exc}")

        except ImportError as e:
            st.warning(f"Explainability requires extra packages: {e}\n"
                       "Run: `pip install shap statsmodels`")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 4 — Model Comparison
    # ══════════════════════════════════════════════════════════════════════
    with tab_compare:
        trained_models = get_value("results.trained_models") or []
        if len(trained_models) < 2:
            st.info("Train at least **2 models** to compare them here.")
        else:
            rows = []
            for m in trained_models:
                row = {"Model": m["name"]}
                row.update({k: round(v, 4) for k, v in m["metrics"].items()
                            if isinstance(v, (int, float))})
                rows.append(row)
            compare_df = pd.DataFrame(rows)
            st.subheader(f"Comparison of {len(trained_models)} models")
            st.dataframe(compare_df, use_container_width=True)
            if task_type == "classification" and "accuracy" in compare_df.columns:
                best = compare_df.loc[compare_df["accuracy"].idxmax(), "Model"]
                st.success(f"🏆 Best accuracy: **{best}**")
            elif task_type == "regression" and "r2" in compare_df.columns:
                best = compare_df.loc[compare_df["r2"].idxmax(), "Model"]
                st.success(f"🏆 Best R²: **{best}**")

    # ══════════════════════════════════════════════════════════════════════
    # Tab 5 — Explainability Plots (auto 2D SHAP, top features)
    # ══════════════════════════════════════════════════════════════════════
    with tab_auto:
        st.subheader("📊 Explainability Plots")
        st.caption(
            "Auto-generated 2D SHAP dependence for top features by mean |SHAP|. "
            "Grey bars = mean SHAP per feature bin. "
            "Dots coloured by the most correlated other feature. "
            "X-axis in **original (unscaled) units**."
        )

        if X_train is None:
            st.warning("Training data not found.")
        else:
            try:
                from core.models.explainability import (
                    get_shap_values, MAX_SHAP_SAMPLES,
                )

                ac1, ac2 = st.columns(2)
                n_top   = ac1.slider("Top N features", 4, 20, 10, key="auto_ntop")
                n_bins  = ac2.slider("X bins per plot", 5, 20, 10, key="auto_bins")
                class_idx_auto = 0

                if task_type == "classification" and class_names:
                    class_idx_auto = st.selectbox(
                        "Class for auto plots",
                        options=list(range(len(class_names))),
                        format_func=lambda i: f"{i} – {class_names[i]}",
                        key="auto_class",
                    )

                # Get or compute SHAP for chosen class
                sample_n = min(MAX_SHAP_SAMPLES, len(X_train))
                X_sc = (X_train.sample(sample_n, random_state=RANDOM_STATE)
                        if len(X_train) > sample_n else X_train.copy())
                X_or = X_train_orig.loc[X_sc.index].copy()

                auto_key = f"eval_auto_shap_{id(model)}_{class_idx_auto}"
                if auto_key not in st.session_state:
                    with st.spinner("Computing SHAP values…"):
                        try:
                            st.session_state[auto_key] = get_shap_values(
                                model, X_sc, class_idx_auto)
                        except Exception as exc:
                            st.warning(f"SHAP failed: {exc}")
                            st.session_state[auto_key] = None

                sv = st.session_state[auto_key]

                if sv is not None:
                    sv_df    = pd.DataFrame(sv, columns=feature_names)
                    top_f    = sv_df.abs().mean().sort_values(
                        ascending=False).head(n_top).index.tolist()

                    def _best_color(feat):
                        fi = feature_names.index(feat)
                        s  = sv[:, fi]
                        best_r, best_f = 0.0, None
                        for other in top_f:
                            if other == feat: continue
                            xo = pd.to_numeric(X_or[other], errors="coerce").values
                            ok = ~(np.isnan(xo) | np.isnan(s))
                            if ok.sum() < 5: continue
                            r = abs(np.corrcoef(xo[ok], s[ok])[0, 1])
                            if not np.isnan(r) and r > best_r:
                                best_r, best_f = r, other
                        return best_f or (top_f[1] if len(top_f) > 1 else top_f[0])

                    if task_type == "regression":
                        n_cols = 2
                        for row_i in range(math.ceil(n_top / n_cols)):
                            cols_ui = st.columns(n_cols)
                            for col_i in range(n_cols):
                                idx = row_i * n_cols + col_i
                                if idx >= len(top_f): break
                                feat = top_f[idx]
                                cf   = _best_color(feat)
                                grp  = feat_to_group.get(feat, "")
                                gc   = _GRP_CLR.get(grp, "#607D8B")
                                with cols_ui[col_i]:
                                    st.markdown(
                                        f"<span style='font-size:12px;font-weight:700;"
                                        f"color:{gc}'>#{idx+1} {feat} [{grp}]</span>",
                                        unsafe_allow_html=True,
                                    )
                                    try:
                                        fig = _plot_2d_shap(sv, X_or, feat, cf, n_bins)
                                        st.pyplot(fig, use_container_width=True)
                                        plt.close(fig)
                                    except Exception as exc:
                                        st.warning(f"{feat}: {exc}")

                    else:  # classification — 5 plots per class
                        n_per = max(n_top, 10)
                        for cls_i, cls_name in enumerate(class_names):
                            st.markdown(f"### Class: **{cls_name}** — Top {n_per} features")
                            cls_key = f"eval_auto_shap_{id(model)}_{cls_i}"
                            if cls_key not in st.session_state:
                                with st.spinner(f"SHAP for class {cls_name}…"):
                                    try:
                                        st.session_state[cls_key] = get_shap_values(
                                            model, X_sc, cls_i)
                                    except Exception:
                                        st.session_state[cls_key] = sv
                            sv_c = st.session_state[cls_key]
                            top_c = (pd.DataFrame(sv_c, columns=feature_names)
                                     .abs().mean()
                                     .sort_values(ascending=False)
                                     .head(n_per).index.tolist())

                            n_cols = 2
                            for row_i in range(math.ceil(n_per / n_cols)):
                                cols_ui = st.columns(n_cols)
                                for col_i in range(n_cols):
                                    idx = row_i * n_cols + col_i
                                    if idx >= len(top_c): break
                                    feat = top_c[idx]
                                    # Best colour for this class
                                    fi   = feature_names.index(feat)
                                    sc_  = sv_c[:, fi]
                                    br, cf = 0.0, top_c[1] if len(top_c) > 1 else top_c[0]
                                    for other in top_c:
                                        if other == feat: continue
                                        xo = pd.to_numeric(X_or[other], errors="coerce").values
                                        ok = ~(np.isnan(xo) | np.isnan(sc_))
                                        if ok.sum() < 5: continue
                                        r = abs(np.corrcoef(xo[ok], sc_[ok])[0, 1])
                                        if not np.isnan(r) and r > br:
                                            br, cf = r, other
                                    grp = feat_to_group.get(feat, "")
                                    gc  = _GRP_CLR.get(grp, "#607D8B")
                                    with cols_ui[col_i]:
                                        st.markdown(
                                            f"<span style='font-size:12px;font-weight:700;"
                                            f"color:{gc}'>#{idx+1} {feat} [{grp}]</span>",
                                            unsafe_allow_html=True,
                                        )
                                        try:
                                            fig = _plot_2d_shap(
                                                sv_c, X_or, feat, cf, n_bins,
                                                title_suffix=f" | class={cls_name}",
                                            )
                                            st.pyplot(fig, use_container_width=True)
                                            plt.close(fig)
                                        except Exception as exc:
                                            st.warning(f"{feat}: {exc}")
                            st.divider()

            except ImportError as e:
                st.warning(f"Requires shap: {e}")

    st.divider()
    st.info("👉 Proceed to **🔮 Predict** to make new predictions.")