"""
core/models/explainability.py
==============================
SHAP analysis, partial dependence plots, and surrogate rule extraction.

Ported from code0/core/explainability.py — no changes to existing logic,
only re-namespaced to fit the refactored package structure.

Public API
----------
get_shap_values(model, X_sample, class_idx)         -> np.ndarray
shap_importance_df(shap_values, feature_names)      -> pd.DataFrame
plot_shap_beeswarm(shap_values, X_sample)           -> plt.Figure
plot_shap_dependence(shap_vals, X, feature)         -> plt.Figure
plot_pdp_1d(model, X_train, feature, ...)           -> plt.Figure
plot_pdp_2d(model, X_train, f1, f2, ...)            -> plt.Figure
extract_rules(X, y, features, class_names, ...)     -> (tree, str)
"""
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

# ── Shared style constants ────────────────────────────────────────────────────
_DOT_COLOR   = "#4a90d9"
_TREND_COLOR = "#f5a623"
_BAND_ALPHA  = 0.18
_DOT_ALPHA   = 0.35
_DOT_SIZE    = 18
_TREND_LW    = 2.8

MAX_SHAP_SAMPLES       = 500
SURROGATE_MAX_DEPTH    = 3
SURROGATE_DEPTH_RANGE  = (2, 6)


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def get_shap_values(model, X_sample: pd.DataFrame, class_idx: int = 0) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sample)

    if isinstance(raw, list):
        return raw[0] if len(raw) == 1 else raw[class_idx]

    arr = np.asarray(raw)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, class_idx]
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def shap_importance_df(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs})
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(shap_values: np.ndarray, X_sample: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    lowess_frac: float = 0.4,
) -> plt.Figure:
    """Scatter of feature value vs SHAP value with orange LOWESS trend."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    idx    = list(X_sample.columns).index(feature)
    x_vals = X_sample[feature].values
    s_vals = shap_values[:, idx]

    order  = np.argsort(x_vals)
    x_s    = x_vals[order]
    s_s    = s_vals[order]
    smooth = lowess(s_s, x_s, frac=lowess_frac, return_sorted=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_vals, s_vals, alpha=_DOT_ALPHA, s=_DOT_SIZE, color=_DOT_COLOR)
    ax.plot(smooth[:, 0], smooth[:, 1], color=_TREND_COLOR, lw=_TREND_LW, label="LOWESS")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(f"SHAP Dependence – {feature}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig




def plot_shap_dependence_2d(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    n_bins: int = 10,
) -> plt.Figure:
    """
    2D SHAP dependence plot:
      - Grey bars   = mean SHAP value per X-axis bin
      - Scatter dots = individual samples coloured by interaction_feature value
      - Colourbar on the right for the interaction feature
      - Dashed zero line
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # ── Validate feature exists in X_sample ──────────────────────────────
    feat_names = list(X_sample.columns)
    if feature not in feat_names:
        raise ValueError(f"Feature '{feature}' not found in X_sample columns: {feat_names}")

    feat_idx = feat_names.index(feature)
    x_vals   = pd.to_numeric(X_sample[feature], errors="coerce").values
    s_vals   = shap_values[:, feat_idx]

    # Drop NaN rows so nothing breaks downstream
    valid    = ~np.isnan(x_vals) & ~np.isnan(s_vals)
    x_vals   = x_vals[valid]
    s_vals   = s_vals[valid]
    X_valid  = X_sample.iloc[valid] if hasattr(valid, "__len__") else X_sample

    # ── Auto-pick interaction feature ─────────────────────────────────────
    if interaction_feature is None:
        numeric_feats = [fn for fn in feat_names
                         if fn != feature and pd.api.types.is_numeric_dtype(X_sample[fn])]
        if numeric_feats:
            correlations = []
            for fn in numeric_feats:
                iv = pd.to_numeric(X_sample[fn], errors="coerce").values[valid]
                if np.isnan(iv).all():
                    continue
                mask = ~np.isnan(iv)
                if mask.sum() < 2:
                    continue
                c = np.corrcoef(iv[mask], s_vals[mask])[0, 1]
                if not np.isnan(c):
                    correlations.append((abs(c), fn))
            correlations.sort(reverse=True)
            interaction_feature = correlations[0][1] if correlations else feat_names[0]
        else:
            interaction_feature = feat_names[0]

    if interaction_feature not in feat_names:
        raise ValueError(f"Interaction feature '{interaction_feature}' not found in X_sample.")

    # Coerce interaction feature to numeric (encode categoricals as codes)
    int_series = X_sample[interaction_feature]
    if not pd.api.types.is_numeric_dtype(int_series):
        int_series = int_series.astype("category").cat.codes.astype(float)
    int_vals = pd.to_numeric(int_series, errors="coerce").values[valid]
    int_vals = np.where(np.isnan(int_vals), 0.0, int_vals)

    # ── Bin the X axis ────────────────────────────────────────────────────
    bins        = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    bin_idx     = np.digitize(x_vals, bins, right=True)
    bin_idx     = np.clip(bin_idx, 1, n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means   = np.array([
        s_vals[bin_idx == b].mean() if (bin_idx == b).any() else 0.0
        for b in range(1, n_bins + 1)
    ])

    # ── Colour mapping ────────────────────────────────────────────────────
    int_min = float(int_vals.min())
    int_max = float(int_vals.max())
    if int_min == int_max:          # constant column — avoid degenerate colorbar
        int_max = int_min + 1.0

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    bar_width = (bins[1] - bins[0]) * 0.7
    ax.bar(
        bin_centers, bin_means,
        width=bar_width, color="lightgrey", edgecolor="white",
        zorder=1, label="Mean SHAP per bin",
    )

    sc = ax.scatter(
        x_vals, s_vals,
        c=int_vals, cmap="coolwarm",
        s=35, alpha=0.85, zorder=2,
        vmin=int_min, vmax=int_max,
    )

    ax.axhline(0, color="grey", lw=0.9, ls="--", zorder=0)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(interaction_feature, rotation=270, labelpad=14)
    mid = (int_min + int_max) / 2
    cbar.set_ticks([int_min, mid, int_max])
    cbar.set_ticklabels([f"{int_min:.2g}", f"{mid:.2g}", f"{int_max:.2g}"])

    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for\n{feature}")
    ax.set_title(f"2D SHAP Dependence — {feature}  |  colour: {interaction_feature}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Partial dependence plots
# ---------------------------------------------------------------------------

def plot_pdp_1d(
    model,
    X_train: pd.DataFrame,
    feature: str,
    class_idx: Optional[int],
    problem_type: str,
) -> plt.Figure:
    """ICE + smoothed mean PDP with marginal histograms."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    feat_idx = list(X_train.columns).index(feature)
    pdp_kw   = dict(features=[feat_idx], X=X_train, kind="both", grid_resolution=50)
    if problem_type == "classification" and class_idx is not None:
        pdp_kw["response_method"] = "predict_proba"

    result   = partial_dependence(model, **pdp_kw)
    grid     = result["grid_values"][0]
    avg_line = result["average"][0]
    ice_vals = result["individual"][0]

    fig = plt.figure(figsize=(9, 6))
    gs  = gridspec.GridSpec(4, 1, hspace=0.05)
    ax_main = fig.add_subplot(gs[:3])
    ax_hist = fig.add_subplot(gs[3], sharex=ax_main)

    for ice in ice_vals:
        ax_main.plot(grid, ice, alpha=0.07, color=_DOT_COLOR, lw=0.8)

    smooth = lowess(avg_line, grid, frac=0.4, return_sorted=True)
    ax_main.plot(smooth[:, 0], smooth[:, 1], color=_TREND_COLOR, lw=_TREND_LW, label="Smoothed mean")
    ax_main.set_ylabel("Predicted" if problem_type == "regression" else f"P(class {class_idx})")
    ax_main.set_title(f"PDP + ICE — {feature}")
    ax_main.legend(fontsize=8)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_hist.hist(X_train[feature].dropna(), bins=30, color=_DOT_COLOR, alpha=0.6, edgecolor="white")
    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_pdp_2d(
    model,
    X_train: pd.DataFrame,
    f1: str,
    f2: str,
    class_idx: Optional[int],
    problem_type: str,
) -> plt.Figure:
    """Filled contour 2D PDP."""
    idx1, idx2 = list(X_train.columns).index(f1), list(X_train.columns).index(f2)
    pdp_kw = dict(features=[(idx1, idx2)], X=X_train, grid_resolution=30)
    if problem_type == "classification" and class_idx is not None:
        pdp_kw["response_method"] = "predict_proba"

    result = partial_dependence(model, **pdp_kw)
    Z      = result["average"][0]
    g1     = result["grid_values"][0]
    g2     = result["grid_values"][1]
    G1, G2 = np.meshgrid(g1, g2)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(G1, G2, Z.T, levels=20, cmap="RdYlGn")
    cs = ax.contour(G1, G2, Z.T, levels=10, colors="black", linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7)
    fig.colorbar(cf, ax=ax)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(f"2D PDP — {f1} × {f2}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Surrogate rule extraction
# ---------------------------------------------------------------------------

def extract_rules(
    X: pd.DataFrame,
    y,
    feature_names: list[str],
    class_names: list[str],
    problem_type: str,
    max_depth: int = SURROGATE_MAX_DEPTH,
):
    """Fit a shallow decision tree and return (tree, text_rules)."""
    if problem_type == "classification":
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X, y)
        text = export_text(tree, feature_names=feature_names)
    else:
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree.fit(X, y)
        text = export_text(tree, feature_names=feature_names)
    return tree, text