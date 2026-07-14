"""
utils/plots.py
==============
Reusable plot builders. All functions return fig objects (matplotlib or plotly).
No Streamlit imports.
"""
from __future__ import annotations

import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# Fixed red-blue palette for binary classification: "false" (safe) = blue, "true" (gel) = red.
# For multiclass we fall back to a high-contrast qualitative sequence.
_BINARY_COLOR_MAP = {
    # safe / no-gel variants → blue
    "false":   "#1f77b4",
    "no gel":  "#1f77b4",
    "nogel":   "#1f77b4",
    "safe":    "#1f77b4",
    "0":       "#1f77b4",
    # gel / unsafe variants → red
    "true":    "#d62728",
    "gel":     "#d62728",
    "gelling": "#d62728",
    "unsafe":  "#d62728",
    "1":       "#d62728",
    # unknown → grey
    "unknown": "#7f7f7f",
}
_MULTICLASS_SEQUENCE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def build_class_color_map(class_names: list[str]) -> dict[str, str]:
    """
    Return {class_name: hex_color}.
    Binary: hard-coded red/blue by label semantics.
    Multiclass: cycles through a high-contrast sequence.
    """
    if len(class_names) == 2:
        cmap = {}
        for name in class_names:
            key = str(name).strip().lower()
            cmap[name] = _BINARY_COLOR_MAP.get(key, "#7f7f7f")
        # If both mapped to the same colour (both unknown), fall back to blue/red
        vals = list(cmap.values())
        if len(set(vals)) < 2:
            cmap[class_names[0]] = "#1f77b4"
            cmap[class_names[1]] = "#d62728"
        return cmap
    return {name: _MULTICLASS_SEQUENCE[i % len(_MULTICLASS_SEQUENCE)]
            for i, name in enumerate(class_names)}


# ---------------------------------------------------------------------------
# Classification plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names: list[str]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true, y_proba) -> plt.Figure:
    prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regression plots
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(y_true, y_pred) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", label="Perfect fit")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred) -> plt.Figure:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_class_distribution(y_labels: pd.Series, title: str = "Target Distribution"):
    vc = y_labels.value_counts(dropna=False)
    return px.bar(x=vc.index.astype(str), y=vc.values,
                  labels={"x": "Class", "y": "Count"}, title=title)


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_histogram_by_class(df: pd.DataFrame, feature: str, target: str):
    return px.histogram(df, x=feature, color=target, barmode="overlay",
                        marginal="box", title=f"{feature} distribution by {target}")


def plot_boxplot_by_class(df: pd.DataFrame, feature: str, target: str):
    return px.box(df, x=target, y=feature, points="all",
                  title=f"{feature} by {target}")


def plot_scatter_pair(df: pd.DataFrame, feat_x: str, feat_y: str, color_col: str):
    return px.scatter(df, x=feat_x, y=feat_y, color=color_col,
                      title=f"{feat_x} vs {feat_y}")


def top_feature_pairs(features: list[str], n: int = 6) -> list[tuple[str, str]]:
    return list(itertools.combinations(features, 2))[:n]


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(fi_df: pd.DataFrame, title: str = "Feature Importance"):
    return px.bar(
        fi_df.sort_values(fi_df.columns[1], ascending=True),
        x=fi_df.columns[1],
        y="Feature",
        orientation="h",
        title=title,
    )


# ---------------------------------------------------------------------------
# Safe region / optimisation plots
# ---------------------------------------------------------------------------

def plot_safe_region_2d(
    scored_df: pd.DataFrame,
    orig_X: pd.DataFrame,
    orig_labels: list[str],
    x_feat: str,
    y_feat: str,
    color_col: str = "pred_class",
    class_names: Optional[list[str]] = None,
    sample_n: int = 5000,
    random_state: int = 42,
) -> go.Figure:
    # Subsample synthetic points so the plot matches the slider value exactly
    if len(scored_df) > sample_n:
        scored_df = scored_df.sample(sample_n, random_state=random_state)

    hover_cols = [c for c in ["safe_probability", "safety_margin", "predicted_value"]
                  if c in scored_df.columns]

    # Build colour map for categorical (classification) plots
    if color_col == "pred_class" and class_names:
        cmap = build_class_color_map(class_names)
        fig = px.scatter(
            scored_df,
            x=x_feat,
            y=y_feat,
            color=color_col,
            color_discrete_map=cmap,
            opacity=0.45,
            hover_data=hover_cols,
            title=f"Predicted region: {x_feat} vs {y_feat}",
            category_orders={color_col: class_names},
        )
    else:
        # Regression: continuous colour scale
        fig = px.scatter(
            scored_df,
            x=x_feat,
            y=y_feat,
            color=color_col,
            color_continuous_scale="RdBu",
            opacity=0.45,
            hover_data=hover_cols,
            title=f"Predicted region: {x_feat} vs {y_feat}",
        )

    # Overlay original data points coloured by their true label
    orig_plot = orig_X[[x_feat, y_feat]].copy()
    orig_plot["label"] = orig_labels

    if class_names and len(class_names) <= 8:
        orig_cmap = build_class_color_map(class_names)
        for cls in class_names:
            subset = orig_plot[orig_plot["label"] == cls]
            if subset.empty:
                continue
            fig.add_trace(go.Scatter(
                x=subset[x_feat],
                y=subset[y_feat],
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="x",
                    color=orig_cmap.get(cls, "#000000"),
                    line=dict(width=2, color="white"),
                ),
                name=f"Actual: {cls}",
                text=subset["label"],
            ))
    else:
        fig.add_trace(go.Scatter(
            x=orig_plot[x_feat],
            y=orig_plot[y_feat],
            mode="markers",
            marker=dict(size=10, symbol="x", color="black"),
            name="Original data",
            text=orig_plot["label"],
        ))

    return fig


def plot_safe_region_3d(
    scored_df: pd.DataFrame,
    x_feat: str,
    y_feat: str,
    z_feat: str,
    color_col: str = "pred_class",
    class_names: Optional[list[str]] = None,
    sample_n: int = 5000,
    random_state: int = 42,
) -> go.Figure:
    sample = scored_df.sample(min(sample_n, len(scored_df)), random_state=random_state)
    hover = [c for c in ["safe_probability", "safety_margin", "predicted_value"]
             if c in scored_df.columns]

    if color_col == "pred_class" and class_names:
        cmap = build_class_color_map(class_names)
        fig = px.scatter_3d(
            sample,
            x=x_feat, y=y_feat, z=z_feat,
            color=color_col,
            color_discrete_map=cmap,
            opacity=0.35,
            hover_data=hover,
            title=f"3D region: {x_feat}, {y_feat}, {z_feat}",
            category_orders={color_col: class_names},
        )
    else:
        fig = px.scatter_3d(
            sample,
            x=x_feat, y=y_feat, z=z_feat,
            color=color_col,
            color_continuous_scale="RdBu",
            opacity=0.35,
            hover_data=hover,
            title=f"3D region: {x_feat}, {y_feat}, {z_feat}",
        )

    fig.update_traces(marker=dict(size=3))
    return fig


def plot_bo_history(bo_df: pd.DataFrame, score_col: str) -> go.Figure:
    df = bo_df.copy()
    df["evaluation"] = range(1, len(df) + 1)
    df["best_so_far"] = df[score_col].cummax()
    fig = px.line(df, x="evaluation", y="best_so_far",
                  title="Bayesian Optimisation Convergence",
                  labels={"evaluation": "Evaluation #", "best_so_far": f"Best {score_col}"})
    fig.add_scatter(x=df["evaluation"], y=df[score_col],
                    mode="markers", opacity=0.4, name="Each evaluation")
    return fig
