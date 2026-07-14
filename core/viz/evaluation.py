"""
core/viz/evaluation.py
Evaluation chart functions — return matplotlib figures, no Streamlit.

Public API
----------
draw_confusion_matrix(y_true, y_pred, labels)       -> plt.Figure
draw_roc_curve(y_true, y_prob, class_names)         -> plt.Figure
draw_residuals(y_true, y_pred)                      -> plt.Figure
draw_pred_vs_actual(y_true, y_pred)                 -> plt.Figure
draw_feature_importance(importance_df, top_n)       -> plt.Figure
draw_learning_curve(model, X, y, cv, scoring)       -> plt.Figure
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold
from core.viz.style import DIVERGING_CMAP, apply_default_style
from config.settings import RANDOM_STATE, DEFAULT_CV_FOLDS


def draw_confusion_matrix(
    y_true,
    y_pred,
    labels: list | None = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    apply_default_style()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    fig.tight_layout()
    return fig


def draw_roc_curve(
    y_true,
    y_prob,
    class_names: list[str] | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Binary or multiclass (one-vs-rest) ROC curve."""
    apply_default_style()
    fig, ax = plt.subplots(figsize=figsize)
    y_prob = np.array(y_prob)

    if y_prob.ndim == 1 or y_prob.shape[1] == 2:
        # Binary
        probs = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    else:
        # Multiclass — one curve per class
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            label = class_names[i] if class_names else str(cls)
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def draw_residuals(
    y_true,
    y_pred,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Residuals vs predicted values scatter plot."""
    apply_default_style()
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_pred, residuals, alpha=0.5, color="#4F8EF7", edgecolors="none")
    ax.axhline(0, color="#F74F4F", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    return fig


def draw_pred_vs_actual(
    y_true,
    y_pred,
    figsize: tuple = (6, 6),
) -> plt.Figure:
    """Perfect-prediction diagonal + scatter of actual vs predicted."""
    apply_default_style()
    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.5, color="#4F8EF7", edgecolors="none")
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    return fig


def draw_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    apply_default_style()
    df = importance_df.head(top_n).sort_values("Importance")
    fw = figsize or (8, max(4, len(df) * 0.35))
    fig, ax = plt.subplots(figsize=fw)
    ax.barh(df["Feature"], df["Importance"], color="#4F8EF7")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    return fig


def draw_learning_curve(
    model,
    X,
    y,
    cv: int = DEFAULT_CV_FOLDS,
    scoring: str = "accuracy",
    task_type: str = "classification",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Train vs validation learning curve across different training sizes."""
    apply_default_style()
    splitter = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        if task_type == "classification"
        else KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    )
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=splitter, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sizes, train_mean, "o-", color="#4F8EF7", label="Train score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#4F8EF7")
    ax.plot(train_sizes, val_mean, "o-", color="#F77F4F", label="Val score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#F77F4F")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel(scoring.capitalize())
    ax.set_title("Learning Curve")
    ax.legend()
    fig.tight_layout()
    return fig
