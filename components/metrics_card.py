"""
components/metrics_card.py
Reusable Streamlit widget: display a metrics dict as styled st.metric cards.

Usage
-----
from components.metrics_card import render_metrics_card

render_metrics_card({"accuracy": 0.94, "f1": 0.93, "roc_auc": 0.97})
render_metrics_card({"r2": 0.87, "rmse": 1.23, "mae": 0.95}, task_type="regression")
"""

from __future__ import annotations
import streamlit as st


# Which metrics are "higher is better" (others assumed lower = better)
_HIGHER_IS_BETTER = {"accuracy", "f1", "precision", "recall", "roc_auc", "r2"}

# Human-readable labels
_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC AUC",
    "r2": "R²",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
}

# Scalar metrics to show as st.metric (exclude complex objects)
_SCALAR_TYPES = (int, float)


def render_metrics_card(
    metrics: dict,
    task_type: str = "classification",
    previous_metrics: dict | None = None,
    columns: int = 4,
) -> None:
    """
    Render numeric metrics as a responsive card grid.

    Parameters
    ----------
    metrics          : Dict of metric_name → value (floats / ints only shown).
    task_type        : "classification" | "regression" — controls ordering.
    previous_metrics : Optional previous run's metrics for delta display.
    columns          : Number of columns in the card grid.
    """
    # Filter to scalar metrics only
    scalar_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, _SCALAR_TYPES) and k not in ("confusion_matrix",)
    }

    if not scalar_metrics:
        st.info("No scalar metrics available to display.")
        return

    # Preferred display order
    if task_type == "classification":
        order = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    else:
        order = ["r2", "rmse", "mae", "mse"]

    # Sort: preferred order first, then alphabetical for any extras
    sorted_keys = [k for k in order if k in scalar_metrics]
    sorted_keys += sorted(k for k in scalar_metrics if k not in order)

    # Render grid
    cols = st.columns(min(columns, len(sorted_keys)))
    for i, key in enumerate(sorted_keys):
        value = scalar_metrics[key]
        label = _LABELS.get(key, key.upper())
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)

        delta = None
        if previous_metrics and key in previous_metrics:
            prev = previous_metrics[key]
            if isinstance(prev, _SCALAR_TYPES):
                diff = value - prev
                delta = f"{diff:+.4f}"

        with cols[i % columns]:
            st.metric(label=label, value=formatted, delta=delta)
