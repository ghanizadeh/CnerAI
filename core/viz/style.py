"""
core/viz/style.py
Shared colour palette and matplotlib/seaborn style helpers.
"""

import io
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#4F8EF7",
    "secondary": "#F77F4F",
    "success": "#4FBF7F",
    "danger": "#F74F4F",
    "neutral": "#AAAAAA",
}

SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "coolwarm"
CATEGORICAL_PALETTE = "tab10"


def apply_default_style() -> None:
    """Apply consistent seaborn / matplotlib defaults for the session."""
    sns.set_theme(style="whitegrid", palette=CATEGORICAL_PALETTE)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#DDDDDD",
            "grid.color": "#EEEEEE",
            "font.size": 11,
        }
    )


def fig_to_st(fig: plt.Figure, caption: str = "") -> None:
    """Render a matplotlib figure in Streamlit and close it."""
    st.pyplot(fig)
    if caption:
        st.caption(caption)
    plt.close(fig)


def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 150) -> bytes:
    """Convert a figure to bytes for download buttons."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
