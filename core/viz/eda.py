"""
core/viz/eda.py
Pure-Python EDA charting functions that return matplotlib figures.
No Streamlit calls here — pass the figure to fig_to_st() in the page.

Public API
----------
draw_correlation_heatmap(df, columns)     -> plt.Figure
draw_boxplots(df, columns)               -> plt.Figure
draw_histograms(df, columns, bins)       -> plt.Figure
draw_scatter(df, x_col, y_col, hue_col) -> plt.Figure
draw_pairplot(df, columns, hue_col)     -> plt.Figure
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from core.viz.style import DIVERGING_CMAP, CATEGORICAL_PALETTE, apply_default_style


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    target: str | None = None,
    top_n: int = 5,
    corr_threshold: float = 0.85,
    figsize: tuple = (10, 8),
):
    """
    Draw correlation heatmap and compute feature recommendations.

    Returns
    -------
    fig : matplotlib.figure.Figure
    corr : pd.DataFrame
        Correlation matrix
    recommended : pd.DataFrame | None
        Top correlated features with target
    high_corr_pairs : pd.DataFrame
        Highly correlated feature pairs
    """

    apply_default_style()

    cols = columns or df.select_dtypes(include="number").columns.tolist()
    corr = df[cols].corr()

    # ------------------------------
    # Heatmap
    # ------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_CMAP,
        center=0,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Correlation Heatmap", fontsize=14, pad=12)
    fig.tight_layout()

    # ------------------------------
    # Feature recommendation
    # ------------------------------
    recommended = None

    if target is not None and target in corr.columns:

        target_corr = (
            corr[target]
            .drop(target)
            .abs()
            .sort_values(ascending=False)
            .head(top_n)
        )

        recommended = pd.DataFrame({
            "Feature": target_corr.index,
            "Correlation |r|": target_corr.values
        })

    # ------------------------------
    # Multicollinearity detection
    # ------------------------------
    high_corr_pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            val = corr.iloc[i, j]

            if abs(val) >= corr_threshold:

                high_corr_pairs.append({
                    "Feature 1": cols[i],
                    "Feature 2": cols[j],
                    "Correlation": val
                })

    high_corr_pairs = pd.DataFrame(high_corr_pairs)

    return fig, corr, recommended, high_corr_pairs


def draw_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    figsize: tuple | None = None,
) -> plt.Figure:
    """Side-by-side boxplots for numeric columns."""
    apply_default_style()
    n = len(columns)
    # dynamic width and height
    fw = max(6, n * 0.9)
    fh = max(5, n * 0.35)
    fig, ax = plt.subplots(figsize=figsize or (fw, fh))
    sns.boxplot(
        data=df[columns],
        orient="v",
        ax=ax,
        palette=CATEGORICAL_PALETTE
    )
    ax.set_title("Boxplots of Numeric Features", fontsize=13)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    # better label handling
    ax.set_xticklabels(columns, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def draw_histograms(
    df: pd.DataFrame,
    columns: list[str],
    bins: int = 30,
    ncols: int = 3,
    kde: bool = True,
) -> plt.Figure:
    """Grid of histograms with optional KDE."""
    
    import seaborn as sns
    
    apply_default_style()
    
    n = len(columns)
    nrows = max(1, -(-n // ncols))  # ceiling division
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, col in zip(axes, columns):

        sns.histplot(
            df[col].dropna(),
            bins=bins,
            kde=kde,
            color="#4F8EF7",
            edgecolor="white",
            ax=ax
        )

        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def draw_scatteqr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Scatter plot with optional colour grouping."""
    apply_default_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, alpha=0.7)
    ax.set_title(f"{x_col} vs {y_col}", fontsize=13)
    fig.tight_layout()
    return fig

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:

    apply_default_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Drop rows where x or y are NaN and coerce to numeric so matplotlib
    # never hits its category-axis path (which rejects float NaN values).
    cols_needed = list(dict.fromkeys([x_col, y_col] + ([hue_col] if hue_col else [])))
    plot_df = df[cols_needed].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col].squeeze(), errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col].squeeze(), errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    # ------------------------------------------------
    # If hue column exists → use heat color scale
    # ------------------------------------------------
    if hue_col is not None:
        hue_vals = pd.to_numeric(plot_df[hue_col], errors="coerce")
        valid = hue_vals.notna()
        sc = ax.scatter(
            plot_df.loc[valid, x_col],
            plot_df.loc[valid, y_col],
            c=hue_vals[valid],
            cmap="viridis",
            alpha=0.8,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(hue_col)

    # ------------------------------------------------
    # No hue → normal scatter
    # ------------------------------------------------
    else:
        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            alpha=0.7,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}", fontsize=13)

    fig.tight_layout()

    return fig
def draw_pairplot(
    df: pd.DataFrame,
    columns: list[str],
    hue_col: str | None = None,
) -> plt.Figure:
    """Seaborn pairplot — returns the underlying Figure."""
    apply_default_style()
    plot_df = df[columns + ([hue_col] if hue_col else [])].dropna()
    g = sns.pairplot(plot_df, hue=hue_col, corner=True, diag_kind="kde")
    g.fig.suptitle("Pair Plot", y=1.01, fontsize=14)
    return g.fig

 
def draw_pairwise_scatter_with_hist(
    df: pd.DataFrame,
    target_col: str,
    show_trendline: bool = True,
) -> list[plt.Figure]:
    """
    For each numeric feature, produce a marginal-histogram scatter plot
    showing its relationship with target_col.
 
    Each figure contains:
      - Main panel : scatter of feature vs target (+ optional R² trendline)
      - Top panel  : histogram of the feature with bar-count labels
      - Right panel: horizontal histogram of the target with bar-count labels
 
    Parameters
    ----------
    df             : DataFrame containing all columns.
    target_col     : Name of the target / y-axis column.
    show_trendline : Overlay a red dashed regression line and R² annotation.
 
    Returns
    -------
    List of matplotlib Figures — one per numeric feature (excluding target_col).
    Call fig_to_st(fig) on each one inside your Streamlit page.
    """
    apply_default_style()
 
    numeric_cols = df.select_dtypes(include="number").columns.drop(target_col, errors="ignore")
    cmap = plt.colormaps["Set1"].resampled(max(len(numeric_cols), 1))
    plots: list[plt.Figure] = []
 
    for i, col in enumerate(numeric_cols):
        data = df[[col, target_col]].dropna()
        x = data[col]
        y = data[target_col]
 
        if len(data) < 2:
            continue
 
        # ── Regression stats ──────────────────────────────────────────
        try:
            corr = np.corrcoef(x, y)[0, 1]
            r2 = corr ** 2
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
        except np.linalg.LinAlgError:
            continue
 
        color = cmap(i)
 
        # ── Figure + GridSpec ─────────────────────────────────────────
        fig = plt.figure(figsize=(7, 7))
        grid = plt.GridSpec(4, 4, hspace=0.05, wspace=0.05)
        ax_main  = fig.add_subplot(grid[1:4, 0:3])
        ax_xhist = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
        ax_yhist = fig.add_subplot(grid[1:4, 3], sharey=ax_main)
 
        # ── Main scatter ──────────────────────────────────────────────
        ax_main.scatter(
            x, y,
            color=color, alpha=0.7,
            edgecolor="black", linewidth=0.5,
        )
        if show_trendline:
            ax_main.plot(
                line_x, line_y,
                color="red", linestyle="--", linewidth=2,
                label=f"R² = {r2:.3f}",
            )
            ax_main.legend(loc="upper left", fontsize=8)
 
        ax_main.set_xlabel(col, fontsize=9)
        ax_main.set_ylabel(target_col, fontsize=9)
 
        # ── Top histogram (feature / x-axis) ─────────────────────────
        counts_x, _, patches_x = ax_xhist.hist(
            x, bins=15, color="steelblue", edgecolor="black"
        )
        for count, patch in zip(counts_x, patches_x):
            if count > 0:
                ax_xhist.text(
                    patch.get_x() + patch.get_width() / 2,
                    count,
                    int(count),
                    ha="center", va="bottom", fontsize=7,
                )
        ax_xhist.tick_params(labelbottom=False)
        ax_xhist.spines["top"].set_visible(False)
        ax_xhist.spines["right"].set_visible(False)
        ax_xhist.spines["left"].set_visible(False)
        ax_xhist.set_ylabel("Count", fontsize=7)
 
        # ── Right histogram (target / y-axis) ────────────────────────
        counts_y, _, patches_y = ax_yhist.hist(
            y, bins=15, orientation="horizontal",
            color="steelblue", edgecolor="black",
        )
        for count, patch in zip(counts_y, patches_y):
            if count > 0:
                ax_yhist.text(
                    count,
                    patch.get_y() + patch.get_height() / 2,
                    int(count),
                    va="center", ha="left", fontsize=7,
                )
        ax_yhist.tick_params(labelleft=False)
        ax_yhist.spines["right"].set_visible(False)
        ax_yhist.spines["top"].set_visible(False)
        ax_yhist.spines["bottom"].set_visible(False)
        ax_yhist.set_xlabel("Count", fontsize=7)
 
        # ── Title ─────────────────────────────────────────────────────
        ax_xhist.set_title(f"{col}  ↔  {target_col}", fontsize=11, pad=8)
 
        plots.append(fig)
 
    return plots