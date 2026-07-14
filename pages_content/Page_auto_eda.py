"""
foam_eda_app.py
================
Standalone Streamlit app for foam half-life EDA scatter plots.

Run:
    streamlit run foam_eda_app.py

Dependencies:
    pip install streamlit pandas numpy matplotlib seaborn openpyxl
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from itertools import product as iproduct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st
from utils.data_filter import render_data_filters

# ─────────────────────────────────────────────────────────────────────────────
#  Group taxonomy (mirrors foam_feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    "Nanoparticle": [
        "HS (%)", "BLH5 (%)", "HSA (%)",
    ],
    "Anionic": [
        "AOS (%)", "alpha-step (%)", "a-step (%)", "SDS (%)", "SLES (%)",
        "n. LABS (%)", "DB45 (%)", "Cola SLAA (%)", "Cola SC (%)",
    ],
    "Nonionic": [
        "APG (%)", "decyl glucoside (%)", "caprylyl glucoside (%)",
        "Tween 80 (%)", "PG (%)", "LAO (%)",
    ],
    "Zwitterionic": [
        "CapB (%)", "2C (%)", "Cola 2C (%)", "amine oxide (%)",
        "Cola LMB (%)", "Amphosol 1c (%)", "SC (%)", "LBHP (%)", "CS50(%)", "DM (%)",
    ],
    "Polymer": [
        "HPAM (%)", "xanthan gum (%) ", "Guar Gum (%)", "FPAM (%)",
        "PAA (%)", "PA (%)", "ClearHib 1000 (%)",
    ],
    "Citric": [
        "Citric (%)",
        "31.2(%) citric+ 13.3(%) KOH  (pH not adjusted, pH=4.46 )",
        "31.2% citric+ 13.3% KOH  (adjusted pH=4) (%)",
        "38.1% citric+   KOH  (pH=5) (%)",
        "31.2% citric+ 13.3% KOH (not adjusted pH) (%)",
        "31.2% citric+ 13.3% KOH (pH=4) (%)",

        "potassium citrate (9.7%)/citric acid buffer (19.22%)  pH=3 (%)",
        "potassium citrate (%)",
    ],
    "Acid": [
        "EDTA (%)", "etidronic acid (%)", "acetic acid (%)",
    ],
    "Antiscalant": [
        "Mem 2000-clear tech (%)", "Mem 2500-clear tech (%)",
        "Mem 4000-clear tech (%)", "Mem 3500-clear tech (%)",
        "Mem 3000-clear tech (%)",
    ],
    "Brine": [
        "Divalent", "Monovalent",
    ],
    "Oil": [
        "Linear paraffins", "Branched paraffins", "Cycloparaffins",
        "Aromatics", "Oxygenates", "Biomarker-like compounds",
        "Sulfur compounds", "Wax fraction (C24+)",
    ],
    "Process": [
        "Temperature", "Dilution Ratio", "Oil (%)",
        "concentrate manufacturing method (Ratio)",
        "Initial Foam Temp (dilution Temp) ",
    ],
}

# Sum feature definitions: name → list of raw columns to sum
SUM_FEATURES: dict[str, str] = {
    "Anionic (All Types)":      "Anionic",
    "Nonionic (All Types)":     "Nonionic",
    "Zwitterionic (All Types)": "Zwitterionic",
    "Sum Surfactant":           "Surfactant",   # special: Anionic+Nonionic+Zwitterionic
    "Nanoparticle (All Types)": "Nanoparticle",
    "Polymer (All Types)":      "Polymer",
    "Acid (All Types)":         "Acid",
    "Citric (All Types)":       "Citric",
}

GRP_CLR = {
    "Nanoparticle": "#1565C0", "Anionic":    "#E53935",
    "Nonionic":     "#FB8C00", "Zwitterionic":"#8E24AA",
    "Surfactant":   "#FF8F00", "Polymer":    "#6A1B9A",
    "Citric":       "#2E7D32", "Acid":       "#C62828",
    "Antiscalant":  "#00695C", "Brine":      "#4527A0",
    "Oil":          "#BF360C", "Process":    "#546E7A",
    "Sum":          "#37474F",
}

TARGET_CANDIDATES = [
    "Half life (method: poly) [h]",
    "Half life (method: poly) [h] 95%",
    "Calculated Half Life (hr_Linear)",
    "Calculated Half Life (hr)",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _slugify(text: str) -> str:
    """Safe folder/file name from arbitrary text."""
    s = str(text).strip()
    s = re.sub(r'[^\w\s\-]', '', s)
    s = re.sub(r'[\s]+', '_', s)
    return s[:60]


def _fname(x_col: str, target: str, color_col: str) -> str:
    """
    Build file name in the format:
        X_{x}_Y_{target}_Color_{color}.png
    All parts are slugified so the name is filesystem-safe.
    """
    x_s   = _slugify(x_col)
    y_s   = _slugify(target)
    c_s   = _slugify(color_col)
    return f"X_{x_s}_Y_{y_s}_Color_{c_s}.png"


def _normalise(name: str) -> str:
    """Lowercase + strip trailing ' (%)' and whitespace for fuzzy matching."""
    s = str(name).strip().lower()
    s = re.sub(r'\s*\(%\)\s*$', '', s).strip()
    return s


def _avail(candidates: list[str], cols: list[str]) -> list[str]:
    """
    Return items from `cols` that fuzzy-match any candidate,
    case-insensitively and ignoring a trailing ' (%)'.
    e.g. candidate 'AOS (%)' matches column 'AOS' and vice-versa.
    `cols` is the list of ACTUAL DataFrame columns.
    `candidates` is the hardcoded default list for a group.
    """
    # Build lookup: normalised_actual_col → actual_col
    norm_to_col = {_normalise(c): c for c in cols}
    matched = []
    seen = set()
    for cand in candidates:
        key = _normalise(cand)
        if key in norm_to_col:
            actual = norm_to_col[key]
            if actual not in seen:
                matched.append(actual)
                seen.add(actual)
    return matched


def _resolve_col(name: str, df: pd.DataFrame) -> str | None:
    """Find the actual DataFrame column that matches `name` fuzzily."""
    if name in df.columns:
        return name
    key = _normalise(name)
    for col in df.columns:
        if _normalise(col) == key:
            return col
    return None


def compute_sum_features(df: pd.DataFrame,
                          sel_groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    Add sum feature columns to df (copy returned).
    Column lookup is fuzzy: case-insensitive, ignores trailing ' (%)'.
    Only adds a sum feature when at least one member column is found.
    """
    out = df.copy()
    for feat_name, group_key in SUM_FEATURES.items():
        if group_key == "Surfactant":
            cols = (sel_groups.get("Anionic", []) +
                    sel_groups.get("Nonionic", []) +
                    sel_groups.get("Zwitterionic", []))
        else:
            cols = sel_groups.get(group_key, [])
        # Resolve each column name fuzzily against the actual DataFrame
        resolved = [_resolve_col(c, out) for c in cols]
        present  = [c for c in resolved if c is not None]
        if present:
            s = pd.Series(0.0, index=out.index)
            for c in present:
                s += _safe_num(out[c]).fillna(0)
            out[feat_name] = s
    return out


def compute_ratio_features(df: pd.DataFrame,
                            ratios: list[tuple[str, str]]) -> pd.DataFrame:
    """Add user-defined ratio columns (numerator / denominator)."""
    out = df.copy()
    for num, den in ratios:
        if num in out.columns and den in out.columns:
            col_name = f"{num} / {den}"
            out[col_name] = _safe_num(out[num]) / _safe_num(out[den]).replace(0, np.nan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

CMAP = "viridis"
POINT_ALPHA = 0.72
POINT_SIZE  = 38
FIG_DPI     = 150


def _scatter(df: pd.DataFrame,
             x_col: str, y_col: str, color_col: str,
             title: str) -> plt.Figure:
    """
    Single 2D scatter: X = x_col, Y = y_col, colour = color_col.
    Returns a matplotlib Figure.
    """
    sub = df[[x_col, y_col, color_col]].dropna()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return fig

    x = sub[x_col].values
    y = sub[y_col].values
    c = sub[color_col].values

    # Colour: numeric → continuous cmap, categorical → discrete
    is_num_color = pd.api.types.is_numeric_dtype(sub[color_col])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    if is_num_color:
        sc = ax.scatter(x, y, c=c, cmap=CMAP, alpha=POINT_ALPHA,
                        s=POINT_SIZE, edgecolors="white", linewidths=0.4)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_col, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    else:
        cats = sub[color_col].astype(str)
        uniq = sorted(cats.unique())
        palette = plt.colormaps["tab10"].resampled(len(uniq))
        for i, cat in enumerate(uniq):
            mask = cats == cat
            ax.scatter(x[mask], y[mask], color=palette(i),
                       alpha=POINT_ALPHA, s=POINT_SIZE,
                       edgecolors="white", linewidths=0.4, label=cat)
        ax.legend(fontsize=7, title=color_col, title_fontsize=7,
                  loc="upper left", framealpha=0.7)

    ax.set_xlabel(x_col, fontsize=9)
    ax.set_ylabel(y_col, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#CCCCCC")
    fig.tight_layout()
    return fig


def _scatter_3d(df: pd.DataFrame,
                x_col: str, y_col: str, z_col: str,
                color_col: str, title: str) -> plt.Figure:
    """
    3D scatter.
    Axes mapping:  X = x_col (feature)
                   Y = y_col (feature)
                   Z = z_col (target — always vertical)
    Colour = color_col.
    """
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    needed = [x_col, y_col, z_col, color_col]
    sub = df[list(dict.fromkeys(needed))].dropna()

    fig = plt.figure(figsize=(9, 6), dpi=FIG_DPI)
    fig.patch.set_facecolor("#FAFAFA")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#F5F5F5")

    if sub.empty:
        ax.text(0.5, 0.5, 0.5, "No data", ha="center", va="center",
                fontsize=12, color="grey")
        ax.set_title(title, fontsize=9, fontweight="bold")
        fig.tight_layout()
        return fig

    xv = sub[x_col].values   # X axis — feature
    yv = sub[y_col].values   # Y axis — feature
    zv = sub[z_col].values   # Z axis — target (vertical)
    cv = sub[color_col].values

    is_num = pd.api.types.is_numeric_dtype(sub[color_col])
    if is_num:
        sc = ax.scatter(xv, yv, zv, c=cv, cmap=CMAP,
                        alpha=POINT_ALPHA, s=POINT_SIZE,
                        edgecolors="white", linewidths=0.3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label(color_col, fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    else:
        cats = sub[color_col].astype(str)
        uniq = sorted(cats.unique())
        palette = plt.colormaps["tab10"].resampled(len(uniq))
        for i, cat in enumerate(uniq):
            mask = cats == cat
            ax.scatter(xv[mask], yv[mask], zv[mask],
                       color=palette(i), alpha=POINT_ALPHA,
                       s=POINT_SIZE, edgecolors="white",
                       linewidths=0.3, label=cat)
        ax.legend(fontsize=6, title=color_col, title_fontsize=6,
                  loc="upper left", framealpha=0.7)

    ax.set_xlabel(x_col,  fontsize=7, labelpad=4)   # X = feature
    ax.set_ylabel(y_col,  fontsize=7, labelpad=4)   # Y = feature
    ax.set_zlabel(z_col,  fontsize=7, labelpad=4)   # Z = target
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def _fname_3d(x_col: str, y_col: str, z_col: str, color_col: str) -> str:
    """Build 3D plot file name."""
    return (f"X_{_slugify(x_col)}_Y_{_slugify(y_col)}"
            f"_Z_{_slugify(z_col)}_Color_{_slugify(color_col)}.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Plot generation — one function per subfolder type
# ─────────────────────────────────────────────────────────────────────────────

def _render_to_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_subfolder(
    df: pd.DataFrame,
    subfolder: str,
    x_cols: list[str],
    color_cols_per_x: dict[str, list[str]],
    target: str,
    prog_offset: int = 0,
    prog_total: int  = 1,
    prog_bar = None,
) -> dict[str, bytes]:
    """
    Core loop: for each x_col create one PNG per colour column.

    Parameters
    ----------
    subfolder          : top-level subfolder name inside "2D Plots/"
    x_cols             : list of X-axis feature names (→ one sub-folder each)
    color_cols_per_x   : {x_col: [colour_col, ...]}  — colour list per X
    prog_offset/total  : for combined progress tracking across subfolders
    """
    figures: dict[str, bytes] = {}
    done = 0
    for x_col in x_cols:
        x_folder = _slugify(x_col)
        for col_col in color_cols_per_x.get(x_col, []):
            title = (f"Y: {target}\n"
                     f"X: {x_col}   |   Colour: {col_col}")
            fig   = _scatter(df, x_col, target, col_col, title)
            fname = _fname(x_col, target, col_col)
            figures[f"{subfolder}/{x_folder}/{fname}"] = _render_to_bytes(fig)
            done += 1
            if prog_bar is not None:
                pct = min((prog_offset + done) / prog_total, 1.0)
                prog_bar.progress(pct,
                    text=f"[{subfolder[:30]}]  X={x_col}  colour={col_col}…")
    return figures


def generate_all_plots(
    df: pd.DataFrame,
    x_cols: list[str],
    color_cols: list[str],
    ratio_cols: list[str],
    oil_cols: list[str],
    target: str,
) -> dict[str, bytes]:
    """
    Generate all three plot subfolders and return {relative_path: png_bytes}.

    Folder structure
    ----------------
    2D Plots/
      Effect of individual component to each other on half life/
        AOS/
          X_AOS_Y_HalfLife_Color_APG.png
          X_AOS_Y_HalfLife_Color_Polarity.png
          ...

      Effect of component Ratio to individual components/
        AOS/
          X_AOS_Y_HalfLife_Color_Anionic_All_Types___Sum_Surfactant.png
          ...

      Effect of component Ratio and individual component to oil properties/
        AOS/
          X_AOS_Y_HalfLife_Color_Polarity.png
          X_AOS_Y_HalfLife_Color_Aromatics.png
          ...
        Anionic_All_Types___Sum_Surfactant/
          X_Anionic_..._Y_HalfLife_Color_Polarity.png
          ...

    Colour rules
    ------------
    SF1 — peers (other x_cols) + user colour-by features
    SF2 — all ratio_cols  (must-have + user-defined), self skipped
    SF3 — oil_cols only; X = individual components + ratio features
    """
    SF1 = "Effect of individual component to each other on Target"
    SF2 = "Effect of component Ratio to individual components"
    SF3 = "Effect of component Ratio and individual component to oil properties"

    # ── SF1 colour map: peers + extra colour-by ───────────────────────────
    sf1_colors: dict[str, list[str]] = {}
    for x in x_cols:
        peers  = [c for c in x_cols    if c != x]
        extras = [c for c in color_cols if c != x and c not in peers]
        sf1_colors[x] = list(dict.fromkeys(peers + extras))

    # ── SF2 colour map: ratio cols (self skipped) ─────────────────────────
    sf2_colors: dict[str, list[str]] = {}
    for x in x_cols:
        sf2_colors[x] = [r for r in ratio_cols if r != x]

    # ── SF3 colour map: oil cols; X = components + ratios ────────────────
    # X axes for SF3: all individual selected components + all ratio features
    sf3_x_cols = list(dict.fromkeys(x_cols + ratio_cols))
    sf3_colors: dict[str, list[str]] = {}
    for x in sf3_x_cols:
        # colour = all oil cols that differ from x
        sf3_colors[x] = [o for o in oil_cols if o != x]

    # ── Count total for progress bar ──────────────────────────────────────
    n_sf1 = sum(len(v) for v in sf1_colors.values())
    n_sf2 = sum(len(v) for v in sf2_colors.values())
    n_sf3 = sum(len(v) for v in sf3_colors.values())
    total = n_sf1 + n_sf2 + n_sf3

    if total == 0:
        st.warning("No plots to generate — check selections.")
        return {}

    prog = st.progress(0, text="Starting…")
    figures: dict[str, bytes] = {}

    # ── Subfolder 1 ──────────────────────────────────────────────────────
    f1 = _generate_subfolder(df, SF1, x_cols, sf1_colors, target,
                              prog_offset=0,
                              prog_total=total, prog_bar=prog)
    figures.update(f1)

    # ── Subfolder 2 ──────────────────────────────────────────────────────
    f2 = _generate_subfolder(df, SF2, x_cols, sf2_colors, target,
                              prog_offset=n_sf1,
                              prog_total=total, prog_bar=prog)
    figures.update(f2)

    # ── Subfolder 3 ──────────────────────────────────────────────────────
    if oil_cols:
        f3 = _generate_subfolder(df, SF3, sf3_x_cols, sf3_colors, target,
                                  prog_offset=n_sf1 + n_sf2,
                                  prog_total=total, prog_bar=prog)
        figures.update(f3)

    prog.empty()
    return figures


def _build_3d_folder_map(
    xz_features: list[str],
    color_features: list[str],
    seen_global: set,
) -> dict[str, list[tuple[str, str]]]:
    """
    Build {x_col: [(z_col, c_col), ...]} for unique unordered triplets.

    Parameters
    ----------
    xz_features   : pool for X and Z axes
    color_features: pool for Colour axis
    seen_global   : shared frozenset cache — pass the same set across
                    multiple subfolders to avoid cross-folder duplicates

    Deduplication
    -------------
    frozenset({x, z, c}) — order of X/Z/Color never matters.
    Each unique combination is plotted exactly once across ALL subfolders
    that share the same seen_global cache.
    """
    folder_map: dict[str, list[tuple[str, str]]] = {}
    for x_col in xz_features:
        for z_col in xz_features:
            if z_col == x_col:
                continue
            for c_col in color_features:
                if c_col in (x_col, z_col):
                    continue
                key = frozenset([x_col, z_col, c_col])
                if key in seen_global:
                    continue
                seen_global.add(key)
                folder_map.setdefault(x_col, []).append((z_col, c_col))
    return folder_map


def _render_3d_subfolder(
    df: pd.DataFrame,
    subfolder: str,
    folder_map: dict[str, list[tuple[str, str]]],
    target: str,
    prog_bar,
    prog_offset: int,
    prog_total: int,
) -> tuple[dict[str, bytes], int]:
    """Render all plots in one 3D subfolder. Returns (figures, done_count)."""
    figures: dict[str, bytes] = {}
    done = 0
    for x_col, zc_pairs in folder_map.items():
        x_folder = _slugify(x_col)
        for z_col, c_col in zc_pairs:
            title = (f"Z (vertical): {target}  |  X: {x_col}  |  "
                     f"Y: {z_col}  |  Colour: {c_col}")
            fig   = _scatter_3d(df, x_col, z_col, target, c_col, title)
            fname = _fname_3d(x_col, target, z_col, c_col)
            figures[f"{subfolder}/{x_folder}/{fname}"] = _render_to_bytes(fig)
            done += 1
            if prog_bar and prog_total > 0:
                prog_bar.progress(
                    min((prog_offset + done) / prog_total, 1.0),
                    text=f"3D [{subfolder[:25]}]  X={x_col}  Z={z_col}  C={c_col}…",
                )
    return figures, done


def generate_3d_plots(
    df: pd.DataFrame,
    feature_cols: list[str],       # individual components + sum features (SF1 & SF2)
    ratio_cols: list[str],         # ratio features (SF2 colour)
    oil_cols: list[str],           # oil properties (SF3 colour only)
    target: str,
) -> dict[str, bytes]:
    """
    Generate 3D scatter plots across three subfolders mirroring 2D structure.

    Folder structure
    ----------------
    3D Plots/
      Effect of individual component to each other on half life/
        AOS/
          X_AOS_Y_Target_Z_APG_Color_Sum_Surfactant.png
          X_AOS_Y_Target_Z_APG_Color_Anionic_All_Types.png
          ...
        APG/ ...

      Effect of component Ratio to individual components/
        AOS/
          X_AOS_Y_Target_Z_APG_Color_Anionic_All_Types_Sum_Surfactant.png
          ...

      Effect of component Ratio and individual component to oil properties/
        AOS/
          X_AOS_Y_Target_Z_APG_Color_Oil_Polarity.png
          ...

    SF1 — X/Z: feature_cols,  Colour: feature_cols
    SF2 — X/Z: feature_cols,  Colour: ratio_cols    (ratios only as colour)
    SF3 — X/Z: feature_cols,  Colour: oil_cols      (oil only as colour)

    Deduplication
    -------------
    Each subfolder maintains its own frozenset cache so a unique (X,Z,C)
    combination is plotted once per subfolder. Across subfolders the colour
    pools differ so the same (X,Z) pair may appear in multiple subfolders
    with different colour columns — that is correct and expected.
    """
    SF1 = "Effect of individual component to each other on half life"
    SF2 = "Effect of component Ratio to individual components"
    SF3 = "Effect of component Ratio and individual component to oil properties"

    feat = list(dict.fromkeys(feature_cols))

    # Build folder maps — each with its own seen cache (different colour pools)
    seen1: set = set()
    fm1 = _build_3d_folder_map(feat, feat, seen1)          # colours = features

    seen2: set = set()
    ratio_colors = list(dict.fromkeys(ratio_cols))
    fm2 = _build_3d_folder_map(feat, ratio_colors, seen2)  # colours = ratios

    seen3: set = set()
    oil_colors = list(dict.fromkeys(oil_cols))
    fm3 = _build_3d_folder_map(feat, oil_colors, seen3)    # colours = oil

    n1 = sum(len(v) for v in fm1.values())
    n2 = sum(len(v) for v in fm2.values())
    n3 = sum(len(v) for v in fm3.values())
    total = n1 + n2 + n3

    if total == 0:
        st.info("No 3D plots to generate — check feature and colour selections.")
        return {}

    figures: dict[str, bytes] = {}
    prog = st.progress(0, text="Generating 3D plots…")
    offset = 0

    f1, d1 = _render_3d_subfolder(df, SF1, fm1, target, prog, offset, total)
    figures.update(f1); offset += d1

    f2, d2 = _render_3d_subfolder(df, SF2, fm2, target, prog, offset, total)
    figures.update(f2); offset += d2

    f3, d3 = _render_3d_subfolder(df, SF3, fm3, target, prog, offset, total)
    figures.update(f3)

    prog.empty()
    return figures


def figures_to_zip(figures: dict[str, bytes]) -> bytes:
    """Pack pre-rendered PNG bytes into an in-memory ZIP.
    Path inside ZIP: 2D Plots/<subfolder>/<x_folder>/<filename>.png
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path_key, png_bytes in figures.items():
            zf.writestr(path_key, png_bytes)   # keys already include root folder
    buf.seek(0)
    return buf.read()


def save_figures_to_disk(figures: dict[str, bytes],
                          base_dir: str) -> int:
    """Save pre-rendered PNG bytes.
    Output: base_dir/{2D Plots|3D Plots}/<subfolder>/<x_folder>/<filename>.png
    Keys already include the root folder prefix (forward-slash separated).
    Uses os.makedirs with explicit string join for Windows compatibility.
    """
    count = 0
    for path_key, png_bytes in figures.items():
        # Split on forward slash — works on both Windows and Linux
        parts = path_key.split("/")
        dest_dir = os.path.join(base_dir, *parts[:-1])
        dest_file = os.path.join(dest_dir, parts[-1])
        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_file, "wb") as fh:
            fh.write(png_bytes)
        count += 1
    return count


def render():

    # ─────────────────────────────────────────────────────────────────────────────
    #  Streamlit app
    # ─────────────────────────────────────────────────────────────────────────────
    #st.header("🧪 Foam EDA — Scatter Plot Generator")
    

    # ── Custom style ─────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Lora:wght@500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Lora', Georgia, serif; }
    h1 { font-family: 'Lora', serif; font-weight: 700; letter-spacing: -0.5px; }
    h2, h3 { font-family: 'Lora', serif; font-weight: 500; }
    code, .stCode { font-family: 'DM Mono', monospace !important; }
    .group-pill {
        display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:0.75rem; font-weight:600; margin:2px;
        font-family:'DM Mono',monospace;
    }
    .section-card {
        background:#f8f9fa; border:1px solid #e0e0e0; border-radius:10px;
        padding:1.2rem 1.5rem; margin-bottom:1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧪 Foam EDA — 2D & 3D Scatter Plot Generator")
    st.caption("Upload data → select components → generate 2D and/or 3D organised scatter plots.")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 1 — Upload
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 1 · Upload Data")
    uploaded = st.file_uploader(
        "CSV (UTF-8) or Excel (.xlsx / .xls)",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded is None:
        st.info("Upload a CSV or Excel file to continue.")
        st.stop()

    is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))

    # Sheet selector (Excel only)
    sheet_name: str | int = 0
    if is_excel:
        file_bytes_peek = uploaded.read()
        uploaded.seek(0)
        try:
            xl = pd.ExcelFile(io.BytesIO(file_bytes_peek))
            sheets = xl.sheet_names
            if len(sheets) > 1:
                sheet_name = st.selectbox(
                    "Select sheet",
                    options=sheets,
                    index=0,
                    key="sheet_selector",
                )
            else:
                sheet_name = sheets[0]
                st.caption(f"Single sheet detected: **{sheet_name}**")
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            st.stop()

    @st.cache_data(show_spinner="Reading file…")
    def _load(file_bytes: bytes, excel: bool, sheet: str | int) -> pd.DataFrame:
        if excel:
            return (pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                    .dropna(how="all"))
        # Try common encodings before giving up (Excel-exported CSVs are
        # often Windows-1252, not UTF-8, and raise UnicodeDecodeError otherwise).
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return (pd.read_csv(io.BytesIO(file_bytes), encoding=enc,
                                    low_memory=False)
                        .dropna(how="all"))
            except UnicodeDecodeError:
                continue
        return (pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1",
                            low_memory=False)
                .dropna(how="all"))

    raw_bytes = uploaded.read()
    df_raw = _load(raw_bytes, is_excel, sheet_name)
    st.success(f"✅  **{len(df_raw):,} rows × {df_raw.shape[1]} columns**")

    with st.expander("Preview raw data (first 5 rows)"):
        st.dataframe(df_raw.head(5), use_container_width=True)

    all_cols = list(df_raw.columns)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 2 — Data Filtering
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 2 · Data Filtering")
    st.caption(
        "Filter rows before plotting. Filters are non-destructive — adjust any time."
    )
    df_raw = render_data_filters(df_raw, key_prefix="eda")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 3 — Target
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 3 · Select Target (Y axis)")
    default_target = next((c for c in TARGET_CANDIDATES if c in all_cols), all_cols[-1])
    target_col = st.selectbox("Target column", all_cols,
                            index=all_cols.index(default_target))
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 4 — Component groups (auto-populated from defaults, user can edit)
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 4 · Chemical Component Groups")
    st.caption(
        "Default columns for each group are pre-selected from your uploaded file. "
        "Add or remove columns freely."
    )

    sel_groups: dict[str, list[str]] = {}
    non_target = [c for c in all_cols if c != target_col]

    # Show groups in a 2-column grid
    grp_items = [g for g in GROUPS if g not in ("Oil", "Process", "Brine")]
    oil_proc   = ["Oil", "Brine"]  # Process handled separately below

    with st.container():
        cols_ui = st.columns(2)
        for gi, grp in enumerate(grp_items):
            with cols_ui[gi % 2]:
                clr = GRP_CLR.get(grp, "#607D8B")
                st.markdown(
                    f'<span class="group-pill" style="background:{clr}22;color:{clr};'
                    f'border:1px solid {clr}66">{grp}</span>',
                    unsafe_allow_html=True,
                )
                defaults = _avail(GROUPS[grp], non_target)
                sel_groups[grp] = st.multiselect(
                    f"Columns for {grp}",
                    options=non_target,
                    default=defaults,
                    key=f"grp_{grp}",
                    label_visibility="collapsed",
                )

    st.markdown("##### 🌊 Brine & 🛢️ Oil")
    bp_cols = st.columns(2)
    for ci, grp in enumerate(["Oil", "Brine"]):
        with bp_cols[ci]:
            clr = GRP_CLR.get(grp, "#607D8B")
            st.markdown(
                f'<span class="group-pill" style="background:{clr}22;color:{clr};'
                f'border:1px solid {clr}66">{grp}</span>',
                unsafe_allow_html=True,
            )
            defaults = _avail(GROUPS[grp], non_target)
            sel_groups[grp] = st.multiselect(
                f"Columns for {grp}",
                options=non_target,
                default=defaults,
                key=f"grp_{grp}",
                label_visibility="collapsed",
            )

    # ── Process — individual pickers per variable ─────────────────────────────────
    st.markdown("##### ⚙️ Process Conditions")
    st.caption("Select the column for each process variable. '(none)' to exclude.")
    NONE = "(none)"
    none_opt = [NONE] + non_target

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        _def_temp = next((c for c in non_target
                        if re.sub(r"\s+", " ", _normalise(c)).strip()
                        in ("temperature", "temperature corrected",
                            "temperature_corrected", "temp")), NONE)
        proc_temp = st.selectbox(
            "🌡️ Temperature",
            none_opt,
            index=none_opt.index(_def_temp) if _def_temp in none_opt else 0,
            key="proc_temp",
        )

    with pc2:
        _def_dil = next((c for c in non_target
                        if re.sub(r"\s+", " ", _normalise(c)).strip()
                        in ("dilution ratio", "dilution ratio corrected",
                            "dilution ratio_corrected", "dilution")), NONE)
        proc_dil = st.selectbox(
            "💧 Dilution Ratio",
            none_opt,
            index=none_opt.index(_def_dil) if _def_dil in none_opt else 0,
            key="proc_dil",
        )

    with pc3:
        # _normalise already strips trailing (%) and lowercases;
        # also collapse all internal whitespace for "oil  (%)" → "oil"
        _def_oilpct = next((c for c in non_target
                            if re.sub(r"\s+", " ", _normalise(c)).strip()
                            in ("oil", "oil percent", "oil pct", "oil%",
                                "oil  pct", "oil  (%)", "oil (%)")), NONE)
        proc_oil_pct = st.selectbox(
            "🛢️ Oil Percent (%)",
            none_opt,
            index=none_opt.index(_def_oilpct) if _def_oilpct in none_opt else 0,
            key="proc_oil_pct",
        )

    # Collect non-none process cols into sel_groups["Process"]
    sel_groups["Process"] = [c for c in [proc_temp, proc_dil, proc_oil_pct]
                            if c != NONE]
    # ── Warn about % columns not assigned to any group ────────────────────────
    _all_selected = set(c for cols in sel_groups.values() for c in cols)
    _pct_unselected = [c for c in non_target
                       if "%" in str(c) and c not in _all_selected]
    if _pct_unselected:
        st.warning(
            f"⚠️ **{len(_pct_unselected)} column(s) containing '%' are not assigned "
            f"to any group** and will be excluded from the model:\n\n"
            + "\n".join(f"- `{c}`" for c in _pct_unselected)
        )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 5 — Computed Features
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 5 · Computed Features")

    df_work     = compute_sum_features(df_raw, sel_groups)
    computed_sums = [f for f in SUM_FEATURES if f in df_work.columns]

    if computed_sums:
        pills = " ".join(
            f'<span class="group-pill" style="background:#e3f2fd;color:#1565C0;'
            f'border:1px solid #90caf9">{s}</span>'
            for s in computed_sums
        )
        st.markdown("**Auto-computed sums:**")
        st.markdown(pills, unsafe_allow_html=True)

    # Custom engineered features
    user_ratios: list[tuple[str, str, str]] = []   # (col1, op, col2)
    ratio_cols:  list[str] = []
    avail_for_ratio = list(dict.fromkeys(non_target + computed_sums))

    with st.expander("🧪 Custom Engineered Features", expanded=False):
        st.caption("Create new features by combining two columns with + or /")
        n_custom = st.number_input("Number of custom engineered features",
                                    0, 20, 0, 1, key="n_custom")
        for ri in range(int(n_custom)):
            st.markdown(f"**Custom Feature #{ri+1}**")
            cc1, cc2, cc3 = st.columns([5, 2, 5])
            c1_sel = cc1.selectbox("Component 1", avail_for_ratio, key=f"cf_c1_{ri}")
            op_sel = cc2.selectbox("Operation",   ["+", "/"],       key=f"cf_op_{ri}")
            c2_sel = cc3.selectbox("Component 2", avail_for_ratio, key=f"cf_c2_{ri}")
            user_ratios.append((c1_sel, op_sel, c2_sel))

    for _c1, _op, _c2 in user_ratios:
        _fname_cf = f"{_c1} {_op} {_c2}"
        if _c1 in df_work.columns and _c2 in df_work.columns and _fname_cf not in df_work.columns:
            _c1_num = _safe_num(df_work[_c1])
            _c2_num = _safe_num(df_work[_c2])
            df_work[_fname_cf] = (_c1_num + _c2_num if _op == "+"
                               else _c1_num / _c2_num.replace(0, float("nan")))
    ratio_cols = [f"{c1} {op} {c2}" for c1, op, c2 in user_ratios
                  if f"{c1} {op} {c2}" in df_work.columns]

    # ── Must-have ratio / sum features ───────────────────────────────────────────
    _oil_pct_col = proc_oil_pct if proc_oil_pct != NONE else None

    MUST_HAVE_RATIOS: list[tuple[str, str]] = [
        ("Anionic (All Types)",      "Sum Surfactant"),
        ("Nonionic (All Types)",     "Sum Surfactant"),
        ("Zwitterionic (All Types)", "Sum Surfactant"),
        ("Nanoparticle (All Types)", "Sum Surfactant"),
        ("Polymer (All Types)",      "Sum Surfactant"),
        ("Acid (All Types)",         "Sum Surfactant"),
        ("Citric (All Types)",       "Sum Surfactant"),
    ]
    if _oil_pct_col and _oil_pct_col in df_work.columns:
        MUST_HAVE_RATIOS += [
            (_oil_pct_col, "Sum Surfactant"),
            (_oil_pct_col, "Nanoparticle (All Types)"),
        ]

    EXCLUDED_IX = {
        "Anionic (All Types) + Nonionic (All Types)",
        "Anionic (All Types) + Zwitterionic (All Types)",
        "Nonionic (All Types) + Anionic (All Types)",
        "Nonionic (All Types) + Zwitterionic (All Types)",
        "Zwitterionic (All Types) + Anionic (All Types)",
        "Zwitterionic (All Types) + Nonionic (All Types)",
        "Anionic (All Types) / Nonionic (All Types)",
        "Anionic (All Types) / Zwitterionic (All Types)",
        "Nonionic (All Types) / Anionic (All Types)",
        "Nonionic (All Types) / Zwitterionic (All Types)",
        "Zwitterionic (All Types) / Anionic (All Types)",
        "Zwitterionic (All Types) / Nonionic (All Types)",
    }

    # ── Compute BOTH + and / for each pair, select winner by |correlation| ──
    _y_for_corr = _safe_num(df_work[target_col])
    _corr_report: list[dict] = []
    _best_cols:   list[str]  = []

    def _corr_with_target(s: pd.Series) -> float:
        both = s.notna() & _y_for_corr.notna()
        if both.sum() < 5:
            return 0.0
        return float(s[both].corr(_y_for_corr[both]))

    for _num, _den in MUST_HAVE_RATIOS:
        if _num == _den:
            continue
        if _num not in df_work.columns or _den not in df_work.columns:
            continue

        _sum_name   = f"{_num} + {_den}"
        _ratio_name = f"{_num} / {_den}"

        if _sum_name not in df_work.columns or _ratio_name not in df_work.columns:
            _num_vals = _safe_num(df_work[_num])
            _den_vals = _safe_num(df_work[_den])
        if _sum_name not in df_work.columns:
            df_work[_sum_name]   = _num_vals + _den_vals
        if _ratio_name not in df_work.columns:
            df_work[_ratio_name] = _num_vals / _den_vals.replace(0, float("nan"))

        _r_sum   = _corr_with_target(df_work[_sum_name])
        _r_ratio = _corr_with_target(df_work[_ratio_name])

        if abs(_r_sum) >= abs(_r_ratio):
            _winner, _loser   = _sum_name,   _ratio_name
            _r_win,  _r_lose  = _r_sum,      _r_ratio
        else:
            _winner, _loser   = _ratio_name, _sum_name
            _r_win,  _r_lose  = _r_ratio,    _r_sum

        if _winner not in EXCLUDED_IX:
            _best_cols.append(_winner)

        _corr_report.append({
            "Pair":        f"{_num}  ×  {_den}",
            "Winner":      _winner,
            "Winner r":    round(_r_win,  4),
            "Winner |r|":  round(abs(_r_win),  4),
            "Loser":       _loser,
            "Loser r":     round(_r_lose, 4),
            "Loser |r|":   round(abs(_r_lose), 4),
        })

    _user_feat_cols = [f"{c1} {op} {c2}" for c1, op, c2 in user_ratios
                       if f"{c1} {op} {c2}" in df_work.columns]

    all_ratio_cols = list(dict.fromkeys(
        [c for c in _best_cols if c not in EXCLUDED_IX] + _user_feat_cols
    ))

    if _corr_report:
        st.markdown("**Auto-computed ratio/sum features** — winner selected by |correlation| with target:")
        _rdf = pd.DataFrame(_corr_report)

        def _style_row(row):
            return [
                "background-color:#e8f5e9;font-weight:bold"
                if col in ("Winner","Winner r","Winner |r|")
                else "color:#aaaaaa"
                for col in row.index
            ]

        st.dataframe(
            _rdf.style.apply(_style_row, axis=1)
                .format({"Winner r": "{:+.4f}", "Winner |r|": "{:.4f}",
                         "Loser r":  "{:+.4f}", "Loser |r|":  "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        if all_ratio_cols:
            _ratio_pills = " ".join(
                f'<span class="group-pill" style="background:#fff8e1;color:#795548;'
                f'border:1px solid #bcaaa4">{c}</span>'
                for c in all_ratio_cols
            )
            st.markdown("**Selected features (winners + custom):**")
            st.markdown(_ratio_pills, unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 6 — X-axis columns and Colour columns
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 6 · Select Plot Axes")

    # Flatten all selected individual columns (from ALL groups including Oil/Brine/Process)
    individual_selected = list(dict.fromkeys(
        c for cols in sel_groups.values() for c in cols
    ))

    # Oil properties (always available as colour even if not in selected groups)
    oil_cols_in_data = _avail(GROUPS["Oil"], non_target)

    # Must-have X columns — always included when present in df_work
    MUST_HAVE_X = [
        "Anionic (All Types)",
        "Nonionic (All Types)",
        "Zwitterionic (All Types)",
        "Sum Surfactant",
        "Nanoparticle (All Types)",
        "Polymer (All Types)",
        "Acid (All Types)",
        "Citric (All Types)",
    ]
    must_have_present = [f for f in MUST_HAVE_X if f in df_work.columns]

    # X-axis: individual + must-have sums + all computed sums + ratios
    x_candidates = list(dict.fromkeys(
        individual_selected + must_have_present + computed_sums + ratio_cols
    ))

    # Colour: same as X plus any oil columns not already included
    color_candidates = list(dict.fromkeys(
        individual_selected + computed_sums + ratio_cols + oil_cols_in_data
    ))

    # ── 2D Plot Configuration ────────────────────────────────────────────────────
    st.markdown("##### 📊 2D Plot Configuration")

    with st.container(border=True):
        st.markdown("**X-axis features** (one folder per X axis will be created):")
        st.caption(
            "🔒 Sum features (Anionic, Nonionic, Zwitterionic, Sum Surfactant, "
            "Nanoparticle, Polymer, Acid, Citric) are always included as X axes "
            "when computable — you can add or remove individual columns below."
        )

        _X_DEFAULTS = ["APG (%)", "AOS (%)", "CapB (%)", "Divalent", "Monovalent"]
        optional_x = [c for c in x_candidates if c not in must_have_present]
        _x_default_sel = _avail(_X_DEFAULTS, optional_x)
        extra_x = st.multiselect(
            "Additional X axis columns",
            options=optional_x,
            default=_x_default_sel if _x_default_sel else optional_x[:min(5, len(optional_x))],
            key="x_cols_extra",
            label_visibility="collapsed",
        )
        x_cols = list(dict.fromkeys(must_have_present + extra_x))

        if must_have_present:
            st.info(
                "✅ Always-included X axes: "
                + ", ".join(f"**{f}**" for f in must_have_present)
            )

        st.markdown("**Colour-by features** (each X folder gets one plot per colour):")
        _COLOR_DEFAULTS = ["Dilution Ratio", "Dilution Ratio_Corrected"]
        _color_default_sel = _avail(_COLOR_DEFAULTS, color_candidates)
        color_cols = st.multiselect(
            "Colour-by columns",
            options=color_candidates,
            default=_color_default_sel if _color_default_sel else color_candidates[:min(1, len(color_candidates))],
            key="color_cols",
            label_visibility="collapsed",
        )

    if not x_cols:
        st.warning("Select at least one X-axis column.")
        st.stop()
    if not color_cols:
        st.warning("Select at least one colour column.")
        st.stop()

    # ── 3D plot axes configuration ────────────────────────────────────────────────
    st.markdown("##### 🧊 3D Plot Configuration")
    st.caption(
        "X and Y axes are features; Z axis (vertical) is always the target. "
        "Colour drawn from feature pool + ratios + oil. "
        "Every unique UNORDERED triplet (X, Y, Color) is plotted exactly once — "
        "so (AOS, APG, Citric) and (APG, AOS, Citric) produce the same single plot."
    )

    # Full feature pool for 3D: individual + sum features + ratios + oil
    # Oil properties are included so they can be used as colour in 3D
    feat_pool_3d = list(dict.fromkeys(
        individual_selected + must_have_present + computed_sums
        + all_ratio_cols + oil_cols_in_data
    ))

    # Default selection for 3D pool:
    #   named individual components + all sum features + oil properties selected by user
    _3D_COMPONENT_DEFAULTS = ["APG (%)", "AOS (%)", "CapB (%)", "Divalent", "Monovalent"]
    _3D_SUM_DEFAULTS = [
        "Anionic (All Types)", "Nonionic (All Types)", "Zwitterionic (All Types)",
        "Sum Surfactant", "Nanoparticle (All Types)", "Polymer (All Types)",
        "Acid (All Types)", "Citric (All Types)",
    ]
    _3d_default = list(dict.fromkeys(
        _avail(_3D_COMPONENT_DEFAULTS, feat_pool_3d)   # matched individual cols
        #+ _avail(_3D_SUM_DEFAULTS, feat_pool_3d)        # matched sum features
        #+ oil_cols_in_data                              # all oil cols in data
    ))

    with st.container(border=True):
        xz_color_pool_sel = st.multiselect(
            "Feature pool for X, Z and Colour axes",
            options=feat_pool_3d,
            default=_3d_default if _3d_default else feat_pool_3d[:min(8, len(feat_pool_3d))],
            key="xz_pool_3d",
            help="Every unique unordered triplet (X, Z, Color) drawn from this pool "
                "produces one plot. X ≠ Z ≠ Color.",
        )
        # Always inject must-have sum features (same as 2D locked X axes)
        xz_color_pool = list(dict.fromkeys(must_have_present + xz_color_pool_sel))

        if must_have_present:
            st.caption(
                "🔒 Always included: "
                + ", ".join(f"**{f}**" for f in must_have_present)
            )

        color_pool_3d = xz_color_pool   # colour drawn from the same pool
        def _count_triplets(xz, colors):
            s = set()
            for _x in xz:
                for _z in xz:
                    if _z == _x: continue
                    for _c in colors:
                        if _c in (_x, _z): continue
                        s.add(frozenset([_x, _z, _c]))
            return len(s)
        # Recompute counts now that xz_color_pool is finalised (includes must-haves)
        n_3d_sf1 = _count_triplets(xz_color_pool, xz_color_pool)
        n_3d_sf2 = _count_triplets(xz_color_pool, all_ratio_cols)
        n_3d_sf3 = _count_triplets(xz_color_pool, oil_cols_in_data)
        n_3d     = n_3d_sf1 + n_3d_sf2 + n_3d_sf3

        st.info(
            f"**{len(xz_color_pool)}** features in pool → "
            f"**{n_3d_sf1}** SF1 (component colour) + "
            f"**{n_3d_sf2}** SF2 (ratio colour) + "
            f"**{n_3d_sf3}** SF3 (oil colour) = "
            f"**{n_3d}** total 3D plots"
        )

    # Per x_col: peers = other x_cols + extra color_cols
    def _count_sf1(x):
        peers  = [c for c in x_cols    if c != x]
        extras = [c for c in color_cols if c != x and c not in peers]
        return len(peers) + len(extras)

    sf3_x = list(dict.fromkeys(x_cols + all_ratio_cols))
    n_sf1   = sum(_count_sf1(x) for x in x_cols)
    n_sf2   = sum(len([r for r in all_ratio_cols if r != x]) for x in x_cols)
    n_sf3   = sum(len([o for o in oil_cols_in_data if o != x]) for x in sf3_x)


    # n_3d counters are computed inside the container after xz_color_pool is set
    n_3d_sf1, n_3d_sf2, n_3d_sf3, n_3d = 0, 0, 0, 0
    n_plots = n_sf1 + n_sf2 + n_sf3 + n_3d
    st.info(
        f"Will generate **{n_plots}** plots total: "
        f"| **{n_sf1}** 2D component-component "
        f"| **{n_sf2}** 2D ratio-component "
        f"| **{n_sf3}** 2D ratio-oil "
        f"| **{n_3d}** 3D unique triplets (duplicates removed)"
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 7 — Save destination
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 7 · Save Destination")
    save_mode = st.radio(
        "How to get the plots?",
        ["⬇️ Download as ZIP (browser)", "💾 Save directly to disk path"],
        horizontal=True,
        key="save_mode",
    )

    disk_path = ""
    if "disk" in save_mode:
        disk_path = st.text_input(
            "Destination folder path",
            value=str(Path.home() / "foam_plots"),
            help="The folder will be created if it does not exist.",
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 8 — Generate
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 8 · Generate Plots")

    gc1, gc2 = st.columns(2)
    do_2d = gc1.checkbox("📊 Generate 2D plots", value=True,  key="do_2d")
    do_3d = gc2.checkbox("🧊 Generate 3D plots", value=False, key="do_3d")

    if not do_2d and not do_3d:
        st.warning("Select at least one plot type above.")
        st.stop()

    if st.button("🚀 Generate Plots", type="primary", use_container_width=True):

        figures: dict[str, bytes] = {}

        if do_2d:
            with st.spinner("Building 2D plots…"):
                figures_2d = generate_all_plots(
                    df_work, x_cols, color_cols,
                    all_ratio_cols, oil_cols_in_data, target_col,
                )
            figures.update({f"2D Plots/{k}": v for k, v in figures_2d.items()})
        else:
            figures_2d = {}

        if do_3d:
            with st.spinner("Building 3D plots…"):
                figures_3d = generate_3d_plots(
                    df_work,
                    xz_color_pool,    # individual + sum features (X/Z axes + SF1 colour)
                    all_ratio_cols,   # ratio features (SF2 colour)
                    oil_cols_in_data, # oil properties (SF3 colour)
                    target_col,
                )
            figures.update({f"3D Plots/{k}": v for k, v in figures_3d.items()})
        else:
            figures_3d = {}

        parts_msg = []
        if do_2d: parts_msg.append(f"**{len(figures_2d)}** 2D")
        if do_3d: parts_msg.append(f"**{len(figures_3d)}** 3D")
        st.success(f"✅ Generated {' + '.join(parts_msg)} = **{len(figures)}** plots total.")

        # ── Save / download ───────────────────────────────────────────────────
        if "disk" in save_mode:
            if not disk_path.strip():
                st.error("Enter a valid folder path.")
            else:
                try:
                    n = save_figures_to_disk(figures, disk_path.strip())
                    st.success(
                        f"💾 Saved **{n}** plots to:\n\n"
                        f"`{disk_path.strip()}/2D Plots/…  and  3D Plots/…`"
                    )
                except Exception as e:
                    st.error(f"Save failed: {e}")

        else:
            zip_bytes = figures_to_zip(figures)
            st.download_button(
                label="⬇️ Download all plots as ZIP",
                data=zip_bytes,
                file_name="foam_scatter_plots.zip",
                mime="application/zip",
                use_container_width=True,
            )
            st.caption(
                "ZIP structure:\n"
                "`2D Plots / Effect of individual component… / <X> / X_…_Color_….png`\n"
                "`2D Plots / Effect of component Ratio… / <X> / X_…_Color_….png`\n"
                "`2D Plots / Effect of component Ratio and individual component to oil… / <X or Ratio> / X_…_Color_OilProp.png`"
            )

        # ── Inline preview of first 6 plots ──────────────────────────────────
        st.divider()
        st.markdown("#### Preview (first 6 plots)")
        preview_items = list(figures.items())[:6]
        prev_cols = st.columns(2)
        for pi, (path_key, png_bytes) in enumerate(preview_items):
            with prev_cols[pi % 2]:
                st.image(io.BytesIO(png_bytes), caption=path_key,
                        use_column_width=True)