"""
Page_auto_ML.py
================
Standalone Streamlit app — Auto ML + SHAP Dependence Plots.

Mirrors Page_auto_eda.py exactly:
  • Same upload / group-selection / computed-feature workflow
  • Instead of scatter plots → 2D SHAP dependence plots
    (X = feature value in original scale, Y = SHAP value,
     colour = another feature, marginal histograms on both axes)
  • Same folder structure as EDA page

Model:
  RF Regressor  if target is numeric (regression)
  RF Classifier if target is categorical / binary

Preprocessing:
  Chemical groups (surfactant, nano, polymer, …) → NaN filled with 0
  Condition columns (Brine, Oil, Process)         → NaN rows dropped
  Categorical features                            → one-hot encoded
  All numeric features                            → StandardScaler

Run:
    streamlit run Page_auto_ML.py

Dependencies:
    pip install streamlit pandas numpy matplotlib scikit-learn shap openpyxl
"""
from __future__ import annotations

import io
import os
import re
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st
from utils.data_filter import render_data_filters

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Shared taxonomy (identical to Page_auto_eda.py)
# ─────────────────────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    "Nanoparticle": ["HS (%)", "BLH5 (%)", "HSA (%)"],
    "Anionic":      ["AOS (%)", "alpha-step (%)", "a-step (%)", "SDS (%)", "SLES (%)",
                     "n. LABS (%)", "DB45 (%)", "Cola SLAA (%)", "Cola SC (%)"],
    "Nonionic":     ["APG (%)", "decyl glucoside (%)", "caprylyl glucoside (%)",
                     "Tween 80 (%)", "PG (%)", "LAO (%)"],
    "Zwitterionic": ["CapB (%)", "2C (%)", "Cola 2C (%)", "amine oxide (%)",
                     "Cola LMB (%)", "Amphosol 1c (%)", "SC (%)",
                     "LBHP (%)", "CS50(%)", "DM (%)"],
    "Polymer":      ["HPAM (%)", "xanthan gum (%) ", "Guar Gum (%)", "FPAM (%)",
                     "PAA (%)", "PA (%)", "ClearHib 1000 (%)"],
    "Citric":       ["Citric (%)",
                     "31.2(%) citric+ 13.3(%) KOH  (pH not adjusted, pH=4.46 )",
                     "31.2% citric+ 13.3% KOH  (adjusted pH=4) (%)",
                     "31.2% citric+ 13.3% KOH (not adjusted pH) (%)",
                     "31.2% citric+ 13.3% KOH (pH=4) (%)",
                     "38.1% citric+   KOH  (pH=5) (%)",
                     "potassium citrate (9.7%)/citric acid buffer (19.22%)  pH=3 (%)",
                     "potassium citrate (%)"],
    "Acid":         ["EDTA (%)", "etidronic acid (%)", "acetic acid (%)"],
    "Antiscalant":  ["Mem 2000-clear tech (%)", "Mem 2500-clear tech (%)",
                     "Mem 4000-clear tech (%)", "Mem 3500-clear tech (%)",
                     "Mem 3000-clear tech (%)"],
    "Brine":        ["Divalent", "Monovalent"],
    "Oil":          ["Linear paraffins", "Branched paraffins", "Cycloparaffins",
                     "Aromatics", "Oxygenates", "Biomarker-like compounds",
                     "Sulfur compounds", "Wax fraction (C24+)"],
    "Process":      ["Temperature", "Dilution Ratio", "Oil (%)",
                     "concentrate manufacturing method (Ratio)",
                     "Initial Foam Temp (dilution Temp) "],
}

SUM_FEATURES: dict[str, str] = {
    "Anionic (All Types)":      "Anionic",
    "Nonionic (All Types)":     "Nonionic",
    "Zwitterionic (All Types)": "Zwitterionic",
    "Sum Surfactant":           "Surfactant",
    "Nanoparticle (All Types)": "Nanoparticle",
    "Polymer (All Types)":      "Polymer",
    "Acid (All Types)":         "Acid",
    "Citric (All Types)":       "Citric",
}

# Chemical groups: NaN = not added → fill 0
CHEM_GROUPS = {"Nanoparticle","Anionic","Nonionic","Zwitterionic",
               "Polymer","Citric","Acid","Antiscalant"}
# Condition groups: NaN = unknown → drop row
COND_GROUPS = {"Brine","Oil","Process"}

TARGET_CANDIDATES = [
    "Half life (method: poly) [h]",
    "Half life (method: poly) [h] 95%",
    "Calculated Half Life (hr_Linear)",
    "Calculated Half Life (hr)",
]

GRP_CLR = {
    "Nanoparticle": "#1565C0", "Anionic":    "#E53935",
    "Nonionic":     "#FB8C00", "Zwitterionic":"#8E24AA",
    "Surfactant":   "#FF8F00", "Polymer":    "#6A1B9A",
    "Citric":       "#2E7D32", "Acid":       "#C62828",
    "Antiscalant":  "#00695C", "Brine":      "#4527A0",
    "Oil":          "#BF360C", "Process":    "#546E7A",
    "Sum":          "#37474F",
}

FIG_DPI = 150

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (shared with EDA page)
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r'[^\w\s\-]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s[:60]


def _normalise(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r'\s*\(%\)\s*$', '', s).strip()
    return s


def _avail(candidates: list[str], cols: list[str]) -> list[str]:
    norm_to_col = {_normalise(c): c for c in cols}
    matched, seen = [], set()
    for cand in candidates:
        key = _normalise(cand)
        if key in norm_to_col:
            actual = norm_to_col[key]
            if actual not in seen:
                matched.append(actual)
                seen.add(actual)
    return matched


def _resolve_col(name: str, df: pd.DataFrame) -> str | None:
    if name in df.columns:
        return name
    key = _normalise(name)
    for col in df.columns:
        if _normalise(col) == key:
            return col
    return None


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_sum_features(df: pd.DataFrame,
                          sel_groups: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for feat_name, group_key in SUM_FEATURES.items():
        cols = (sel_groups.get("Anionic", []) +
                sel_groups.get("Nonionic", []) +
                sel_groups.get("Zwitterionic", [])) \
               if group_key == "Surfactant" \
               else sel_groups.get(group_key, [])
        present = [c for c in [_resolve_col(c, out) for c in cols] if c]
        if present:
            s = pd.Series(0.0, index=out.index)
            for c in present:
                s += _safe_num(out[c]).fillna(0)
            out[feat_name] = s
    return out


def compute_ratio_features(df: pd.DataFrame,
                            ratios: list[tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    for num, den in ratios:
        if num in out.columns and den in out.columns:
            out[f"{num} / {den}"] = _safe_num(out[num]) / _safe_num(out[den]).replace(0, np.nan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame,
               feature_cols: list[str],
               target_col: str,
               chem_cols: list[str],
               cond_cols: list[str],
               task: str,
               chem_nan_strategy: str = "fill_zero",
) -> tuple[pd.DataFrame, pd.Series, list[str], object]:
    """
    Clean data, encode categoricals, scale numerics.

    NaN policy
    ----------
    Chemical groups (surfactant, nano, polymer, citric, acid, antiscalant)
        chem_nan_strategy="fill_zero" → NaN = not added = fill 0
        chem_nan_strategy="drop_row"  → NaN rows dropped (same as conditions)
    Condition groups (Brine, Oil, Process)
        → NaN = unknown = DROP ROW  (always)
    All other NaN in features → row dropped after the above steps

    Returns
    -------
    X_scaled   : scaled feature DataFrame (for model)
    y          : target Series
    feat_names : list of feature names after encoding
    scaler     : fitted StandardScaler
    """
    from sklearn.preprocessing import StandardScaler

    data = df[feature_cols + [target_col]].copy()

    # 1. Handle chemical NaN per user strategy
    for col in chem_cols:
        if col not in data.columns:
            continue
        data[col] = _safe_num(data[col])
        if chem_nan_strategy == "fill_zero":
            data[col] = data[col].fillna(0)
        # "drop_row": leave NaN — will be caught by dropna below

    # 1b. fill_zero: also fill derived sum/ratio features whose NaN
    #     comes from zero-filled chem parents (e.g. Anionic (All Types),
    #     Nanoparticle + Sum Surfactant, etc.)
    if chem_nan_strategy == "fill_zero":
        for col in feature_cols:
            if col not in data.columns:
                continue
            if " + " in col or " / " in col:
                data[col] = _safe_num(data[col]).fillna(0)
        _SUM_NAMES = ["Anionic (All Types)", "Nonionic (All Types)",
                      "Zwitterionic (All Types)", "Sum Surfactant",
                      "Nanoparticle (All Types)", "Polymer (All Types)",
                      "Acid (All Types)", "Citric (All Types)"]
        for col in feature_cols:
            if col in data.columns and col in _SUM_NAMES:
                data[col] = _safe_num(data[col]).fillna(0)

    # 2. Drop rows with NaN in condition cols (always)
    cond_present = [c for c in cond_cols if c in data.columns]
    if cond_present:
        data = data.dropna(subset=cond_present)

    # 2b. If chem strategy is drop_row, drop rows with NaN in any chem col
    if chem_nan_strategy == "drop_row":
        chem_present = [c for c in chem_cols if c in data.columns]
        if chem_present:
            data = data.dropna(subset=chem_present)

    # 3. Clean target
    if task == "regression":
        data[target_col] = _safe_num(data[target_col])
        data = data.dropna(subset=[target_col])
        data = data[data[target_col] <= 1e5]
    else:
        data = data.dropna(subset=[target_col])

    # 4. One-hot encode categorical feature columns
    cat_cols = [c for c in feature_cols
                if c in data.columns
                and not pd.api.types.is_numeric_dtype(data[c])]
    if cat_cols:
        data = pd.get_dummies(data, columns=cat_cols, drop_first=False)

    # Rebuild feature list after encoding
    feat_names = [c for c in data.columns if c != target_col]

    # 5. Fill any remaining NaN in features with 0
    data[feat_names] = data[feat_names].fillna(0)

    y = data[target_col].copy()
    X = data[feat_names].copy()

    # 6. Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feat_names, index=X.index)

    return X_scaled, y, feat_names, scaler


def _render_target_conversion(df: pd.DataFrame, target_col: str):
    """
    Optionally bin a numeric target into 2 or 3 classes.
    Returns (y_series, task_str).
    Only shown when target is numeric — caller is responsible for the guard.
    """
    import plotly.express as _px
    with st.container(border=True):
        y_orig = pd.to_numeric(df[target_col], errors="coerce").dropna()
        c1, c2 = st.columns([2, 3])
        mode = c1.radio(
            "Target mode",
            ["Regression (continuous)", "Binary: Low / High", "Ternary: Low / Mid / High"],
            key="tconv_mode",
        )
        y_out = pd.to_numeric(df[target_col], errors="coerce")

        if mode == "Regression (continuous)":
            c2.info(
                f"Continuous: **{target_col}**\n\n"
                f"Range: {y_orig.min():.2f} – {y_orig.max():.2f}  |  "
                f"Median: {y_orig.median():.2f}"
            )
            task = "regression"

        elif mode == "Binary: Low / High":
            cut = c2.slider(
                "Cut-off  (Low < cut ≤ High)",
                float(y_orig.min()), float(y_orig.max()),
                float(y_orig.median()), key="tconv_cut2",
            )
            y_out = (y_out > cut).map({False: "Low", True: "High"})
            vc = y_out.value_counts()
            c2.success(f"Low: **{vc.get('Low', 0)}**  |  High: **{vc.get('High', 0)}**")
            task = "classification"

        else:  # ternary
            lo_d = float(y_orig.quantile(0.33))
            hi_d = float(y_orig.quantile(0.67))
            rng  = c2.slider(
                "Boundaries  (Low ≤ lo < Mid ≤ hi < High)",
                float(y_orig.min()), float(y_orig.max()),
                (lo_d, hi_d), key="tconv_cut3",
            )
            lo_cut, hi_cut = rng
            def _bin3(v):
                if pd.isna(v): return np.nan
                if v <= lo_cut: return "Low"
                if v <= hi_cut: return "Mid"
                return "High"
            y_out = y_out.apply(_bin3)
            vc = y_out.value_counts()
            c2.success(
                f"Low: **{vc.get('Low', 0)}**  |  "
                f"Mid: **{vc.get('Mid', 0)}**  |  "
                f"High: **{vc.get('High', 0)}**"
            )
            task = "classification"

        # Target distribution histogram
        fig = _px.histogram(
            x=y_orig, nbins=40,
            title="Original Target Distribution",
            labels={"x": target_col},
            template="plotly_white",
            color_discrete_sequence=["#1565C0"],
        )
        fig.update_layout(height=230, margin=dict(t=35, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    return y_out, task


def detect_task(y: pd.Series) -> str:
    """
    Auto-detect regression vs classification.
    Coerces to numeric first (handles string-encoded floats from Excel).
    Ignores NaN rate — target columns often have many NaN rows.
    Only looks at the non-NaN values.
    """
    y_num   = pd.to_numeric(y, errors="coerce")
    y_clean = y_num.dropna()
    if len(y_clean) < 5 or y_clean.nunique() < 2:
        return "classification"
    n_unique     = y_clean.nunique()
    frac_nonint  = (y_clean % 1 != 0).mean()
    if n_unique <= 20:
        return "classification"
    if frac_nonint >= 0.02:
        return "regression"              # has fractional values → continuous
    return "regression" if n_unique / len(y_clean) > 0.50 else "classification"


# ─────────────────────────────────────────────────────────────────────────────
#  Model training
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training Random Forest…")
def train_model(X_hash: str, X: pd.DataFrame, y: pd.Series, task: str,
                n_estimators: int, max_depth: int | None, random_state: int):
    """Train RF and compute SHAP values. Cached by data hash + hyperparams."""
    import shap
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=3, random_state=random_state, n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=3, random_state=random_state, n_jobs=-1,
        )

    model.fit(X, y)

    # SHAP on a subsample for speed
    n_shap = min(600, len(X))
    X_shap = X.sample(n_shap, random_state=random_state) if len(X) > n_shap else X.copy()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)

    # Normalise shap_vals to always be 2D [n_samples, n_features]:
    #   Old SHAP: returns list of 2D arrays (one per class)
    #   New SHAP: returns 3D array [n_samples, n_features, n_classes]
    if isinstance(shap_vals, list):
        # list of [n_samples, n_features] arrays
        if len(shap_vals) == 2:
            shap_vals = shap_vals[1]                            # binary: positive class
        else:
            shap_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)  # multiclass: mean abs
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # 3D array [n_samples, n_features, n_classes]
        if shap_vals.shape[2] == 2:
            shap_vals = shap_vals[:, :, 1]                     # binary: positive class
        else:
            shap_vals = np.abs(shap_vals).mean(axis=2)         # multiclass: mean abs over classes
    # shap_vals is now guaranteed 2D [n_samples, n_features]

    return model, X_shap, shap_vals


# ─────────────────────────────────────────────────────────────────────────────
#  SHAP dependence plot
# ─────────────────────────────────────────────────────────────────────────────

def _shap_dependence(
    X_orig: pd.DataFrame,        # original-scale feature values (for x-axis)
    shap_vals: np.ndarray,        # shap values array [n_samples × n_features]
    feat_names: list[str],
    x_feat: str,                  # feature on x-axis
    color_feat: str,              # feature used for colouring
    title: str,
) -> plt.Figure:
    """
    2D SHAP dependence plot with marginal histograms — mirrors the reference image:
      X axis  : feature value (original scale)
      Y axis  : SHAP value (log-odds or regression impact)
      Colour  : another feature (continuous → coolwarm, categorical → tab10)
      Margins : histogram of X (top) and SHAP value (right)
    """
    if x_feat not in feat_names or x_feat not in X_orig.columns:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
        ax.text(0.5, 0.5, f"Feature '{x_feat}' not available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    xi   = feat_names.index(x_feat)
    xv   = X_orig[x_feat].values.astype(float)
    sv   = shap_vals[:, xi]

    # Colour values
    if color_feat in X_orig.columns:
        cv = X_orig[color_feat].values
        is_num_c = pd.api.types.is_numeric_dtype(X_orig[color_feat])
    else:
        cv = sv          # fallback: colour by SHAP value itself
        is_num_c = True
        color_feat = f"SHAP({x_feat})"

    # Remove NaN rows
    valid = ~(np.isnan(xv) | np.isnan(sv))
    xv, sv = xv[valid], sv[valid]
    cv = cv[valid] if hasattr(cv, '__len__') else cv

    # ── Layout: main scatter + top histogram + right histogram ───────────
    fig = plt.figure(figsize=(8, 6), dpi=FIG_DPI)
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[5, 1], height_ratios=[1, 5],
        hspace=0.05, wspace=0.05,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # ── Main scatter ─────────────────────────────────────────────────────
    if is_num_c:
        cv_float = cv.astype(float)
        c_min, c_max = float(np.nanmin(cv_float)), float(np.nanmax(cv_float))
        if c_min == c_max:
            c_max = c_min + 1
        sc = ax_main.scatter(
            xv, sv, c=cv_float,
            cmap="coolwarm",
            vmin=c_min, vmax=c_max,
            alpha=0.75, s=22,
            edgecolors="none",
        )
        # Colourbar inside main axes (right side, compact)
        cbar = fig.colorbar(sc, ax=ax_main, fraction=0.03, pad=0.01)
        cbar.ax.tick_params(labelsize=6)
        mid  = (c_min + c_max) / 2
        cbar.set_ticks([c_min, mid, c_max])
        cbar.set_ticklabels([f"{c_min:.2g}", "Med", f"{c_max:.2g}"])
        cbar.set_label(color_feat, fontsize=7, rotation=270, labelpad=10)
        # Add Min/Max labels like reference image
        cbar.ax.text(0.5, 1.04, "Max", ha="center", va="bottom",
                     fontsize=6, transform=cbar.ax.transAxes)
        cbar.ax.text(0.5, -0.04, "Min", ha="center", va="top",
                     fontsize=6, transform=cbar.ax.transAxes)
    else:
        cats    = pd.Categorical(cv.astype(str))
        uniq    = list(cats.categories)
        palette = plt.colormaps["tab10"].resampled(len(uniq))
        for i, cat in enumerate(uniq):
            mask = np.array(cv.astype(str)) == cat
            ax_main.scatter(xv[mask], sv[mask], color=palette(i),
                            alpha=0.75, s=22, edgecolors="none", label=cat)
        ax_main.legend(title=color_feat, fontsize=6, title_fontsize=6,
                       loc="upper right", framealpha=0.6,
                       markerscale=0.8, handletextpad=0.3)

    # ── Trend line + confidence shade (LOWESS) ───────────────────────────
    if len(xv) >= 10:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
            _order = np.argsort(xv)
            _xs, _ys = xv[_order], sv[_order]
            _frac = max(0.15, min(0.5, 40 / len(_xs)))   # adaptive bandwidth
            _smooth = _lowess(_ys, _xs, frac=_frac, return_sorted=True)
            _tx, _ty = _smooth[:, 0], _smooth[:, 1]

            # Residual std for the ±1 std confidence band
            _resid   = np.interp(_xs, _tx, _ty)
            _std_res = np.std(_ys - _resid)

            ax_main.fill_between(
                _tx, _ty - _std_res, _ty + _std_res,
                alpha=0.18, color="#E53935", zorder=1, linewidth=0,
            )
            ax_main.plot(
                _tx, _ty,
                color="#E53935", lw=1.8, zorder=2, label="LOWESS trend",
            )
        except Exception:
            # statsmodels not available — fall back to degree-2 polynomial
            try:
                _order  = np.argsort(xv)
                _xs, _ys = xv[_order], sv[_order]
                _coef   = np.polyfit(_xs, _ys, 2)
                _poly   = np.poly1d(_coef)
                _tx     = np.linspace(_xs.min(), _xs.max(), 200)
                _ty     = _poly(_tx)
                _resid  = _ys - _poly(_xs)
                _std_res = np.std(_resid)
                ax_main.fill_between(
                    _tx, _ty - _std_res, _ty + _std_res,
                    alpha=0.18, color="#E53935", zorder=1, linewidth=0,
                )
                ax_main.plot(
                    _tx, _ty,
                    color="#E53935", lw=1.8, zorder=2, label="Poly-2 trend",
                )
            except Exception:
                pass

    ax_main.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)
    ax_main.set_xlabel(f"{x_feat}  (original scale)", fontsize=8)
    ax_main.set_ylabel("SHAP value  (log-odds add.)", fontsize=8)
    ax_main.tick_params(labelsize=7)
    ax_main.set_facecolor("#F8F8F8")
    for sp in ax_main.spines.values():
        sp.set_linewidth(0.4)

    # ── Top histogram (X feature values) ─────────────────────────────────
    ax_top.hist(xv, bins=40, color="#AAAAAA", edgecolor="none", alpha=0.8)
    ax_top.set_ylabel("Count", fontsize=6)
    ax_top.tick_params(labelsize=6, left=False, labelleft=False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_facecolor("#F8F8F8")
    for sp in ax_top.spines.values():
        sp.set_linewidth(0.3)

    # ── Right histogram (SHAP values) ─────────────────────────────────────
    ax_right.hist(sv, bins=40, color="#AAAAAA", edgecolor="none",
                  alpha=0.8, orientation="horizontal")
    ax_right.set_xlabel("Count", fontsize=6)
    ax_right.tick_params(labelsize=6, bottom=False, labelbottom=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_facecolor("#F8F8F8")
    for sp in ax_right.spines.values():
        sp.set_linewidth(0.3)

    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def _render_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  Plot generation — same folder structure as EDA page
# ─────────────────────────────────────────────────────────────────────────────

def _fname_shap(x_feat: str, color_feat: str) -> str:
    return f"SHAP_{_slugify(x_feat)}_Color_{_slugify(color_feat)}.png"


def generate_shap_plots(
    X_orig: pd.DataFrame,
    shap_vals: np.ndarray,
    feat_names: list[str],
    x_feats: list[str],
    color_feats: list[str],
    subfolder: str,
) -> dict[str, bytes]:
    """
    Mirrors EDA 2D page colour logic:

    Folder structure:
      SHAP Plots/
        {subfolder}/
          {x_feat}/
            SHAP_{x}_Color_{peer}.png   ← peer = another X feature
            SHAP_{x}_Color_{user}.png   ← user colour-by selection

    Colour set per X folder:
      peers  = all other x_feats  (individual + sum features as colours)
      extras = user colour_feats that are not already a peer
    Self-colour always skipped.
    """
    figures: dict[str, bytes] = {}

    def _colours_for(x_feat: str) -> list[str]:
        peers  = [c for c in x_feats    if c != x_feat]
        extras = [c for c in color_feats if c != x_feat and c not in peers]
        return list(dict.fromkeys(peers + extras))

    valid_x = [x for x in x_feats if x in feat_names]
    total   = sum(len(_colours_for(x)) for x in valid_x)
    if total == 0:
        return figures

    prog = st.progress(0, text="Generating SHAP plots…")
    done = 0

    for x_feat in valid_x:
        x_folder = _slugify(x_feat)
        for color_feat in _colours_for(x_feat):
            if color_feat not in X_orig.columns and color_feat not in feat_names:
                done += 1
                prog.progress(min(done / total, 1.0))
                continue
            title = f"SHAP Dependence  |  X={x_feat}  |  Colour={color_feat}"
            fig   = _shap_dependence(X_orig, shap_vals, feat_names,
                                     x_feat, color_feat, title)
            fname = _fname_shap(x_feat, color_feat)
            figures[f"{subfolder}/{x_folder}/{fname}"] = _render_to_bytes(fig)
            done += 1
            prog.progress(min(done / total, 1.0),
                          text=f"SHAP [{x_feat}] colour={color_feat}…")

    prog.empty()
    return figures


def figures_to_zip(figures: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path_key, png_bytes in figures.items():
            # keys already include SF1/ or SF2/ prefix — wrap in SHAP Plots/
            zf.writestr(f"SHAP Plots/{path_key}", png_bytes)
    buf.seek(0)
    return buf.read()


def save_figures_to_disk(figures: dict[str, bytes], base_dir: str) -> int:
    count = 0
    for path_key, png_bytes in figures.items():
        parts     = path_key.split("/")
        dest_dir  = os.path.join(base_dir, "SHAP Plots", *parts[:-1])
        dest_file = os.path.join(dest_dir, parts[-1])
        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_file, "wb") as fh:
            fh.write(png_bytes)
        count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
#  Model Performance plots
# ─────────────────────────────────────────────────────────────────────────────

def _generate_model_performance(
    model, X_scaled, y, task, feat_names, shap_vals, X_shap_orig, target_col
) -> dict[str, bytes]:
    """
    Professional model performance plots.
    Regression:     01_Predicted_vs_Actual, 02_Residuals,
                    03_Residual_Distribution, 04_SHAP_Feature_Importance,
                    05_RF_MDI, 06_SHAP_Beeswarm
    Classification: 01_Confusion_Matrix, 02_Classification_Report,
                    03_ROC_Curve, 04_SHAP_Feature_Importance,
                    05_RF_MDI, 06_SHAP_Beeswarm
    """
    from sklearn.metrics import (
        r2_score, mean_absolute_error, mean_squared_error,
        confusion_matrix, classification_report, roc_curve, auc,
        ConfusionMatrixDisplay,
    )
    import warnings; warnings.filterwarnings("ignore")

    figs: dict[str, bytes] = {}

    def _save(fig, name):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=FIG_DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        figs[name] = buf.read()

    y_pred = model.predict(X_scaled)

    if task == "regression":
        y_num = pd.to_numeric(y, errors="coerce").values
        y_pr  = y_pred.astype(float)
        r2    = r2_score(y_num, y_pr)
        mae   = mean_absolute_error(y_num, y_pr)
        rmse  = float(np.sqrt(mean_squared_error(y_num, y_pr)))

        # Predicted vs Actual
        fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
        err = np.abs(y_num - y_pr)
        sc  = ax.scatter(y_num, y_pr, c=err, cmap="RdYlGn_r",
                         s=25, alpha=0.7, edgecolors="none")
        lims = [min(y_num.min(), y_pr.min()), max(y_num.max(), y_pr.max())]
        ax.plot(lims, lims, "k--", lw=1.2)
        fig.colorbar(sc, ax=ax, label="|Error|", shrink=0.8)
        ax.set_xlabel(f"Actual {target_col}", fontsize=9)
        ax.set_ylabel(f"Predicted {target_col}", fontsize=9)
        ax.set_title(f"Predicted vs Actual\nR²={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}",
                     fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); fig.tight_layout()
        _save(fig, "01_Predicted_vs_Actual.png")

        # Residuals vs Predicted
        residuals = y_num - y_pr
        fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
        ax.scatter(y_pr, residuals, alpha=0.6, s=20,
                   edgecolors="none", color="#1565C0")
        ax.axhline(0, color="red", lw=1.2, ls="--")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.set_title("Residuals vs Predicted", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); fig.tight_layout()
        _save(fig, "02_Residuals_vs_Predicted.png")

        # Residual distribution
        fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
        ax.hist(residuals, bins=40, color="#1565C0", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="red", lw=1.2, ls="--")
        ax.set_xlabel("Residual", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(
            f"Residual Distribution  (mean={residuals.mean():.2f}, std={residuals.std():.2f})",
            fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); fig.tight_layout()
        _save(fig, "03_Residual_Distribution.png")

    else:  # classification
        classes = sorted(y.unique(), key=str)

        # Confusion Matrix
        cm  = confusion_matrix(y, y_pred, labels=classes)
        sz  = max(5, len(classes) * 1.5)
        fig, ax = plt.subplots(figsize=(sz, sz * 0.8), dpi=FIG_DPI)
        ConfusionMatrixDisplay(
            cm, display_labels=[str(c) for c in classes]
        ).plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); fig.tight_layout()
        _save(fig, "01_Confusion_Matrix.png")

        # Classification Report heatmap
        report = classification_report(
            y, y_pred, labels=classes,
            target_names=[str(c) for c in classes],
            output_dict=True, zero_division=0,
        )
        rdf = (pd.DataFrame(report).T
                 .select_dtypes(include="number")
                 .drop(columns=["support"], errors="ignore"))
        fig, ax = plt.subplots(figsize=(6, max(3, len(rdf) * 0.5)), dpi=FIG_DPI)
        im = ax.imshow(rdf.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(rdf.columns)))
        ax.set_xticklabels(rdf.columns, fontsize=8)
        ax.set_yticks(range(len(rdf.index)))
        ax.set_yticklabels(rdf.index, fontsize=8)
        for (i, j), v in np.ndenumerate(rdf.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.3 < v < 0.8 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Classification Report", fontsize=9, fontweight="bold")
        fig.tight_layout()
        _save(fig, "02_Classification_Report.png")

        # ROC Curve
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_scaled)
                fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
                if len(classes) == 2:
                    from sklearn.preprocessing import LabelEncoder
                    y_enc = LabelEncoder().fit_transform(y)
                    fpr, tpr, _ = roc_curve(y_enc, y_prob[:, 1])
                    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr, tpr):.3f}")
                else:
                    from sklearn.preprocessing import label_binarize
                    y_bin = label_binarize(y, classes=classes)
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                        ax.plot(fpr, tpr, lw=2,
                                label=f"{cls}  AUC={auc(fpr, tpr):.2f}")
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_xlabel("FPR", fontsize=9)
                ax.set_ylabel("TPR", fontsize=9)
                ax.set_title("ROC Curve", fontsize=9, fontweight="bold")
                ax.legend(fontsize=7, loc="lower right")
                ax.tick_params(labelsize=7)
                fig.tight_layout()
                _save(fig, "03_ROC_Curve.png")
            except Exception:
                pass

    # SHAP Summary Bar (both tasks)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp = pd.Series(mean_abs, index=feat_names).sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(7, max(4, len(imp) * 0.35)), dpi=FIG_DPI)
    ax.barh(imp.index, imp.values, color="#1565C0", edgecolor="none")
    ax.set_xlabel("Mean |SHAP|", fontsize=9)
    ax.set_title("Top 20 Features — Mean |SHAP|", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7); fig.tight_layout()
    _save(fig, "04_SHAP_Feature_Importance.png")

    # RF MDI Feature Importance
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feat_names
                       ).sort_values(ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(7, max(4, len(fi) * 0.35)), dpi=FIG_DPI)
        ax.barh(fi.index, fi.values, color="#E53935", edgecolor="none")
        ax.set_xlabel("MDI", fontsize=9)
        ax.set_title("Top 20 Features — RF Feature Importance (MDI)",
                     fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7); fig.tight_layout()
        _save(fig, "05_RF_Feature_Importance_MDI.png")

    # SHAP Beeswarm
    try:
        import shap as _shap
        top_idx  = np.argsort(mean_abs)[::-1][:15]
        top_feat = [feat_names[i] for i in top_idx]
        sv_top   = shap_vals[:, top_idx]
        if all(f in X_shap_orig.columns for f in top_feat):
            # plot_size=None tells SHAP not to size the figure at all, which
            # leaves matplotlib's tiny default canvas — long feature-name
            # labels then eat almost all the width, squeezing the beeswarm
            # itself into a sliver (the "00" duplicate-zero-tick symptom).
            # Give it real estate scaled to the number of features instead.
            _bee_h = max(4.5, len(top_feat) * 0.45)
            _shap.summary_plot(sv_top, X_shap_orig[top_feat],
                               show=False, max_display=15, plot_size=(10, _bee_h))
            plt.title("SHAP Beeswarm — Top 15 Features",
                      fontsize=9, fontweight="bold")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=FIG_DPI,
                        bbox_inches="tight", facecolor="white")
            plt.close(); buf.seek(0)
            figs["06_SHAP_Beeswarm.png"] = buf.read()
    except Exception:
        pass

    return figs


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.title("🤖 Foam Auto ML — SHAP Dependence Plots")
    st.caption(
        "Upload data → select groups → train RF model → generate organised SHAP plots. "
        "Same folder structure as the EDA page."
    )
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

    is_excel   = uploaded.name.lower().endswith((".xlsx", ".xls"))
    sheet_name: str | int = 0
    if is_excel:
        peek_bytes = uploaded.read(); uploaded.seek(0)
        try:
            sheets = pd.ExcelFile(io.BytesIO(peek_bytes)).sheet_names
            sheet_name = (st.selectbox("Select sheet", sheets, key="sheet")
                        if len(sheets) > 1 else sheets[0])
        except Exception as e:
            st.error(f"Cannot read Excel: {e}"); st.stop()

    @st.cache_data(show_spinner="Reading file…")
    def _load(fb: bytes, excel: bool, sheet) -> pd.DataFrame:
        if excel:
            return pd.read_excel(io.BytesIO(fb), sheet_name=sheet).dropna(how="all")
        # Try common encodings before giving up (Excel-exported CSVs are
        # often Windows-1252, not UTF-8, and raise UnicodeDecodeError otherwise).
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return pd.read_csv(io.BytesIO(fb), encoding=enc, low_memory=False).dropna(how="all")
            except UnicodeDecodeError:
                continue
        return pd.read_csv(io.BytesIO(fb), encoding="latin-1", low_memory=False).dropna(how="all")

    raw_bytes = uploaded.read()
    df_raw    = _load(raw_bytes, is_excel, sheet_name)
    st.success(f"✅  **{len(df_raw):,} rows × {df_raw.shape[1]} columns**")
    with st.expander("Preview raw data"):
        st.dataframe(df_raw, use_container_width=True)

    all_cols   = list(df_raw.columns)
    non_target = all_cols   # refined after target selection
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 2 — Data Filtering
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 2 · Data Filtering")
    st.caption(
        "Filter rows before training. Filters are non-destructive — adjust any time."
    )
    df_raw = render_data_filters(df_raw, key_prefix="automl")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 3 — Target
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 3 · Select Target")
    default_target = next((c for c in TARGET_CANDIDATES if c in all_cols), all_cols[-1])
    target_col = st.selectbox("Target column", all_cols,
                            index=all_cols.index(default_target))
    non_target = [c for c in all_cols if c != target_col]

    # ── Target conversion (numeric targets only) ──────────────────────────────
    # Evaluate on non-NaN values only — target columns often have many NaN rows
    _y_probe  = pd.to_numeric(df_raw[target_col], errors="coerce")
    _y_dropna = _y_probe.dropna()
    _n_unique    = _y_dropna.nunique()
    _frac_nonint = (_y_dropna % 1 != 0).mean() if len(_y_dropna) > 0 else 0
    _is_numeric_target = (
        len(_y_dropna) >= 5             # at least 5 non-NaN values
        and _n_unique > 20              # enough unique values
        and (
            _frac_nonint >= 0.02        # has fractional parts (e.g. 151.25)
            or _n_unique / max(len(_y_dropna), 1) > 0.50
        )
    )
    if _is_numeric_target:
        st.markdown("#### 🎯 Target Conversion")
        st.caption("Target is numeric — optionally bin into classes.")
        y_converted, task_from_conversion = _render_target_conversion(df_raw, target_col)
    else:
        y_converted = df_raw[target_col].copy()
        task_from_conversion = "classification"
        _vc = y_converted.dropna().value_counts()
        st.info(
            f"🏷️ Target **{target_col}** is categorical "
            f"({_vc.shape[0]} classes: {', '.join(str(c) for c in _vc.index[:5])}"
            f"{'…' if len(_vc) > 5 else ''}) → task set to **classification**."
        )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 4 — Chemical component groups
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 4 · Chemical Component Groups")
    st.caption("Default columns pre-selected from uploaded file. Add or remove freely.")

    sel_groups: dict[str, list[str]] = {}
    grp_items = [g for g in GROUPS if g not in ("Oil","Brine","Process")]

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
                sel_groups[grp] = st.multiselect(
                    f"Columns for {grp}",
                    options=non_target,
                    default=_avail(GROUPS[grp], non_target),
                    key=f"grp_{grp}",
                    label_visibility="collapsed",
                )

    st.markdown("##### 🌊 Brine & 🛢️ Oil")
    bp_cols = st.columns(2)
    for ci, grp in enumerate(["Oil","Brine"]):
        with bp_cols[ci]:
            clr = GRP_CLR.get(grp, "#607D8B")
            st.markdown(
                f'<span class="group-pill" style="background:{clr}22;color:{clr};'
                f'border:1px solid {clr}66">{grp}</span>',
                unsafe_allow_html=True,
            )
            sel_groups[grp] = st.multiselect(
                f"Columns for {grp}",
                options=non_target,
                default=_avail(GROUPS[grp], non_target),
                key=f"grp_{grp}",
                label_visibility="collapsed",
            )

    st.markdown("##### ⚙️ Process Conditions")
    st.caption("Select one column per process variable.")
    NONE     = "(none)"
    none_opt = [NONE] + non_target

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        _def_temp = next((c for c in non_target
                        if re.sub(r"\s+", " ", _normalise(c))
                        in ("temperature","temperature corrected","temp")), NONE)
        proc_temp = st.selectbox("🌡️ Temperature", none_opt,
                                index=none_opt.index(_def_temp) if _def_temp in none_opt else 0,
                                key="proc_temp")
    with pc2:
        _def_dil = next((c for c in non_target
                        if re.sub(r"\s+", " ", _normalise(c))
                        in ("dilution ratio","dilution ratio corrected","dilution")), NONE)
        proc_dil = st.selectbox("💧 Dilution Ratio", none_opt,
                                index=none_opt.index(_def_dil) if _def_dil in none_opt else 0,
                                key="proc_dil")
    with pc3:
        _def_oilpct = next((c for c in non_target
                            if re.sub(r"\s+", " ", _normalise(c))
                            in ("oil","oil percent","oil pct")), NONE)
        proc_oil_pct = st.selectbox("🛢️ Oil Percent (%)", none_opt,
                                    index=none_opt.index(_def_oilpct) if _def_oilpct in none_opt else 0,
                                    key="proc_oil_pct")

    sel_groups["Process"] = [c for c in [proc_temp, proc_dil, proc_oil_pct] if c != NONE]

    # ── Additional Process Features (free multiselect) ────────────────────────
    # Columns that look like process variables but aren't covered by the three
    # fixed pickers above — e.g. concentrate ratio, pH, pressure, foam volume, etc.
    _proc_already = set(sel_groups["Process"])
    _all_assigned  = set(c for g, cols in sel_groups.items() for c in cols)

    # Auto-suggest: unassigned columns whose name hints at a process variable
    _PROC_HINTS = (
        "ratio", "concentrate", "conc", "ph", "pressure", "flow", "rate",
        "volume", "speed", "rpm", "time", "duration", "cycle", "foam",
        "initial", "temp", "method", "manufacturing",
    )
    _proc_suggestions = [
        c for c in non_target
        if c not in _all_assigned
        and any(h in _normalise(c) for h in _PROC_HINTS)
    ]

    st.markdown("**➕ Additional Process Features**")
    st.caption(
        "Columns that don't fit Temperature / Dilution / Oil — e.g. concentrate ratio, "
        "pH, manufacturing method. Treated as condition columns (NaN → row dropped)."
    )
    proc_extra = st.multiselect(
        "Additional process feature columns",
        options=[c for c in non_target if c not in _proc_already],
        default=_proc_suggestions,
        key="proc_extra",
        label_visibility="collapsed",
    )
    sel_groups["Process"] = list(dict.fromkeys(sel_groups["Process"] + proc_extra))

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
    # Step 5 — Computed features
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

    # Custom engineered features — user picks A, operator (+  or /), B
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
        _fname = f"{_c1} {_op} {_c2}"
        if _c1 in df_work.columns and _c2 in df_work.columns                 and _fname not in df_work.columns:
            _c1_num = _safe_num(df_work[_c1])
            _c2_num = _safe_num(df_work[_c2])
            df_work[_fname] = (_c1_num + _c2_num if _op == "+"
                               else _c1_num / _c2_num.replace(0, float("nan")))
    ratio_cols = [f"{c1} {op} {c2}" for c1, op, c2 in user_ratios
                  if f"{c1} {op} {c2}" in df_work.columns]

    # ── Must-have ratio / sum features ───────────────────────────────────────────
    # These are always computed when their parent sum features exist.
    # Only SUM (A + B) is created; division ratios are optional (user-defined).
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

    # Interactions to EXCLUDE (between-subtype surfactant only — redundant):
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

    # ── Compute BOTH + and / for each pair, pick winner by |correlation| ─────
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
            _op_win, _op_lose = "+",          "/"
        else:
            _winner, _loser   = _ratio_name, _sum_name
            _r_win,  _r_lose  = _r_ratio,    _r_sum
            _op_win, _op_lose = "/",          "+"
        if _winner not in EXCLUDED_IX:
            _best_cols.append(_winner)
        _corr_report.append({
            "Pair":       f"{_num}  ×  {_den}",
            "Winner":     _winner,
            "Winner r":   round(_r_win,        4),
            "Winner |r|": round(abs(_r_win),   4),
            "Loser":      _loser,
            "Loser r":    round(_r_lose,        4),
            "Loser |r|":  round(abs(_r_lose),  4),
        })

    _user_feat_cols = [f"{c1} {op} {c2}" for c1, op, c2 in user_ratios
                       if f"{c1} {op} {c2}" in df_work.columns]
    all_ratio_cols = list(dict.fromkeys(
        [c for c in _best_cols if c not in EXCLUDED_IX] + _user_feat_cols
    ))

    # ── Display competition table ─────────────────────────────────────────────
    if _corr_report:
        st.markdown("**Auto-computed ratio/sum features** — winner selected by |correlation| with target:")

        def _style_row(row):
            return [
                "background-color:#e8f5e9;font-weight:bold"
                if col in ("Winner", "Winner r", "Winner |r|")
                else "color:#aaaaaa"
                for col in row.index
            ]

        _rdf = pd.DataFrame(_corr_report)
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
    # Step 6 — NaN Handling for Chemical Groups
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 6 · NaN Handling")

    with st.container(border=True):
        st.caption(
            "**Chemical groups** (Surfactant, Nanoparticle, Polymer, Citric, "
            "Acid, Antiscalant): choose how missing values are treated.  "
            "All other columns (Brine, Oil, Process, Target) always drop the row."
        )
        nh1, nh2 = st.columns([2, 3])
        chem_nan_strategy = nh1.radio(
            "Chemical group NaN strategy",
            options=["fill_zero", "drop_row"],
            format_func=lambda x: {
                "fill_zero": "✅ Fill with 0  (not added to formulation)",
                "drop_row":  "🗑️ Drop row  (treat as missing experiment)",
            }[x],
            index=0,
            key="chem_nan_strategy",
        )
        with nh2:
            if chem_nan_strategy == "fill_zero":
                st.info(
                    "**Fill with 0:** A missing surfactant or nanoparticle concentration "
                    "means the chemical was not added. The row is kept; NaN → 0."
                )
            else:
                st.warning(
                    "**Drop row:** Any row where a selected chemical group column "
                    "is NaN will be removed. Use when missing = unreliable measurement."
                )

        # Preview impact
        all_chem_cols_sel = [c for g in CHEM_GROUPS for c in sel_groups.get(g, [])
                            if c in df_work.columns]
        if all_chem_cols_sel:
            nan_counts = df_work[all_chem_cols_sel].isnull().sum()
            n_rows_with_nan = df_work[all_chem_cols_sel].isnull().any(axis=1).sum()
            n_total = len(df_work)
            if chem_nan_strategy == "fill_zero":
                st.caption(
                    f"**{n_rows_with_nan:,}** rows have at least one chemical NaN "
                    f"({100*n_rows_with_nan/n_total:.1f}%) → will be filled with 0.  "
                    f"All **{n_total:,}** rows kept."
                )
            else:
                st.caption(
                    f"**{n_rows_with_nan:,}** rows have at least one chemical NaN "
                    f"({100*n_rows_with_nan/n_total:.1f}%) → will be dropped.  "
                    f"**{n_total - n_rows_with_nan:,}** rows remain."
                )
        else:
            st.caption("No chemical group columns selected yet.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 7 — Model settings
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 7 · Model Settings")

    with st.container(border=True):
        ms1, ms2, ms3, ms4 = st.columns(4)
        n_estimators = ms1.number_input("Trees", 50, 1000, 300, 50, key="n_est")
        max_depth    = ms2.number_input("Max depth (0=unlimited)", 0, 30, 0, 1, key="mdepth")
        max_depth    = None if max_depth == 0 else int(max_depth)
        random_state = ms3.number_input("Random seed", 0, 999, 42, 1, key="rseed")

        # Task — driven by target conversion; override available
        task_override = ms4.selectbox(
            "Task type",
            ["From target conversion", "regression", "classification"],
            key="task_override",
        )
        task = task_from_conversion if task_override == "From target conversion" else task_override
        st.caption(f"From target conversion: **{task_from_conversion}** → Using: **{task}**")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 7 — SHAP plot configuration (mirrors 2D/3D config in EDA page)
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 8 · SHAP Plot Configuration")

    # Feature pool
    individual_selected = list(dict.fromkeys(
        c for cols in sel_groups.values() for c in cols
    ))
    oil_cols_in_data = _avail(GROUPS["Oil"], non_target)

    MUST_HAVE_X = [
        "Anionic (All Types)", "Nonionic (All Types)", "Zwitterionic (All Types)",
        "Sum Surfactant", "Nanoparticle (All Types)", "Polymer (All Types)",
        "Acid (All Types)", "Citric (All Types)",
    ]
    must_have_present = [f for f in MUST_HAVE_X if f in df_work.columns]

    # x_candidates: individual cols + sum features + all ratio/sum cols (excl. noise)
    x_candidates     = list(dict.fromkeys(
        individual_selected + must_have_present + computed_sums + all_ratio_cols
    ))
    # color_candidates: same + oil properties
    color_candidates = list(dict.fromkeys(
        individual_selected + computed_sums + all_ratio_cols + oil_cols_in_data
    ))

    # All columns available in final df_work
    all_x_candidates = list(dict.fromkeys(
        individual_selected + must_have_present + computed_sums + all_ratio_cols
    ))
    all_color_candidates = list(dict.fromkeys(
        individual_selected + must_have_present + computed_sums
        + all_ratio_cols + oil_cols_in_data
    ))

    with st.container(border=True):
        st.markdown("##### 📊 SHAP Plot Configuration")

        # Single X-axis multiselect: any feature from final df
        st.markdown("**X-axis features** (one folder per feature):")
        st.caption("Select any individual, sum, or ratio feature from the final dataset.")
        _X_DEFAULTS = ["APG (%)", "AOS (%)", "CapB (%)", "Divalent", "Monovalent"]
        _x_def = _avail(_X_DEFAULTS, all_x_candidates)
        x_feats_sf1 = st.multiselect(
            "X-axis features",
            options=all_x_candidates,
            default=list(dict.fromkeys(must_have_present + _x_def)),
            key="x_feats_extra",
            label_visibility="collapsed",
        )
        x_feats_sf2: list[str] = []   # kept for compat, unused

        st.markdown("**Colour-by features** (peers + these, per folder):")
        st.caption("Select any column — individual, sum, ratio, or oil property.")
        _COLOR_DEFAULTS = ["Dilution Ratio", "Dilution Ratio_Corrected"]
        _c_def          = _avail(_COLOR_DEFAULTS, all_color_candidates)
        color_feats     = st.multiselect(
            "Colour features",
            options=all_color_candidates,
            default=_c_def or all_color_candidates[:min(3, len(all_color_candidates))],
            key="color_feats",
            label_visibility="collapsed",
        )

    def _n_cols(x):
        peers  = [c for c in x_feats_sf1 if c != x]
        extras = [c for c in color_feats  if c != x and c not in peers]
        return len(peers) + len(extras)

    n_plots = sum(_n_cols(x) for x in x_feats_sf1 if x in df_work.columns)
    st.info(f"Will generate **{n_plots}** SHAP plots across **{len(x_feats_sf1)}** X-axis folders.")
    st.divider()

    # ── Data overview before training ─────────────────────────────────────────────
    with st.expander("📋 Data Overview (before training)", expanded=True):

        # Simulate preprocessing to show what will actually go into the model
        _all_feat_cols = list(dict.fromkeys(
            c for g, cols in sel_groups.items() for c in cols
        ))
        _feature_cols_prev = list(dict.fromkeys(
            _all_feat_cols + computed_sums + all_ratio_cols
        ))
        _feature_cols_prev = [c for c in _feature_cols_prev
                            if c in df_work.columns and c != target_col]
        _chem_cols = [c for g in CHEM_GROUPS for c in sel_groups.get(g, [])
                    if c in df_work.columns]
        _cond_cols = [c for g in COND_GROUPS for c in sel_groups.get(g, [])
                    if c in df_work.columns]

        # Apply NaN policy (mirror of preprocess() — no scaling)
        _prev = df_work[_feature_cols_prev + [target_col]].copy()
        for col in _chem_cols:
            if col in _prev.columns:
                _prev[col] = _safe_num(_prev[col])
                if chem_nan_strategy == "fill_zero":
                    _prev[col] = _prev[col].fillna(0)
        if _cond_cols:
            _prev = _prev.dropna(subset=[c for c in _cond_cols if c in _prev.columns])
        if chem_nan_strategy == "drop_row":
            _chem_prev = [c for c in _chem_cols if c in _prev.columns]
            if _chem_prev:
                _prev = _prev.dropna(subset=_chem_prev)
        # For categorical targets _safe_num returns NaN for string values —
        # only coerce to numeric for regression; keep strings for classification
        _prev_target_num = _safe_num(_prev[target_col])
        if _prev_target_num.notna().mean() > 0.5:
            _prev[target_col] = _prev_target_num
        _prev = _prev.dropna(subset=[target_col])

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows into model", f"{len(_prev):,}")
        m2.metric("Rows dropped",     f"{len(df_raw) - len(_prev):,}")
        m3.metric("Features",         len(_feature_cols_prev))
        _tgt_num = _safe_num(_prev[target_col])
        if _tgt_num.notna().sum() > 0:
            m4.metric("Target median", f"{_tgt_num.median():.2f}")
            m5.metric("Target range",  f"{_tgt_num.min():.1f} – {_tgt_num.max():.1f}")
        else:
            _vc = _prev[target_col].value_counts()
            m4.metric("Target classes", str(_vc.shape[0]))
            m5.metric("Largest class",  f"{_vc.index[0]} ({_vc.iloc[0]})")

        # NaN summary per column
        _nan_pct = (_prev[_feature_cols_prev].isnull().mean() * 100).round(1)
        _nan_remaining = _nan_pct[_nan_pct > 0]
        if _nan_remaining.empty:
            st.success("✅ No remaining NaN in any feature column.")
        else:
            st.warning(f"⚠️ {len(_nan_remaining)} feature(s) still have NaN — "
                    "will be filled with 0 before scaling.")
            st.dataframe(_nan_remaining.rename("NaN %").to_frame(),
                        use_container_width=True)

        # # Target distribution
        # oc1, oc2 = st.columns(2)
        # with oc1:
        #     import matplotlib.pyplot as _plt
        #     fig_prev, ax_prev = _plt.subplots(figsize=(5, 3), dpi=120)
        #     ax_prev.hist(_prev[target_col].dropna(), bins=40,
        #                 color="#1565C0", edgecolor="none", alpha=0.8)
        #     ax_prev.set_xlabel(target_col, fontsize=8)
        #     ax_prev.set_ylabel("Count", fontsize=8)
        #     ax_prev.set_title("Target Distribution", fontsize=9, fontweight="bold")
        #     ax_prev.tick_params(labelsize=7)
        #     fig_prev.tight_layout()
        #     st.pyplot(fig_prev, use_container_width=True)
        #     _plt.close(fig_prev)

        # with oc2:
        #     # Top 10 features by non-zero count (most informative)
        #     _nonzero = (_prev[_feature_cols_prev] != 0).sum().sort_values(ascending=False).head(10)
        #     fig_nz, ax_nz = _plt.subplots(figsize=(5, 3), dpi=120)
        #     ax_nz.barh(_nonzero.index[::-1], _nonzero.values[::-1], color="#FF8F00")
        #     ax_nz.set_xlabel("Non-zero rows", fontsize=8)
        #     ax_nz.set_title("Top 10 Features by Non-zero Count", fontsize=9, fontweight="bold")
        #     ax_nz.tick_params(labelsize=6)
        #     fig_nz.tight_layout()
        #     st.pyplot(fig_nz, use_container_width=True)
        #     _plt.close(fig_nz)

        # Feature preview table
        #with st.expander("Feature matrix preview (first 10 rows, unscaled)"):
        st.dataframe(_prev[_feature_cols_prev], use_container_width=True)
            
    # ══════════════════════════════════════════════════════════════════════════════
    # Step 8 — Save destination
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 9 · Save Destination")
    save_mode = st.radio("How to get the plots?",
                        ["⬇️ Download as ZIP", "💾 Save to disk path"],
                        horizontal=True, key="save_mode")
    disk_path = ""
    if "disk" in save_mode:
        disk_path = st.text_input("Destination folder", value=str(Path.home() / "foam_shap_plots"))
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 9 — Train & Generate
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown("### 10 · Train Model & Generate SHAP Plots")

    if not x_feats_sf1:
        st.warning("Select at least one X-axis feature.")
        st.stop()
    if not color_feats:
        st.warning("Select at least one colour feature.")
        st.stop()

    if st.button("🚀 Train Model & Generate Plots", type="primary", use_container_width=True):

        # ── Determine chem / cond cols for preprocessing ──────────────────────
        all_feature_cols = list(dict.fromkeys(
            c for g, cols in sel_groups.items() for c in cols
        ) )
        # Add computed features to the working df before preprocessing
        df_for_model = df_work.copy()
        df_for_model[target_col] = y_converted.values  # use converted target

        chem_cols = [c for g in CHEM_GROUPS for c in sel_groups.get(g, []) if c in df_for_model.columns]
        cond_cols = [c for g in COND_GROUPS for c in sel_groups.get(g, []) if c in df_for_model.columns]

        # Feature columns = selected raw + computed sums + all ratio/sum cols
        feature_cols = list(dict.fromkeys(
            all_feature_cols + computed_sums + all_ratio_cols
        ))
        feature_cols = [c for c in feature_cols if c in df_for_model.columns and c != target_col]

        with st.spinner("Preprocessing…"):
            try:
                X_scaled, y, feat_names, scaler = preprocess(
                    df_for_model, feature_cols, target_col,
                    chem_cols, cond_cols, task,
                    chem_nan_strategy=chem_nan_strategy,
                )
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                st.stop()

        st.success(f"✅ Preprocessed: **{len(X_scaled):,} rows × {len(feat_names)} features**  |  Task: **{task}**")

        # Original-scale X (for SHAP x-axis display)
        X_orig = df_for_model.loc[X_scaled.index, [c for c in feat_names
                                                    if c in df_for_model.columns]].copy()
        # For one-hot encoded cols not in df_for_model, use scaled values (already 0/1)
        for col in feat_names:
            if col not in X_orig.columns:
                X_orig[col] = X_scaled[col]

        # ── Train ─────────────────────────────────────────────────────────────
        import hashlib as _hl
        _hash_src = str(X_scaled.shape) + str(list(X_scaled.columns)) + str(int(n_estimators))
        X_hash = _hl.md5(_hash_src.encode()).hexdigest()
        model, X_shap, shap_vals = train_model(
            X_hash, X_scaled, y, task,
            int(n_estimators), max_depth, int(random_state),
        )
        X_shap_orig = X_orig.loc[X_shap.index].copy()

        # Model metrics
        from sklearn.metrics import r2_score, accuracy_score
        y_pred = model.predict(X_scaled)
        if task == "regression":
            score = r2_score(y, y_pred)
            st.metric("R² (train)", f"{score:.4f}")
        else:
            score = accuracy_score(y, y_pred)
            st.metric("Accuracy (train)", f"{score:.4f}")

        # Mean |SHAP| importance bar
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        imp_df = pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_abs_shap})
        imp_df = imp_df.sort_values("Mean |SHAP|", ascending=False).head(20)
        with st.expander("📊 Top 20 Features by Mean |SHAP|", expanded=True):
            st.bar_chart(imp_df.set_index("Feature"))

        # ── Generate SHAP plots ───────────────────────────────────────────────
        # Only plot for features that exist in feat_names
        # color_feats_avail: features available in X_shap_orig or as encoded columns
        color_feats_avail = [f for f in color_feats if f in X_shap_orig.columns or f in feat_names]

        SF_PERF = "Model Performance"
        x_avail = [f for f in x_feats_sf1 if f in feat_names]

        figures: dict[str, bytes] = {}
        f1: dict[str, bytes] = {}

        with st.spinner("Generating SHAP dependence plots…"):
            f1 = generate_shap_plots(
                X_shap_orig, shap_vals, feat_names,
                x_avail, color_feats_avail, "",
            )
            figures.update(f1)

        # ── Model Performance folder ──────────────────────────────────────────
        with st.spinner("Generating Model Performance plots…"):
            perf_figs = _generate_model_performance(
                model, X_scaled, y, task, feat_names, shap_vals, X_shap_orig, target_col
            )
            figures.update({f"{SF_PERF}/{k}": v for k, v in perf_figs.items()})

        # ── Most Important Features folder ────────────────────────────────────
        # Top N features by mean |SHAP| — one SHAP dependence plot per feature,
        # coloured by the feature with next-highest correlation to its SHAP values
        SF_TOP = "Most Important Features"
        with st.spinner("Generating Most Important Features SHAP plots…"):
            _n_top  = min(20, len(feat_names))
            _mean_abs = np.abs(shap_vals).mean(axis=0)
            _top_idx  = np.argsort(_mean_abs)[::-1][:_n_top]
            _top_feats = [feat_names[i] for i in _top_idx]

            def _best_color(x_feat, x_idx):
                """Pick colour = top feat with highest |corr| to SHAP of x_feat."""
                sv_x = shap_vals[:, x_idx]
                best_r, best_f = 0.0, None
                for other in _top_feats:
                    if other == x_feat: continue
                    if other not in X_shap_orig.columns: continue
                    ov = pd.to_numeric(X_shap_orig[other], errors="coerce").values
                    ok = ~(np.isnan(ov) | np.isnan(sv_x))
                    if ok.sum() < 5: continue
                    r = abs(float(np.corrcoef(ov[ok], sv_x[ok])[0, 1]))
                    if not np.isnan(r) and r > best_r:
                        best_r, best_f = r, other
                return best_f or (color_feats_avail[0] if color_feats_avail else _top_feats[1] if len(_top_feats) > 1 else x_feat)

            top_figs: dict[str, bytes] = {}
            for rank, (xi, x_feat) in enumerate(zip(_top_idx, _top_feats), 1):
                if x_feat not in X_shap_orig.columns: continue
                col_feat = _best_color(x_feat, xi)
                title = (f"Top #{rank}  |  X={x_feat}  |  "
                         f"Colour={col_feat}  (|r|={abs(float(np.corrcoef(shap_vals[:,xi], pd.to_numeric(X_shap_orig.get(col_feat, pd.Series([0])), errors='coerce').fillna(0).values)[0,1])):.3f})")
                fig = _shap_dependence(X_shap_orig, shap_vals, feat_names,
                                       x_feat, col_feat, title)
                fname = f"{rank:02d}_{_slugify(x_feat)}_Color_{_slugify(col_feat)}.png"
                top_figs[fname] = _render_to_bytes(fig)

            figures.update({f"{SF_TOP}/{k}": v for k, v in top_figs.items()})

        st.success(
            f"✅ Generated **{len(f1)}** SHAP + "
            f"**{len(perf_figs)}** performance + "
            f"**{len(top_figs)}** top-feature = **{len(figures)}** plots total."
        )

        # ── Save / Download ───────────────────────────────────────────────────
        if "disk" in save_mode:
            if not disk_path.strip():
                st.error("Enter a valid path.")
            else:
                try:
                    n = save_figures_to_disk(figures, disk_path.strip())
                    st.success(f"💾 Saved **{n}** plots to `{disk_path.strip()}/SHAP Plots/`")
                except Exception as e:
                    st.error(f"Save failed: {e}")
        else:
            zip_bytes = figures_to_zip(figures)
            st.download_button(
                "⬇️ Download all SHAP plots as ZIP",
                data=zip_bytes,
                file_name="foam_shap_plots.zip",
                mime="application/zip",
                use_container_width=True,
            )
            st.caption("ZIP: `SHAP Plots / Effect of individual component… / <feature> / SHAP_…_Color_….png`")

        # ── Preview first 4 plots ─────────────────────────────────────────────
        st.divider()
        st.markdown("#### Preview (first 4 plots)")
        prev_cols = st.columns(2)
        for pi, (path_key, png_bytes) in enumerate(list(figures.items())[:4]):
            with prev_cols[pi % 2]:
                st.image(io.BytesIO(png_bytes), caption=path_key.split("/")[-1],
                        use_column_width=True)