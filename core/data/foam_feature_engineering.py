"""
utils/foam_feature_engineering.py
==================================
Self-contained foam-chemistry feature engineering module.

Public API
----------
build_foam_features(df, sel, include_ix, custom_cols)
    → (X_engineered: DataFrame, feat_to_group: dict)

render_feature_engineering_ui(df, key_prefix, header)
    → (df_out: DataFrame, feature_cols: list[str], feat_to_group: dict)
    Renders the full Streamlit UI (group pickers, process selectors,
    interactions toggle, overview) and returns results.
    key_prefix ensures widget keys are unique per page
    (e.g. "prep", "eda", "custom_tab").

Constants exported
------------------
DEF_NANO, DEF_ANIONIC, DEF_NONION, DEF_ZW, DEF_POLY, DEF_CITRIC,
DEF_ACID, DEF_ANTI, DEF_BRINE, DEF_OIL, DEF_PROCESS
GRP_CLR          — group → hex colour
CHEM_GROUPS      — set of group names whose NaN = 0 (not condition groups)
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Default column lists
# ─────────────────────────────────────────────────────────────────────────────
DEF_NANO    = ["HS (%)", "BLH5 (%)", "HSA (%)"]
DEF_ANIONIC = ["AOS (%)", "DB45 (%)", "alpha-step (%)", "SDS (%)", "SLES (%)", "n. LABS (%)"]
DEF_NONION  = ["APG (%)", "decyl glucoside (%)", "caprylyl glucoside (%)",
               "Tween 80 (%)", "PG (%)"]
DEF_ZW      = ["CapB (%)", "2C (%)", "DM (%)","Cola 2C (%)",  "Cola SLAA (%)", "Cola SC (%)","amine oxide (%)",
               "Cola LMB (%)", "Amphosol 1c (%)", "SC (%)", "LBHP (%)", "CS50 (%)","LAO (%)"]
DEF_POLY    = ["HPAM (%)", "xanthan gum (%) ", "Guar Gum (%)", "FPAM (%)", "PAA (%)", "PA (%)"]
DEF_CITRIC  = ["Citric (%)",
               "31.2(%) citric+ 13.3(%) KOH  (pH not adjusted, pH=4.46 )",
               "31.2% citric+ 13.3% KOH  (adjusted pH=4) (%)", "31.2% citric+ 13.3% KOH (pH=4) (%)",
               "38.1% citric+   KOH  (pH=5) (%)",
               "potassium citrate (9.7%)/citric acid buffer (19.22%)  pH=3 (%)",
               "potassium citrate (%)"]
DEF_ACID    = ["EDTA (%)", "etidronic acid (%)", "acetic acid (%)"]
DEF_ANTI    = ["Mem 2000-clear tech (%)", "Mem 2500-clear tech (%)",
               "Mem 4000-clear tech (%)", "Mem 3500-clear tech (%)",
               "Mem 3000-clear tech (%)", "ClearHib 1000 (%)"]
DEF_BRINE   = ["Divalent", "Monovalent"]
DEF_OIL     = ["Linear paraffins", "Branched paraffins", "Cycloparaffins",
               "Aromatics", "Oxygenates", "Biomarker-like compounds",
               "Sulfur compounds", "Wax fraction (C24+)"]
DEF_PROCESS = ["Temperature", "Dilution Ratio", "Oil  (%)",
               "concentrate manufacturing method (Ratio)",
               "Initial Foam Temp (dilution Temp) "]

# Pre-selected by default in "Additional Process Features" (Temperature /
# Dilution / Oil % have their own dedicated pickers already; this is for
# extras worth defaulting to). Covers both naming conventions seen across
# datasets (Oil workbook uses the _Normalized suffix, others don't).
DEF_PROC_EXTRA = [
    "concentrate manufacturing method (Ratio)_Normalized",
    "concentrate manufacturing method (Ratio)",
]

# One pill + multiselect per individual group, in display order (2-column grid).
# (var_name, group_label, default_columns) — var_name matches the local
# variable render_feature_engineering_ui() assigns it to (sel_nano, sel_an, …)
# so the "Custom Engineered Features" section below it keeps working unchanged.
_GROUP_SPECS: list[tuple[str, str, list[str]]] = [
    ("sel_nano",   "Nanoparticle",  DEF_NANO),
    ("sel_an",     "Anionic",       DEF_ANIONIC),
    ("sel_non",    "Nonionic",      DEF_NONION),
    ("sel_zw",     "Zwitterionic",  DEF_ZW),
    ("sel_poly",   "Polymer",       DEF_POLY),
    ("sel_citric", "Citric",        DEF_CITRIC),
    ("sel_acid",   "Acid/Chelant",  DEF_ACID),
    ("sel_anti",   "Antiscalant",   DEF_ANTI),
]

# Groups whose NaN means "not added" → fill 0 before building features
CHEM_GROUPS = {
    "Nanoparticle", "Anionic", "Nonionic", "Zwitterionic",
    "Polymer", "Citric", "Acid/Chelant", "Antiscalant",
}

GRP_CLR = {
    "Nanoparticle":  "#1565C0", "Anionic":       "#E53935",
    "Nonionic":      "#FB8C00", "Zwitterionic":  "#8E24AA",
    "Surfactant":    "#FF8F00", "Polymer":       "#6A1B9A",
    "Citric":        "#2E7D32", "Acid/Chelant":  "#C62828",
    "Antiscalant":   "#00695C", "Brine":         "#4527A0",
    "Oil":           "#BF360C", "Process":       "#546E7A",
    "Interaction":   "#795548", "Custom":        "#37474F",
}

# Oil column → clean feature name
_OIL_LBL = {
    "Linear paraffins":          "Oil Linear Paraffins",
    "Branched paraffins":        "Oil Branched Paraffins",
    "Cycloparaffins":            "Oil Cycloparaffins",
    "Aromatics":                 "Oil Aromatics",
    "Oxygenates":                "Oil Oxygenates",
    "Biomarker-like compounds":  "Oil Biomarkers",
    "Sulfur compounds":          "Oil Sulfur",
    "Wax fraction (C24+)":       "Oil Wax Fraction",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_label(col: str) -> str:
    s = str(col).strip()
    s = re.sub(r'\s*\(%\)\s*$', '', s)
    s = re.sub(r'\(%\)', '', s)
    s = re.sub(r'%', ' pct', s)
    s = re.sub(r'[()]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _normalise(name: str) -> str:
    """Lower-case/strip and drop a trailing '(%)' so DEF_* candidate names
    still match columns whose formatting differs slightly (case, spacing)."""
    s = str(name).strip().lower()
    s = re.sub(r'\s*\(%\)\s*$', '', s).strip()
    return s


def _avail(defaults: list[str], df_cols: list[str]) -> list[str]:
    norm_to_col = {_normalise(c): c for c in df_cols}
    matched, seen = [], set()
    for cand in defaults:
        key = _normalise(cand)
        if key in norm_to_col:
            actual = norm_to_col[key]
            if actual not in seen:
                matched.append(actual)
                seen.add(actual)
    return matched


def _safe_sum_zero(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Sum cols treating NaN as 0 (chemical groups: not added = zero)."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0.0, index=df.index)
    result = pd.Series(0.0, index=df.index)
    for c in present:
        result += pd.to_numeric(df[c], errors="coerce").fillna(0)
    return result


def _guard(feats: dict, a: str, b: str) -> bool:
    """True when both features exist and each has < 10% NaN."""
    if a not in feats or b not in feats:
        return False
    return feats[a].isna().mean() <= 0.10 and feats[b].isna().mean() <= 0.10


def _interact_sum(feats: dict, a: str, b: str, name: str) -> None:
    """a + b  — captures combined concentration effect."""
    if _guard(feats, a, b):
        feats[name] = feats[a] + feats[b]


def _interact_ratio(feats: dict, a: str, b: str, name: str) -> None:
    """a / b  — captures relative proportion; b=0 → NaN (not filled)."""
    if _guard(feats, a, b):
        feats[name] = feats[a] / feats[b].replace(0, float("nan"))


def _method_onehot(df: pd.DataFrame, col: str,
                    feats: dict, fg: dict) -> None:
    """One-hot encode a categorical Process column (e.g. manufacturing
    method) into feats/fg dicts. Feature name prefix is derived from the
    column itself, so this works for any text Process column, not just a
    single designated 'method' role."""
    mc     = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    prefix = _clean_label(col).replace(" ", "_") or "Process_Category"
    for val in sorted(mc.unique(), key=str):
        slug = re.sub(r'[^a-z0-9]+', '_', val.lower())[:40].strip('_')
        name = f"{prefix}_{slug}"
        feats[name] = (mc == val).astype(float).reset_index(drop=True)
        fg[name]    = "Process"


# ─────────────────────────────────────────────────────────────────────────────
#  Core builder — pure Python, no Streamlit
# ─────────────────────────────────────────────────────────────────────────────

def build_foam_features(
    df: pd.DataFrame,
    sel: dict,
    include_ix: bool,        # kept for backward-compat; ignored when ix_sum/ix_ratio used
    custom_cols: list[str],
    ix_sum: bool = False,
    ix_ratio: bool = False,
    custom_operations: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build engineered feature DataFrame from column-group selections.

    Parameters
    ----------
    df : DataFrame
        Input data (chemical NaN already filled 0; condition NaN rows already
        dropped — caller's responsibility).
    sel : dict
        Keys: group names (see CHEM_GROUPS + "Brine", "Oil", "Process",
        "_proc_map"). Values: list of column names (except "_proc_map" which
        is a dict of {role: column_name}).
    include_ix : bool
        Legacy flag — set ix_sum/ix_ratio explicitly instead.
        If both ix_sum and ix_ratio are False this flag is used as a fallback
        to enable sum interactions for backward compatibility.
    custom_cols : list[str]
        Extra columns to include as-is (prefixed "Custom ").
    ix_sum : bool
        Add sum interactions  (A + B) — combined concentration effect.
    ix_ratio : bool
        Add ratio interactions (A / B) — relative proportion effect.

    Returns
    -------
    X : DataFrame  — engineered feature matrix (rows reset to 0..N)
    fg : dict      — feature_name → group_name
    """
    # backward-compat: old callers pass include_ix=True without ix_sum/ix_ratio
    if include_ix and not ix_sum and not ix_ratio:
        ix_sum = True
    feats: dict[str, pd.Series] = {}
    fg:    dict[str, str]       = {}

    def add(name: str, series: pd.Series, group: str) -> None:
        s = series.reset_index(drop=True) if hasattr(series, "reset_index") else series
        feats[name] = s
        fg[name]    = group

    # ── 1. Individual raw columns ─────────────────────────────────────────
    used: dict[str, bool] = {}
    for group, cols in sel.items():
        if group.startswith("_"):        # skip _proc_map etc.
            continue
        if not isinstance(cols, list):
            continue
        for col in cols:
            if col not in df.columns:
                continue
            lbl = _clean_label(col)
            if lbl in used:
                lbl = f"{lbl} {group}"
            used[lbl] = True
            add(lbl, pd.to_numeric(df[col], errors="coerce"), group)

    # ── 2. Group totals & fractions ───────────────────────────────────────
    nano_tot   = _safe_sum_zero(df, sel.get("Nanoparticle", []))
    surf_an    = _safe_sum_zero(df, sel.get("Anionic",      []))
    surf_non   = _safe_sum_zero(df, sel.get("Nonionic",     []))
    surf_zw    = _safe_sum_zero(df, sel.get("Zwitterionic", []))
    surf_tot   = surf_an + surf_non + surf_zw
    poly_tot   = _safe_sum_zero(df, sel.get("Polymer",       []))
    citric_tot = _safe_sum_zero(df, sel.get("Citric",        []))
    acid_tot   = _safe_sum_zero(df, sel.get("Acid/Chelant",  []))
    anti_tot   = _safe_sum_zero(df, sel.get("Antiscalant",   []))

    if sel.get("Nanoparticle"):
        add("Nanoparticle (All Types)",  nano_tot,                     "Nanoparticle")
        add("Has Nanoparticle",   (nano_tot > 0).astype(float),  "Nanoparticle")

    if sel.get("Anionic") or sel.get("Nonionic") or sel.get("Zwitterionic"):
        add("Sum Surfactant",        surf_tot,                    "Surfactant")
        add("Anionic (All Types)",      surf_an,                     "Anionic")
        add("Nonionic (All Types)",     surf_non,                    "Nonionic")
        add("Zwitterionic (All Types)", surf_zw,                     "Zwitterionic")
        add("Anionic (All Types)/Sum Surfactant",        surf_an  / (surf_tot+1e-9),  "Anionic")
        add("Nonionic (All Types)/Sum Surfactant",       surf_non / (surf_tot+1e-9),  "Nonionic")
        add("Zwitterionic (All Types)/Sum Surfactant",   surf_zw  / (surf_tot+1e-9),  "Zwitterionic")

    if sel.get("Polymer"):
        add("Polymer (All Types)", poly_tot,                     "Polymer")
        add("Has Polymer",  (poly_tot > 0).astype(float),  "Polymer")

    if sel.get("Citric"):
        add("Citric (All Types)",         citric_tot,                     "Citric")
        add("Citric (All Types) / Sum Surfactant", citric_tot / (surf_tot + 1e-9), "Citric")

    if sel.get("Acid/Chelant"):
        add("Acid (All Types)", acid_tot, "Acid/Chelant")

    if sel.get("Antiscalant"):
        add("Antiscalant (All Types)", anti_tot, "Antiscalant")

    # ── 3. Brine ──────────────────────────────────────────────────────────
    if sel.get("Brine"):
        div_s  = (pd.to_numeric(df["Divalent"],   errors="coerce")
                  if "Divalent"   in df.columns
                  else pd.Series(np.nan, index=df.index))
        mono_s = (pd.to_numeric(df["Monovalent"], errors="coerce")
                  if "Monovalent" in df.columns
                  else pd.Series(np.nan, index=df.index))
        add("Divalent",                  div_s,                   "Brine")
        add("Monovalent",                mono_s,                  "Brine")
        #add("Divalent Monovalent Ratio", div_s / (mono_s + 1e-9), "Brine")
        #add("Log Divalent",              np.log1p(div_s),         "Brine")
        #add("Log Monovalent",            np.log1p(mono_s),        "Brine")

    # ── 4. Oil ────────────────────────────────────────────────────────────
    if sel.get("Oil"):
        for raw in sel["Oil"]:
            lbl = _OIL_LBL.get(raw, "Oil " + _clean_label(raw))
            if raw in df.columns:
                add(lbl, pd.to_numeric(df[raw], errors="coerce"), "Oil")

    # ── 5. Process — role-based (_proc_map) + any extra flat columns ─────
    #
    # _proc_map assigns the core roles (temp/dilution/oil_pct[/init_temp/
    # method]) to clean, fixed feature names. Any OTHER column the caller
    # adds to sel["Process"] (e.g. via an "Additional Process Features"
    # picker) is handled generically here: numeric columns are added
    # as-is, text columns (e.g. manufacturing method) are automatically
    # one-hot encoded. This keeps Process useful without requiring every
    # role to be wired into _proc_map.
    pm = sel.get("_proc_map", {}) or {}
    _proc_role_cols: set[str] = set()

    _ROLE_TO_FEATNAME = {
        "temp":      "Temperature",
        "init_temp": "Initial Foam Temp",
        "dilution":  "Dilution Ratio",
        "oil_pct":   "Oil Percent",
    }
    for role, feat_name in _ROLE_TO_FEATNAME.items():
        col = pm.get(role)
        if col and col in df.columns:
            add(feat_name, pd.to_numeric(df[col], errors="coerce"), "Process")
            _proc_role_cols.add(col)

    if pm.get("method") and pm["method"] in df.columns:
        _method_onehot(df, pm["method"], feats, fg)
        _proc_role_cols.add(pm["method"])

    for extra_col in sel.get("Process", []):
        if extra_col not in df.columns or extra_col in _proc_role_cols:
            continue
        numeric = pd.to_numeric(df[extra_col], errors="coerce")
        if numeric.notna().mean() >= 0.5:
            add(_clean_label(extra_col), numeric, "Process")
        else:
            _method_onehot(df, extra_col, feats, fg)

    # ── 6. Custom columns ─────────────────────────────────────────────────
    for col in custom_cols:
        if col in df.columns:
            lbl = "Custom " + _clean_label(col)
            add(lbl, pd.to_numeric(df[col], errors="coerce"), "Custom")

    # ── 7. Interactions — dynamically built from what is in feats ───────────
    #
    # Four "left-hand side" group totals drive interactions:
    #   Nanoparticle (All Types), Sum Surfactant Anionic,
    #   Zwitterionic (All Types), Nonionic (All Types)
    #
    # Each is crossed with every available "right-hand side" feature:
    #   Sum Surfactant, Sum Surfactant Anionic/Nonionic/Zwitterionic,
    #   Divalent, Monovalent,
    #   Polymer (All Types), Citric (All Types), Acid (All Types),
    #   every Oil column the user selected (Oil Linear Paraffins, Oil Wax Fraction, …)
    #
    # Oil columns are discovered dynamically from feats so they reflect
    # exactly which oil columns the user selected — not a hard-coded list.

    if ix_sum or ix_ratio:

        # Left-hand side groups (each crossed with all RHS)
        _LHS = [
            "Nanoparticle (All Types)",
            "Anionic (All Types)",
            "Zwitterionic (All Types)",
            "Nonionic (All Types)",
        ]

        # Fixed right-hand side features
        _RHS_FIXED = [
            "Sum Surfactant",
            "Anionic (All Types)",
            "Nonionic (All Types)",
            "Zwitterionic (All Types)",
            "Divalent",
            "Monovalent",
            "Polymer (All Types)",
            "Citric (All Types)",
            "Acid (All Types)",
        ]

        # Dynamic oil RHS — every feature in feats whose name starts with "Oil "
        _RHS_OIL = [f for f in feats if f.startswith("Oil ")]

        _RHS_ALL = _RHS_FIXED + _RHS_OIL

        for lhs in _LHS:
            if lhs not in feats:
                continue
            for rhs in _RHS_ALL:
                if rhs not in feats:
                    continue
                if lhs == rhs:
                    continue
                if ix_sum:
                    _interact_sum(feats, lhs, rhs, f"{lhs} + {rhs}")
                if ix_ratio:
                    _interact_ratio(feats, lhs, rhs, f"{lhs} / {rhs}")

    # ── 7B. User-defined custom operations ─────────────────────────────

    if custom_operations:

        # UI candidates are raw column names OR already-clean feature names.
        # Build a lookup: raw_col_or_clean_name → key actually in feats.
        # Priority: exact match first, then cleaned-label match.
        _feat_keys = set(feats.keys())
        _clean_to_feat: dict[str, str] = {}
        for fk in _feat_keys:
            _clean_to_feat[fk] = fk                     # identity
        # also map every raw column → its cleaned label (if present in feats)
        for group, cols in sel.items():
            if group.startswith("_") or not isinstance(cols, list):
                continue
            for col in cols:
                lbl = _clean_label(col)
                if lbl in _feat_keys:
                    _clean_to_feat[col] = lbl

        def _resolve(name: str) -> str | None:
            """Return the key in feats for a UI candidate name, or None."""
            if name in _feat_keys:
                return name
            return _clean_to_feat.get(name)

        for op in custom_operations:

            a_raw    = op.get("a")
            b_raw    = op.get("b")
            operator = op.get("op")

            a = _resolve(a_raw)
            b = _resolve(b_raw)

            if a is None or b is None:
                continue

            try:

                if operator == "+":
                    name = f"{a_raw} + {b_raw}"
                    feats[name] = feats[a] + feats[b]
                    fg[name] = "Custom"

                elif operator == "/":
                    name = f"{a_raw} / {b_raw}"
                    feats[name] = feats[a] / feats[b].replace(0, np.nan)
                    fg[name] = "Custom"

            except Exception:
                pass

    # ── 8. Assemble ───────────────────────────────────────────────────────
    X = pd.DataFrame(feats).reset_index(drop=True)
    X = X.loc[:, X.nunique() > 0]           # drop constant columns
    X = X.loc[:, ~X.isnull().all()]         # drop all-NaN columns
    fg = {k: v for k, v in fg.items() if k in X.columns}
    return X, fg


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit UI — renders pickers + overview, calls build_foam_features
# ─────────────────────────────────────────────────────────────────────────────

def render_feature_engineering_ui(
    df: pd.DataFrame,
    key_prefix: str = "fe",
    header: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Render the full feature-engineering UI and return results.

    Parameters
    ----------
    df : DataFrame
        Current working data (after filtering).
    key_prefix : str
        Unique prefix for all widget keys — use different values per page
        to avoid Streamlit key collisions.
        Examples: "prep", "eda", "predict_tab"
    header : bool
        Whether to render the section header.

    Returns
    -------
    df_out      : DataFrame  — df with engineered columns merged in
    feature_cols: list[str]  — names of engineered feature columns
    feat_to_group: dict      — feature_name → group_name
    """
    if header:
        st.markdown("### ⚗️ Feature Engineering")
        st.caption(
            "Select chemical groups — group totals, fractions and "
            "interactions are built automatically."
        )

    all_cols = list(df.columns)

    with st.container(border=True):
        use_foam = st.checkbox(
            "Use foam chemical group engineering",
            value=st.session_state.get(f"{key_prefix}_enabled", True),
            key=f"{key_prefix}_enabled",
            help="Builds group totals, fractions and interaction features automatically.",
        )

        if not use_foam:
            st.info("Feature engineering disabled — raw columns used as-is.")
            return df, [], {}

        st.divider()

        # ── Chemical Component Groups (one pill + multiselect per group) ───
        st.markdown("##### 🧴 Chemical Component Groups")
        st.caption("Default columns pre-selected from uploaded file. Add or remove freely.")
        _group_vars: dict[str, list[str]] = {}
        cols_ui = st.columns(2)
        for gi, (var_name, grp_label, defaults) in enumerate(_GROUP_SPECS):
            with cols_ui[gi % 2]:
                clr = GRP_CLR.get(grp_label, "#607D8B")
                st.markdown(
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
                    f'font-size:0.75rem;font-weight:600;margin:2px;background:{clr}22;color:{clr};'
                    f'border:1px solid {clr}66">{grp_label}</span>',
                    unsafe_allow_html=True,
                )
                _group_vars[var_name] = st.multiselect(
                    f"Columns for {grp_label}",
                    options=all_cols,
                    default=_avail(defaults, all_cols),
                    format_func=_clean_label,
                    key=f"{key_prefix}_grp_{var_name}",
                    label_visibility="collapsed",
                )
        sel_nano   = _group_vars["sel_nano"]
        sel_an     = _group_vars["sel_an"]
        sel_non    = _group_vars["sel_non"]
        sel_zw     = _group_vars["sel_zw"]
        sel_poly   = _group_vars["sel_poly"]
        sel_citric = _group_vars["sel_citric"]
        sel_acid   = _group_vars["sel_acid"]
        sel_anti   = _group_vars["sel_anti"]

        # ── Brine & Oil composition ────────────────────────────────────────
        st.markdown("##### 🌊 Brine & 🛢️ Oil composition")
        bp_cols = st.columns(2)
        with bp_cols[0]:
            clr = GRP_CLR.get("Oil", "#607D8B")
            st.markdown(
                f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
                f'font-size:0.75rem;font-weight:600;margin:2px;background:{clr}22;color:{clr};'
                f'border:1px solid {clr}66">Oil</span>',
                unsafe_allow_html=True,
            )
            sel_oil = st.multiselect(
                "Oil", all_cols, default=_avail(DEF_OIL, all_cols),
                format_func=_clean_label, key=f"{key_prefix}_grp_oil",
                label_visibility="collapsed",
            )
        with bp_cols[1]:
            clr = GRP_CLR.get("Brine", "#607D8B")
            st.markdown(
                f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
                f'font-size:0.75rem;font-weight:600;margin:2px;background:{clr}22;color:{clr};'
                f'border:1px solid {clr}66">Brine</span>',
                unsafe_allow_html=True,
            )
            sel_brine = st.multiselect(
                "Brine", all_cols, default=_avail(DEF_BRINE, all_cols),
                format_func=_clean_label, key=f"{key_prefix}_grp_brine",
                label_visibility="collapsed",
            )

        # ── Process Conditions — 3 fixed pickers + free-form extras ───────
        st.markdown("##### ⚙️ Process Conditions")
        st.caption("Select one column per process variable.")
        NONE     = "(none)"
        none_opt = [NONE] + all_cols
        col_map  = {c.lower().strip(): c for c in all_cols}

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            _def_temp = next((col_map[c] for c in
                              ["temperature_corrected", "temperature"]
                              if c in col_map), None)
            proc_temp = st.selectbox("🌡️ Temperature", none_opt,
                                      index=none_opt.index(_def_temp) if _def_temp in none_opt else 0,
                                      format_func=_clean_label,
                                      key=f"{key_prefix}_proc_temp")
        with pc2:
            _def_dil = next((col_map[c] for c in
                             ["dilution ratio_corrected", "dilution ratio"]
                             if c in col_map), None)
            proc_dil = st.selectbox("💧 Dilution Ratio", none_opt,
                                     index=none_opt.index(_def_dil) if _def_dil in none_opt else 0,
                                     format_func=_clean_label,
                                     key=f"{key_prefix}_proc_dil")
        with pc3:
            _def_oilpct = next((col_map[c] for c in
                                ["oil  (%)_corrected", "oil  (%)", "oil (%)"]
                                if c in col_map), None)
            proc_oil_pct = st.selectbox("🛢️ Oil Percent (%)", none_opt,
                                         index=none_opt.index(_def_oilpct) if _def_oilpct in none_opt else 0,
                                         format_func=_clean_label,
                                         key=f"{key_prefix}_proc_oilpct")

        proc_cols = [c for c in [proc_temp, proc_dil, proc_oil_pct] if c != NONE]

        # ── Additional Process Features (free multiselect, empty by default) ─
        _proc_already = set(proc_cols)
        st.markdown("**➕ Additional Process Features**")
        st.caption(
            "Columns that don't fit Temperature / Dilution / Oil — e.g. concentrate "
            "ratio, pH, manufacturing method, Initial Foam Temp. Numeric columns are "
            "added as-is; text columns (e.g. manufacturing method) are automatically "
            "one-hot encoded."
        )
        _proc_extra_options = [c for c in all_cols if c not in _proc_already]
        proc_extra = st.multiselect(
            "Additional process feature columns",
            options=_proc_extra_options,
            default=_avail(DEF_PROC_EXTRA, _proc_extra_options),
            format_func=_clean_label,
            key=f"{key_prefix}_proc_extra",
            label_visibility="collapsed",
        )
        proc_cols = list(dict.fromkeys(proc_cols + proc_extra))

        # ── Custom user-defined engineered features ───────────────────────

        st.divider()
        st.markdown("##### 🧪 Custom Engineered Features")

        custom_feature_candidates = sorted(list(set(

            sel_nano +
            sel_an +
            sel_non +
            sel_zw +
            sel_poly +
            sel_citric +
            sel_acid +
            sel_anti +
            sel_brine +
            sel_oil +
            proc_cols +

            [
                "Nanoparticle (All Types)",
                "Sum Surfactant",
                "Anionic (All Types)",
                "Nonionic (All Types)",
                "Zwitterionic (All Types)",
                "Polymer (All Types)",
                "Acid (All Types)",
                "Citric (All Types)",
                "Antiscalant (All Types)",
            ]

        )))

        n_custom_ops = st.number_input(
            "Number of custom engineered features",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key=f"{key_prefix}_n_custom_ops",
        )

        custom_operations = []

        for i in range(n_custom_ops):

            st.markdown(f"**Custom Feature #{i+1}**")

            c1, c2, c3 = st.columns([3, 1, 3])

            with c1:
                feat_a = st.selectbox(
                    "Component 1",
                    options=custom_feature_candidates,
                    key=f"{key_prefix}_cust_a_{i}",
                )

            with c2:
                op = st.selectbox(
                    "Operation",
                    options=["+", "/"],
                    key=f"{key_prefix}_cust_op_{i}",
                )

            with c3:
                feat_b = st.selectbox(
                    "Component 2",
                    options=custom_feature_candidates,
                    key=f"{key_prefix}_cust_b_{i}",
                )

            if feat_a == feat_b:
                st.warning("Component 1 and 2 are identical.")

            custom_operations.append({
                "a": feat_a,
                "b": feat_b,
                "op": op,
            })
        st.markdown("##### 🔗Automatic Interaction Features")
        st.caption(
            "**Sum (A+B):** combined concentration — e.g. Nano Total + Surfactant Total.  "
            "**Ratio (A/B):** relative proportion — e.g. Nano Total / Surfactant Total."
        )
        ix_col1, ix_col2 = st.columns(2)
        ix_sum = ix_col1.checkbox(
            "➕ Sum interactions  (A + B)",
            value=st.session_state.get(f"{key_prefix}_ix_sum", True),
            key=f"{key_prefix}_ix_sum",
        )
        ix_ratio = ix_col2.checkbox(
            "➗ Ratio interactions  (A / B)",
            value=st.session_state.get(f"{key_prefix}_ix_ratio", False),
            key=f"{key_prefix}_ix_ratio",
        )
        include_ix = ix_sum or ix_ratio   # kept for backward-compat pass-through

    # ── Build sel dict ────────────────────────────────────────────────────
    # init_temp/method no longer have dedicated pickers — any such column is
    # instead picked up via the "Additional Process Features" catch-all and
    # handled generically in build_foam_features() (numeric vs. one-hot).
    proc_map = {
        "dilution":  proc_dil     if proc_dil     != NONE else None,
        "temp":      proc_temp    if proc_temp    != NONE else None,
        "oil_pct":   proc_oil_pct if proc_oil_pct != NONE else None,
        "init_temp": None,
        "method":    None,
    }
    sel = {
        "Nanoparticle":  sel_nano,
        "Anionic":       sel_an,
        "Nonionic":      sel_non,
        "Zwitterionic":  sel_zw,
        "Polymer":       sel_poly,
        "Citric":        sel_citric,
        "Acid/Chelant":  sel_acid,
        "Antiscalant":   sel_anti,
        "Brine":         sel_brine,
        "Oil":           sel_oil,
        "Process":       proc_cols,
        "_proc_map":     proc_map,
    }

    # Fill chemical NaN → 0 before building
    df_fe = df.copy()
    for g in CHEM_GROUPS:
        for col in sel.get(g, []):
            if col in df_fe.columns:
                df_fe[col] = pd.to_numeric(df_fe[col], errors="coerce").fillna(0)

    X_eng, fg = build_foam_features(
        df_fe,
        sel,
        include_ix,
        [],   # "Custom extra columns" picker removed — use Custom Engineered Features instead
        ix_sum=ix_sum,
        ix_ratio=ix_ratio,
        custom_operations=custom_operations,
    )
    # ── Overview ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("##### 📊 Engineered Feature Overview")
    n_total  = X_eng.shape[1]
    n_ix     = sum(1 for g in fg.values() if g == "Interaction")
    n_custom = sum(1 for g in fg.values() if g == "Custom")
    n_chem   = n_total - n_ix - n_custom

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total features",       n_total)
    m2.metric("Chemistry features",   n_chem)
    m3.metric("Interaction features", n_ix)
    m4.metric("Custom features",      n_custom)
    m5.metric("Rows",                 len(X_eng))

    group_counts = pd.Series(fg).value_counts()
    fig_gc = px.bar(
        x=group_counts.index, y=group_counts.values,
        color=group_counts.index, color_discrete_map=GRP_CLR,
        labels={"x": "Group", "y": "# Features"},
        title="Features per Group", template="plotly_white",
    )
    fig_gc.update_layout(height=260, showlegend=False,
                          margin=dict(t=40, b=20, l=10, r=10))
    st.plotly_chart(fig_gc, use_container_width=True)


    # Report any oil interactions that were skipped (parent feature > 10% NaN)

    expected_oil_ix = [
        f"Nanoparticle (All Types) + Oil Wax Fraction",
        f"Nanoparticle (All Types) + Oil Aromatics",
        f"Anionic (All Types) + Oil Wax Fraction",
    ]
    skipped = [n for n in expected_oil_ix
               if (ix_sum or ix_ratio) and n not in X_eng.columns
               and "Oil Wax Fraction" in X_eng.columns or "Oil Aromatics" in X_eng.columns]
    if skipped:
        st.info(
            f"ℹ️ {len(skipped)} oil interaction(s) skipped (>10% NaN): "
            + ", ".join(skipped)
        )

    # Merge engineered columns back into df
    df_out = df.copy().reset_index(drop=True)
    for col in X_eng.columns:
        df_out[col] = X_eng[col].values

    # ── Formulation ID + Analysis ──────────────────────────────────────────
    # A "formulation" = identical values across the chemical groups selected
    # above (Nanoparticle/Anionic/Nonionic/Zwitterionic/Polymer/Citric/Acid/
    # Antiscalant). Rows sharing a formulation usually differ only in a
    # process condition (dilution ratio, temperature, brine) — worth knowing
    # before trusting a correlation or model score built on this data.
    _chem_cols_for_formulation = list(dict.fromkeys(
        sel_nano + sel_an + sel_non + sel_zw + sel_poly + sel_citric + sel_acid + sel_anti
    ))
    if _chem_cols_for_formulation:
        _fid_df = pd.DataFrame({
            c: pd.to_numeric(df_fe[c], errors="coerce").fillna(0).round(6)
            for c in _chem_cols_for_formulation
        })
        formulation_id = _fid_df.astype(str).agg("|".join, axis=1).reset_index(drop=True)
    else:
        formulation_id = pd.Series(range(len(df_out))).astype(str)

    _fid_counts = formulation_id.value_counts()
    df_out["_formulation_id"]          = formulation_id.values
    df_out["_formulation_repeat_count"] = formulation_id.map(_fid_counts).values

    st.divider()
    st.markdown("##### 🧬 Formulation Analysis")

    if not _chem_cols_for_formulation:
        st.caption("Select at least one chemical group above to enable formulation grouping.")
    else:
        n_repeated = int((_fid_counts >= 2).sum())

        fa1, fa2, fa3 = st.columns(3)
        fa1.metric("Rows", f"{len(df_out):,}")
        fa2.metric("Unique formulations", f"{formulation_id.nunique():,}")
        fa3.metric("Formulations w/ repeats", f"{n_repeated:,}")

        if n_repeated == 0:
            st.caption(
                "No repeated formulations with the current group selection — "
                "every row is a distinct chemistry."
            )
        else:
            inspect_col = st.selectbox(
                "Column to inspect across repeats",
                options=[c for c in df_out.columns
                         if c not in ("_formulation_id", "_formulation_repeat_count")],
                key=f"{key_prefix}_fid_inspect_col",
                help="Pick a target or process column to see how much it varies "
                     "across rows that share the same formulation.",
            )
            _num_inspect = pd.to_numeric(df_out[inspect_col], errors="coerce")

            with st.expander(
                f"📋 Repeat Formulations ({n_repeated} formulations, ≥2 rows each)",
                expanded=False,
            ):
                _summary_rows = []
                for fid, cnt in _fid_counts[_fid_counts >= 2].items():
                    mask = formulation_id == fid
                    vals = _num_inspect[mask]
                    row = {
                        "Formulation": fid[:40] + ("…" if len(fid) > 40 else ""),
                        "Rows": int(cnt),
                    }
                    if vals.notna().sum() > 0:
                        row[f"{inspect_col} min"]  = round(float(vals.min()), 3)
                        row[f"{inspect_col} max"]  = round(float(vals.max()), 3)
                        row[f"{inspect_col} mean"] = round(float(vals.mean()), 3)
                        row[f"{inspect_col} std"]  = round(float(vals.std()), 3) if cnt > 1 else 0.0
                    _summary_rows.append(row)
                _summary_df = pd.DataFrame(_summary_rows).sort_values("Rows", ascending=False)
                st.dataframe(_summary_df, use_container_width=True, hide_index=True)

            with st.expander("🔍 Formulation Explorer — inspect one formulation's rows", expanded=False):
                _repeat_ids = _fid_counts[_fid_counts >= 2].index.tolist()
                picked_fid = st.selectbox(
                    "Pick a formulation (only ones with repeats shown)",
                    options=_repeat_ids,
                    format_func=lambda f: f[:60] + ("…" if len(f) > 60 else ""),
                    key=f"{key_prefix}_fid_explore",
                )
                _explore_cols = list(dict.fromkeys(proc_cols + [inspect_col]))
                _explore_cols = [c for c in _explore_cols if c in df_out.columns]
                _explore_df = df_out.loc[formulation_id == picked_fid, _explore_cols]
                if _explore_cols:
                    _explore_df = _explore_df.sort_values(_explore_cols[0])
                st.dataframe(_explore_df, use_container_width=True)

    excluded_Features_ix = [
        f"Anionic (All Types) + Sum Surfactant",
        f"Nonionic (All Types) + Sum Surfactant",
        f"Zwitterionic (All Types) + Sum Surfactant",
        f"Anionic (All Types) + Nonionic (All Types)",
        f"Anionic (All Types) + Zwitterionic (All Types)",
        f"Nonionic (All Types) + Anionic (All Types)",
        f"Nonionic (All Types) + Zwitterionic (All Types)",
        f"Zwitterionic (All Types) + Anionic (All Types)",
        f"Zwitterionic (All Types) + Nonionic (All Types)",
    ]
    X_eng = X_eng.drop(columns=excluded_Features_ix, errors="ignore")
    with st.expander("Preview engineered feature matrix (first 8 rows)"):
        st.dataframe(X_eng, use_container_width=True)

    return df_out, list(X_eng.columns), fg