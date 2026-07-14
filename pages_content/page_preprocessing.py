"""
pages_content/page_preprocessing.py
Sequential preprocessing pipeline — no tabs.

Sections (in order):
  0  Target Conversion   — keep regression OR bin to Low/High or Low/Mid/High
  1  Data Filtering      — brine/dilution/concentrate + dynamic custom rules
  2  Feature Engineering — foam chemical groups + auto interactions (foam_app style)
  3  Outlier Removal     — IQR / Z-score with live boxplot preview
  4  Scaling             — standard / minmax / robust (fit on full set; scaler stored for train page)
  5  Final Overview      — shape, class balance, feature matrix preview → Apply & Save

Data-isolation contract (unchanged from previous design):
  - EDA page reads only  data.raw  (never touched here).
  - Shared state keys are written ONLY when "Apply & Save" is clicked.
  - Working data is rebuilt from scratch every rerun from session state flags.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from itertools import product as iter_product

from state.session import init_state, get_value, set_state
from components.column_selector import render_column_selector
from core.data.preprocessor import (
    impute_missing, detect_outliers, remove_outliers,
    encode_categoricals, scale_features,
)
from core.viz.eda import draw_boxplots
from core.viz.style import fig_to_st
from config.settings import DEFAULT_IQR_FACTOR, DEFAULT_ZSCORE_THRESHOLD
from utils.data_filter import render_data_filters
from core.data.foam_feature_engineering import GRP_CLR

# ── Auto-ML–style group taxonomy (mirrors page_auto_ml.py) ───────────────────
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

# Pre-selected by default in "Additional Process Features" (Temperature /
# Dilution / Oil % have their own dedicated pickers already; this is for
# extras worth defaulting to). Covers both naming conventions seen across
# datasets (Oil workbook uses the _Normalized suffix, others don't).
DEF_PROC_EXTRA = [
    "concentrate manufacturing method (Ratio)_Normalized",
    "concentrate manufacturing method (Ratio)",
]

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
CHEM_GROUPS_AUTO = {"Nanoparticle", "Anionic", "Nonionic", "Zwitterionic",
                    "Polymer", "Citric", "Acid", "Antiscalant"}
# Condition groups: NaN = unknown → drop row
COND_GROUPS_AUTO = {"Brine", "Oil", "Process"}


def _normalise(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r'\s*\(%\)\s*$', '', s).strip()
    return s


def _avail_auto(candidates: list[str], cols: list[str]) -> list[str]:
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


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _build_group_id(data: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Hash of `cols`' values per row — rows with identical values share an ID.
    Used to build formulation/oil/brine identity IDs for Train's
    Leave-One-Group-Out CV. Falls back to a per-row unique ID when no
    columns are given/available (degenerates to a random split).
    """
    present = [c for c in cols if c in data.columns]
    if present:
        return (data[present].apply(_safe_num).fillna(0)
                .round(6).astype(str).agg("|".join, axis=1))
    return pd.Series(data.index, index=data.index).astype(str)


def _compute_sum_features(df: pd.DataFrame,
                           sel_groups: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for feat_name, group_key in SUM_FEATURES.items():
        cols = (sel_groups.get("Anionic", []) +
                sel_groups.get("Nonionic", []) +
                sel_groups.get("Zwitterionic", [])) \
               if group_key == "Surfactant" \
               else sel_groups.get(group_key, [])
        present = [c for c in cols if c in out.columns]
        if present:
            s = pd.Series(0.0, index=out.index)
            for c in present:
                s += _safe_num(out[c]).fillna(0)
            out[feat_name] = s
    return out




# ─────────────────────────────────────────────────────────────────────────────
#  Section renderers (pure UI — each returns updated `data` DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(icon: str, number: int, title: str, subtitle: str = ""):
    st.markdown(f"### {icon} Step {number} — {title}")
    if subtitle:
        st.caption(subtitle)


def _render_target_conversion(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Section 0: optionally bin a numeric target into 2 or 3 classes.
    Returns the (possibly transformed) target Series and writes
    task_mode to session state.
    """
    _section_header("🎯", 0, "Target Conversion",
                    "Keep as continuous regression OR bin to classification classes.")

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
                f"Target kept as **continuous** ({target_col}).\n\n"
                f"Range: {y_orig.min():.2f} – {y_orig.max():.2f}  |  "
                f"Median: {y_orig.median():.2f}"
            )
            st.session_state["tconv_task"] = "regression"

        elif mode == "Binary: Low / High":
            default_cut = float(y_orig.median())
            cut = c2.slider(
                "Cut-off  (Low < cut ≤ High)",
                min_value=float(y_orig.min()),
                max_value=float(y_orig.max()),
                value=default_cut,
                key="tconv_cut2",
            )
            y_out = (y_out > cut).map({False: "Low", True: "High"})
            counts = y_out.value_counts()
            c2.success(
                f"Low: **{counts.get('Low', 0)}** rows  |  "
                f"High: **{counts.get('High', 0)}** rows"
            )
            st.session_state["tconv_task"] = "classification"

        else:  # ternary
            lo_default = float(y_orig.quantile(0.33))
            hi_default = float(y_orig.quantile(0.67))
            rng = c2.slider(
                "Boundaries  (Low < lo ≤ Mid < hi ≤ High)",
                min_value=float(y_orig.min()),
                max_value=float(y_orig.max()),
                value=(lo_default, hi_default),
                key="tconv_cut3",
            )
            lo_cut, hi_cut = rng

            def _bin3(v):
                if pd.isna(v): return np.nan
                if v <= lo_cut:  return "Low"
                if v <= hi_cut:  return "Mid"
                return "High"

            y_out = y_out.apply(_bin3)
            counts = y_out.value_counts()
            c2.success(
                f"Low: **{counts.get('Low', 0)}**  |  "
                f"Mid: **{counts.get('Mid', 0)}**  |  "
                f"High: **{counts.get('High', 0)}**"
            )
            st.session_state["tconv_task"] = "classification"

        # Mini histogram of original target
        fig = px.histogram(
            x=y_orig, nbins=40, title="Original Target Distribution",
            labels={"x": target_col}, template="plotly_white",
            color_discrete_sequence=["#1565C0"],
        )
        fig.update_layout(height=260, margin=dict(t=35, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    return y_out




def _render_outlier_removal(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Section 3: outlier detection and removal with live preview."""
    _section_header("⚠️", 3, "Outlier Removal",
                    "Detect and remove outliers from numeric feature columns.")

    num_cols = [c for c in feature_cols if c in data.columns
                and pd.api.types.is_numeric_dtype(data[c])]

    with st.container(border=True):
        if not num_cols:
            st.info("No numeric feature columns — skipping.")
            return data

        method = st.selectbox("Detection method",
                               ["None", "IQR", "Z-score"],
                               key="out_method")

        if method == "None":
            if st.session_state.get("prep_confirmed_outlier"):
                st.info(f"✔ Outlier removal active — "
                        f"{st.session_state.get('prep_outlier_params', {}).get('method')}")
                if st.button("↩ Reset outlier step", key="out_reset"):
                    st.session_state.prep_confirmed_outlier = False
                    st.session_state.pop("prep_outlier_params", None)
                    st.rerun()
            else:
                st.success("No outlier removal selected.")
            return data

        oc1, oc2 = st.columns(2)
        iqr_f = oc1.slider("IQR factor",    1.0, 3.0, DEFAULT_IQR_FACTOR,      0.1, key="out_iqr")
        z_t   = oc2.slider("Z-score threshold", 2.0, 5.0, DEFAULT_ZSCORE_THRESHOLD, 0.1, key="out_z")

        summary = detect_outliers(data, num_cols, method=method,
                                   iqr_factor=iqr_f, z_thresh=z_t)
        has_outliers = summary["Outliers"].sum() > 0

        with st.expander("Outlier summary table", expanded=has_outliers):
            st.dataframe(summary.style.background_gradient(subset=["Outliers"], cmap="Reds"),
                         use_container_width=True)

        # Boxplots for top-5 most affected features
        top5 = summary.sort_values("Outliers", ascending=False).head(5)["Feature"].tolist()
        if top5:
            with st.expander("Boxplots (top 5 most affected features)", expanded=False):
                fig_bp = go.Figure()
                for col in top5:
                    fig_bp.add_trace(go.Box(y=data[col].dropna(), name=col, boxpoints="outliers"))
                fig_bp.update_layout(title="Distribution before removal",
                                     height=350, template="plotly_white")
                st.plotly_chart(fig_bp, use_container_width=True)

        if has_outliers:
            total_out = int(summary["Outliers"].sum())
            st.warning(f"{total_out} potential outlier rows detected across {(summary['Outliers']>0).sum()} features.")

        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("🧹 Remove Outliers & Confirm", key="out_apply", type="primary"):
            cleaned, n_removed = remove_outliers(
                data, num_cols, method=method, iqr_factor=iqr_f, z_thresh=z_t,
            )
            st.session_state.prep_outlier_params    = {"method": method, "iqr_factor": iqr_f, "z_thresh": z_t}
            st.session_state.prep_confirmed_outlier = True
            st.success(f"✅ Removed **{n_removed}** outlier rows. Remaining: **{len(cleaned):,}**")
            st.rerun()

        if st.session_state.get("prep_confirmed_outlier"):
            p = st.session_state.get("prep_outlier_params", {})
            st.info(f"✔ Outlier removal active — method: **{p.get('method')}**, "
                    f"IQR: {p.get('iqr_factor')}, Z: {p.get('z_thresh')}")
            if col_btn2.button("↩ Reset outlier step", key="out_reset2"):
                st.session_state.prep_confirmed_outlier = False
                st.session_state.pop("prep_outlier_params", None)
                st.rerun()

    # Apply confirmed outlier removal
    if st.session_state.get("prep_confirmed_outlier") and "prep_outlier_params" in st.session_state:
        p = st.session_state.prep_outlier_params
        data, _ = remove_outliers(data, num_cols,
                                   method=p["method"],
                                   iqr_factor=p["iqr_factor"],
                                   z_thresh=p["z_thresh"])
    return data


def _render_scaling(feature_cols: list[str]) -> str:
    """Section 4: scaling strategy picker. Returns strategy string."""
    _section_header("📏", 4, "Feature Scaling",
                    "Scaler is stored and applied properly inside the Train page (fit on train, transform test). "
                    "Original feature values are preserved for plots.")

    with st.container(border=True):
        sc1, sc2 = st.columns([2, 3])
        strategy = sc1.selectbox(
            "Scaling strategy",
            ["none", "standard", "robust", "minmax"],
            index=1,
            key="prep_scaling",
            help=(
                "standard = zero mean, unit variance\n"
                "robust   = median / IQR (outlier-resistant)\n"
                "minmax   = scale to [0, 1]"
            ),
        )
        descriptions = {
            "none":     "No scaling applied.",
            "standard": "StandardScaler: subtract mean, divide std. Best for linear models.",
            "robust":   "RobustScaler: subtract median, divide IQR. Best when outliers remain.",
            "minmax":   "MinMaxScaler: scale to [0, 1]. Sensitive to outliers.",
        }
        sc2.info(descriptions[strategy])
        st.caption("⚠️ Scaling is done internally at train time. All plots and SHAP show **original scale** values.")
    return strategy


def _render_final_overview(
    X: pd.DataFrame,
    y: pd.Series,
    fg: dict,
    task: str,
) -> None:
    """Section 5: final summary before Apply."""
    _section_header("📋", 5, "Final Dataset Overview",
                    "Review the prepared dataset before saving to session.")

    with st.container(border=True):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rows", f"{len(X):,}")
        r2.metric("Features", X.shape[1])
        r3.metric("Task type", task)
        r4.metric("Missing values", int(X.isnull().sum().sum()))

        # Target distribution
        ct1, ct2 = st.columns(2)
        with ct1:
            if task == "regression":
                fig_y = px.histogram(x=y, nbins=40,
                                     title="Target Distribution",
                                     template="plotly_white",
                                     color_discrete_sequence=["#1565C0"])
                fig_y.update_layout(height=260, margin=dict(t=35,b=10,l=10,r=10),
                                    showlegend=False,
                                    xaxis_title=y.name or "Target")
                st.plotly_chart(fig_y, use_container_width=True)
            else:
                vc = y.value_counts()
                fig_y = px.bar(x=vc.index, y=vc.values,
                               color=vc.index, title="Class Distribution",
                               template="plotly_white",
                               labels={"x": "Class", "y": "Count"})
                fig_y.update_layout(height=260, showlegend=False,
                                    margin=dict(t=35,b=10,l=10,r=10))
                st.plotly_chart(fig_y, use_container_width=True)

        with ct2:
            # Group importance bar
            if fg:
                gc = pd.Series(fg).value_counts()
                fig_gc = px.bar(x=gc.index, y=gc.values,
                                color=gc.index, color_discrete_map=GRP_CLR,
                                title="Features per Group",
                                template="plotly_white",
                                labels={"x": "Group", "y": "# Features"})
                fig_gc.update_layout(height=260, showlegend=False,      
                                     margin=dict(t=35,b=10,l=10,r=10))
                st.plotly_chart(fig_gc, use_container_width=True)

        with st.expander("Feature matrix preview (first 20 rows)"):
            st.dataframe(X, use_container_width=True)

        # NaN report
        nan_cols = X.isnull().mean()
        nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
        if len(nan_cols):
            with st.expander(f"⚠️ {len(nan_cols)} features have remaining NaN"):
                st.dataframe(nan_cols.rename("NaN fraction").to_frame()
                              .style.background_gradient(cmap="Oranges"),
                             use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    init_state()

    st.title("⚙️ Preprocessing & Feature Engineering")
    st.caption(
        "Work through each step sequentially. "
        "Changes only take effect when you click **✅ Apply & Save** at the bottom."
    )
    st.divider()

    # ── Guard ─────────────────────────────────────────────────────────────
    df_raw = get_value("data.raw")
    if df_raw is None:
        st.warning("⚠️ Load data first — go to **📂 Data**.")
        st.stop()

    all_cols = list(df_raw.columns)

    # ══════════════════════════════════════════════════════════════════════
    # Feature & Target Selection
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 📌 Feature & Target Selection")
    with st.expander("Select features and target column", expanded=True):
        selected_features, selected_target = render_column_selector(
            df_raw,
            default_features=get_value("data.feature_names"),
            default_target=get_value("data.target_name"),
            key_prefix="prep",
        )

    if not selected_features or selected_target is None:
        st.info("Select at least one feature and a target column above, then continue.")
        st.stop()

    non_target = [c for c in all_cols if c != selected_target]

    # Working data reset when selection changes
    sel_key = tuple(sorted(selected_features)) + (selected_target,)
    if sel_key != st.session_state.get("prep_selection_key") or "working_data_base" not in st.session_state:
        st.session_state.working_data_base      = df_raw[selected_features + [selected_target]].copy()
        st.session_state.prep_selection_key     = sel_key
        st.session_state.prep_confirmed_outlier = False
        st.session_state.pop("prep_outlier_params", None)

    data = st.session_state.working_data_base.copy()

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Target Conversion
    # ══════════════════════════════════════════════════════════════════════
    y_converted = _render_target_conversion(data, selected_target)
    task_mode   = st.session_state.get("tconv_task", "regression")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Data Filtering
    # ══════════════════════════════════════════════════════════════════════
    _section_header("🔍", 2, "Data Filtering",
                    "Filter rows sent to the model. Filters are non-destructive — adjust any time.")
    data = render_data_filters(data, key_prefix="prep")

    # Align y to filtered rows
    y_converted = y_converted.loc[data.index].reset_index(drop=True)
    data        = data.reset_index(drop=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — Chemical Component Groups  (mirrors auto_ml step 4)
    # ══════════════════════════════════════════════════════════════════════
    _section_header("🧴", 3, "Chemical Component Groups",
                    "Default columns pre-selected from uploaded file. Add or remove freely.")

    sel_groups: dict[str, list[str]] = {}
    grp_items = [g for g in GROUPS if g not in ("Oil", "Brine", "Process")]

    with st.container():
        cols_ui = st.columns(2)
        for gi, grp in enumerate(grp_items):
            with cols_ui[gi % 2]:
                clr = GRP_CLR.get(grp, "#607D8B")
                st.markdown(
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
                    f'font-size:0.75rem;font-weight:600;margin:2px;background:{clr}22;color:{clr};'
                    f'border:1px solid {clr}66">{grp}</span>',
                    unsafe_allow_html=True,
                )
                sel_groups[grp] = st.multiselect(
                    f"Columns for {grp}",
                    options=non_target,
                    default=_avail_auto(GROUPS[grp], non_target),
                    key=f"prep_grp_{grp}",
                    label_visibility="collapsed",
                )

    st.markdown("##### 🌊 Brine & 🛢️ Oil compositions")
    bp_cols = st.columns(2)
    for ci, grp in enumerate(["Oil", "Brine"]):
        with bp_cols[ci]:
            clr = GRP_CLR.get(grp, "#607D8B")
            st.markdown(
                f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
                f'font-size:0.75rem;font-weight:600;margin:2px;background:{clr}22;color:{clr};'
                f'border:1px solid {clr}66">{grp}</span>',
                unsafe_allow_html=True,
            )
            sel_groups[grp] = st.multiselect(
                f"Columns for {grp}",
                options=non_target,
                default=_avail_auto(GROUPS[grp], non_target),
                key=f"prep_grp_{grp}",
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
                          in ("temperature", "temperature corrected", "temp")), NONE)
        proc_temp = st.selectbox("🌡️ Temperature", none_opt,
                                 index=none_opt.index(_def_temp) if _def_temp in none_opt else 0,
                                 key="prep_proc_temp")
    with pc2:
        _def_dil = next((c for c in non_target
                         if re.sub(r"\s+", " ", _normalise(c))
                         in ("dilution ratio", "dilution ratio corrected", "dilution")), NONE)
        proc_dil = st.selectbox("💧 Dilution Ratio", none_opt,
                                index=none_opt.index(_def_dil) if _def_dil in none_opt else 0,
                                key="prep_proc_dil")
    with pc3:
        _def_oilpct = next((c for c in non_target
                            if re.sub(r"\s+", " ", _normalise(c))
                            in ("oil", "oil percent", "oil pct")), NONE)
        proc_oil_pct = st.selectbox("🛢️ Oil Percent (%)", none_opt,
                                    index=none_opt.index(_def_oilpct) if _def_oilpct in none_opt else 0,
                                    key="prep_proc_oil_pct")

    sel_groups["Process"] = [c for c in [proc_temp, proc_dil, proc_oil_pct] if c != NONE]

    # ── Additional Process Features (free multiselect, empty by default) ──────
    _proc_already = set(sel_groups["Process"])

    st.markdown("**➕ Additional Process Features**")
    st.caption(
        "Columns that don't fit Temperature / Dilution / Oil — e.g. concentrate ratio, "
        "pH, manufacturing method. Treated as condition columns (NaN → row dropped)."
    )
    _proc_extra_options = [c for c in non_target if c not in _proc_already]
    proc_extra = st.multiselect(
        "Additional process feature columns",
        options=_proc_extra_options,
        default=_avail_auto(DEF_PROC_EXTRA, _proc_extra_options),
        key="prep_proc_extra",
        label_visibility="collapsed",
    )
    sel_groups["Process"] = list(dict.fromkeys(sel_groups["Process"] + proc_extra))

    # Warn about % columns not assigned
    _all_selected = set(c for cols in sel_groups.values() for c in cols)
    _pct_unassigned = [c for c in non_target if "%" in str(c) and c not in _all_selected]
    if _pct_unassigned:
        st.warning(
            f"⚠️ **{len(_pct_unassigned)} column(s) containing '%' not assigned to any group** "
            "and will be excluded:\n\n" + "\n".join(f"- `{c}`" for c in _pct_unassigned)
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — Computed Features  (mirrors auto_ml step 5)
    # ══════════════════════════════════════════════════════════════════════
    _section_header("⚗️", 4, "Computed Features",
                    "Auto-computed group sums + correlation-winner ratio/sum pairs + custom features.")

    df_work       = _compute_sum_features(data, sel_groups)
    computed_sums = [f for f in SUM_FEATURES if f in df_work.columns]

    if computed_sums:
        pills = " ".join(
            f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
            f'font-size:0.75rem;font-weight:600;margin:2px;background:#e3f2fd;color:#1565C0;'
            f'border:1px solid #90caf9">{s}</span>'
            for s in computed_sums
        )
        st.markdown("**Auto-computed sums:**")
        st.markdown(pills, unsafe_allow_html=True)

    # Custom engineered features
    user_ratios: list[tuple[str, str, str]] = []
    avail_for_ratio = list(dict.fromkeys(non_target + computed_sums))

    with st.expander("🧪 Custom Engineered Features", expanded=False):
        st.caption("Create new features by combining two columns with + or /")
        n_custom = st.number_input("Number of custom engineered features",
                                   0, 20, 0, 1, key="prep_n_custom")
        for ri in range(int(n_custom)):
            st.markdown(f"**Custom Feature #{ri+1}**")
            cc1, cc2, cc3 = st.columns([5, 2, 5])
            c1_sel = cc1.selectbox("Component 1", avail_for_ratio, key=f"prep_cf_c1_{ri}")
            op_sel = cc2.selectbox("Operation",   ["+", "/"],       key=f"prep_cf_op_{ri}")
            c2_sel = cc3.selectbox("Component 2", avail_for_ratio, key=f"prep_cf_c2_{ri}")
            user_ratios.append((c1_sel, op_sel, c2_sel))

    for _c1, _op, _c2 in user_ratios:
        _fname_fe = f"{_c1} {_op} {_c2}"
        if _c1 in df_work.columns and _c2 in df_work.columns and _fname_fe not in df_work.columns:
            _c1_num = _safe_num(df_work[_c1])
            _c2_num = _safe_num(df_work[_c2])
            df_work[_fname_fe] = (_c1_num + _c2_num if _op == "+"
                                  else _c1_num / _c2_num.replace(0, float("nan")))
    ratio_cols = [f"{c1} {op} {c2}" for c1, op, c2 in user_ratios
                  if f"{c1} {op} {c2}" in df_work.columns]

    # Must-have ratio/sum pairs — winner selected by |correlation| with target
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

    _y_for_corr   = _safe_num(df_work[selected_target])
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
            _winner, _loser  = _sum_name,   _ratio_name
            _r_win,  _r_lose = _r_sum,      _r_ratio
        else:
            _winner, _loser  = _ratio_name, _sum_name
            _r_win,  _r_lose = _r_ratio,    _r_sum
        if _winner not in EXCLUDED_IX:
            _best_cols.append(_winner)
        _corr_report.append({
            "Pair":        f"{_num}  ×  {_den}",
            "Winner":      _winner,
            "Winner r":    round(_r_win,       4),
            "Winner |r|":  round(abs(_r_win),  4),
            "Loser":       _loser,
            "Loser r":     round(_r_lose,      4),
            "Loser |r|":   round(abs(_r_lose), 4),
        })

    _user_feat_cols = [f"{c1} {op} {c2}" for c1, op, c2 in user_ratios
                       if f"{c1} {op} {c2}" in df_work.columns]
    all_ratio_cols = list(dict.fromkeys(
        [c for c in _best_cols if c not in EXCLUDED_IX] + _user_feat_cols
    ))

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
                f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
                f'font-size:0.75rem;font-weight:600;margin:2px;background:#fff8e1;color:#795548;'
                f'border:1px solid #bcaaa4">{c}</span>'
                for c in all_ratio_cols
            )
            st.markdown("**Selected features (winners + custom):**")
            st.markdown(_ratio_pills, unsafe_allow_html=True)

    # Build final feature_cols and fg for downstream steps
    all_feature_cols = list(dict.fromkeys(
        c for g, cols in sel_groups.items() for c in cols
    ))
    feature_cols = list(dict.fromkeys(
        all_feature_cols + computed_sums + all_ratio_cols
    ))
    feature_cols = [c for c in feature_cols if c in df_work.columns and c != selected_target]
    fg = {}  # group mapping for overview chart
    for grp, cols in sel_groups.items():
        for col in cols:
            fg[col] = grp
    for f in computed_sums:
        fg[f] = "Sum"
    for f in all_ratio_cols:
        fg[f] = "Interaction"

    # Use df_work as the working data from here on
    data = df_work.copy()

    # Re-align y_converted to data index (data may have been filtered)
    y_converted = y_converted.loc[data.index].reset_index(drop=True)
    data        = data.reset_index(drop=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 — NaN Handling  (mirrors auto_ml step 6)
    # ══════════════════════════════════════════════════════════════════════
    _section_header("🧩", 5, "NaN Handling",
                    "Choose how missing values in chemical groups are treated before modelling.")

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
            key="prep_chem_nan_strategy",
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

        # Impact preview
        all_chem_cols_sel = [c for g in CHEM_GROUPS_AUTO for c in sel_groups.get(g, [])
                             if c in data.columns]
        if all_chem_cols_sel:
            n_rows_with_nan = data[all_chem_cols_sel].isnull().any(axis=1).sum()
            n_total = len(data)
            if chem_nan_strategy == "fill_zero":
                st.caption(
                    f"**{n_rows_with_nan:,}** rows have at least one chemical NaN "
                    f"({100*n_rows_with_nan/max(n_total,1):.1f}%) → will be filled with 0.  "
                    f"All **{n_total:,}** rows kept."
                )
            else:
                st.caption(
                    f"**{n_rows_with_nan:,}** rows have at least one chemical NaN "
                    f"({100*n_rows_with_nan/max(n_total,1):.1f}%) → will be dropped.  "
                    f"**{n_total - n_rows_with_nan:,}** rows remain."
                )
        else:
            st.caption("No chemical group columns selected yet.")

    # Apply NaN strategy immediately so outlier removal sees clean data
    chem_cols_present = [c for g in CHEM_GROUPS_AUTO for c in sel_groups.get(g, [])
                         if c in data.columns]
    cond_cols_present = [c for g in COND_GROUPS_AUTO for c in sel_groups.get(g, [])
                         if c in data.columns]

    if chem_nan_strategy == "fill_zero":
        for col in chem_cols_present:
            data[col] = _safe_num(data[col]).fillna(0)
        # Also fill derived sum/ratio features that come from chem columns
        for col in feature_cols:
            if col in data.columns and (" + " in col or " / " in col):
                data[col] = _safe_num(data[col]).fillna(0)
        for col in computed_sums:
            if col in data.columns:
                data[col] = _safe_num(data[col]).fillna(0)
    else:
        if chem_cols_present:
            data = data.dropna(subset=chem_cols_present)

    if cond_cols_present:
        data = data.dropna(subset=cond_cols_present)

    # ── Group IDs (for group-aware validation in Train — LOGO) ──────────────
    # Hash rows by chemistry (formulation), oil identity, or brine identity —
    # lets Train's Leave-One-Group-Out CV test generalization along whichever
    # axis matters: new dilution/temperature of a known formulation, or a
    # genuinely new oil / brine never seen in training. All three are carried
    # through the same re-indexing as y_converted so they always line up
    # with `data` / eventually data.X.
    formulation_id    = _build_group_id(data, chem_cols_present)
    oil_identity_id   = _build_group_id(data, sel_groups.get("Oil", []))
    brine_identity_id = _build_group_id(data, sel_groups.get("Brine", []))

    # Re-align y after NaN drops
    y_converted        = y_converted.loc[data.index].reset_index(drop=True)
    formulation_id     = formulation_id.loc[data.index].reset_index(drop=True)
    oil_identity_id    = oil_identity_id.loc[data.index].reset_index(drop=True)
    brine_identity_id  = brine_identity_id.loc[data.index].reset_index(drop=True)
    data               = data.reset_index(drop=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6 — Outlier Removal
    # ══════════════════════════════════════════════════════════════════════
    data = _render_outlier_removal(data, feature_cols)

    # Re-align y after outlier removal
    y_converted        = y_converted.loc[data.index].reset_index(drop=True)
    formulation_id     = formulation_id.loc[data.index].reset_index(drop=True)
    oil_identity_id    = oil_identity_id.loc[data.index].reset_index(drop=True)
    brine_identity_id  = brine_identity_id.loc[data.index].reset_index(drop=True)
    data               = data.reset_index(drop=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 7 — Scaling
    # ══════════════════════════════════════════════════════════════════════
    scaling_strategy = _render_scaling(feature_cols)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8 — Final Overview
    # ══════════════════════════════════════════════════════════════════════
    feat_cols_final = list(dict.fromkeys(
        c for c in feature_cols if c in data.columns and c != selected_target
    ))

    X_preview = data[feat_cols_final].copy()

    #_render_final_overview(X_preview, y_converted, fg, task_mode)
    #st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 9 — Unscaled Data Preview
    # ══════════════════════════════════════════════════════════════════════
    _section_header("🔬", 5, "Unscaled Data Preview",
                    "Inspect the final dataset in original units before any scaling. "
                    "Use this to verify filters, engineered features and outlier removal are correct.")

    with st.container(border=True):

        # ── Pipeline row-count summary ────────────────────────────────────
        raw_n     = len(df_raw)
        curr_n    = len(X_preview)
        removed_n = raw_n - curr_n

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw rows",     f"{raw_n:,}")
        m2.metric("Current rows", f"{curr_n:,}",
                  delta=f"-{removed_n:,} removed",
                  delta_color="off" if removed_n == 0 else "inverse")
        m3.metric("Features",     X_preview.shape[1])
        m4.metric("Target",       selected_target)

        st.divider()

        # ── Column picker ─────────────────────────────────────────────────
        all_prev_cols = X_preview.columns.tolist()
        sel_prev_cols = st.multiselect(
            "Columns to display",
            options=all_prev_cols,
            default=all_prev_cols[:min(15, len(all_prev_cols))],
            key="preview_cols",
            help="Select which columns to inspect. Defaults to first 15.",
        )
        show_tgt = st.checkbox("Include target column", value=True, key="preview_show_target")

        display_df = X_preview[sel_prev_cols].copy()
        if show_tgt:
            display_df.insert(0, selected_target, y_converted.values)

        # ── Quick value filter ────────────────────────────────────────────
        qf1, qf2 = st.columns([2, 3])
        search_feature = qf1.selectbox(
            "Quick-filter by column",
            options=["(none)"] + sel_prev_cols,
            key="preview_search_col",
        )
        if search_feature != "(none)":
            col_vals = X_preview[search_feature]
            if pd.api.types.is_numeric_dtype(col_vals):
                lo, hi = float(col_vals.min()), float(col_vals.max())
                if lo < hi:
                    rng = qf2.slider(
                        f"{search_feature} range",
                        lo, hi, (lo, hi), key="preview_num_range",
                    )
                    display_df = display_df[
                        (col_vals >= rng[0]) & (col_vals <= rng[1])
                    ]
            else:
                opts  = col_vals.dropna().unique().tolist()
                picks = qf2.multiselect(
                    f"{search_feature} values",
                    options=opts, default=opts, key="preview_cat_vals",
                )
                display_df = display_df[col_vals.isin(picks)]

        st.caption(
            f"Showing **{len(display_df):,}** rows × **{len(display_df.columns)}** columns "
            f"— original scale, no scaling applied."
        )
        st.dataframe(display_df, use_container_width=True, height=420)

        # ── Descriptive stats ─────────────────────────────────────────────
        # with st.expander("📈 Descriptive statistics (unscaled)"):
        #     num_prev = display_df.select_dtypes(include="number").columns.tolist()
        #     if num_prev:
        #         desc = display_df[num_prev].describe().T
        #         desc["missing %"] = (
        #             display_df[num_prev].isnull().mean() * 100
        #         ).round(2)
        #         st.dataframe(
        #             desc.style
        #                 .format("{:.4f}", na_rep="—")
        #                 .background_gradient(subset=["missing %"], cmap="Oranges"),
        #             use_container_width=True,
        #         )
        #     else:
        #         st.info("No numeric columns in current selection.")

        # ── Distribution spot-check ───────────────────────────────────────
        # with st.expander("📊 Distribution spot-check"):
        #     num_opts = [c for c in sel_prev_cols
        #                 if pd.api.types.is_numeric_dtype(X_preview[c])]
        #     cat_opts = ["(none)"] + [
        #         c for c in display_df.columns
        #         if not pd.api.types.is_numeric_dtype(display_df[c])
        #     ]
        #     sp1, sp2 = st.columns([2, 3])
        #     if num_opts:
        #         spot_col   = sp1.selectbox("Feature to plot", num_opts, key="preview_spot_col")
        #         color_col  = sp2.selectbox("Colour by",       cat_opts, key="preview_spot_color")
        #         import plotly.express as _px
        #         if color_col == "(none)":
        #             fig_sp = _px.histogram(
        #                 display_df, x=spot_col, nbins=40,
        #                 title=f"Distribution of  {spot_col}  (original scale)",
        #                 template="plotly_white",
        #                 color_discrete_sequence=["#1565C0"],
        #             )
        #         else:
        #             fig_sp = _px.histogram(
        #                 display_df, x=spot_col, color=color_col,
        #                 nbins=40, barmode="overlay", opacity=0.7,
        #                 title=f"{spot_col}  by  {color_col}",
        #                 template="plotly_white",
        #             )
        #         fig_sp.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        #         st.plotly_chart(fig_sp, use_container_width=True)
        #     else:
        #         st.info("No numeric columns selected.")

        # ── Download ──────────────────────────────────────────────────────
        st.download_button(
            "⬇️ Download as CSV",
            data=X_preview.to_csv(index=False).encode("utf-8-sig"),
            file_name="unscaled_preview.csv",
            mime="text/csv",
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # APPLY & SAVE
    # ══════════════════════════════════════════════════════════════════════
    if st.button("✅ Apply Preprocessing & Save", type="primary", use_container_width=True):

        pipeline_data = data[feat_cols_final + [selected_target]].copy()
        pipeline_data[selected_target]      = y_converted.values
        pipeline_data["_formulation_id"]    = formulation_id.values
        pipeline_data["_oil_identity_id"]   = oil_identity_id.values
        pipeline_data["_brine_identity_id"] = brine_identity_id.values

        # Safety net: drop any remaining NaN rows
        n_before_drop = len(pipeline_data)
        pipeline_data = pipeline_data.dropna()
        if len(pipeline_data) < n_before_drop:
            st.warning(f"Auto-dropped {n_before_drop - len(pipeline_data)} rows with remaining NaN.")

        X = pipeline_data[feat_cols_final].copy()
        y = pipeline_data[selected_target].copy()
        formulation_id_final    = pipeline_data["_formulation_id"].copy()
        oil_identity_id_final   = pipeline_data["_oil_identity_id"].copy()
        brine_identity_id_final = pipeline_data["_brine_identity_id"].copy()

        # Encode categoricals if any slipped through
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        encoder_map = {}
        if cat_cols:
            X, encoder_map = encode_categoricals(X, strategy="label")

        # Build and store scaler (fit on full X here; train page applies properly)
        scaler = None
        if scaling_strategy != "none":
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            scaler = {"standard": StandardScaler(),
                      "minmax":   MinMaxScaler(),
                      "robust":   RobustScaler()}[scaling_strategy]
            num_sc = X.select_dtypes(include="number").columns.tolist()
            # Store original X (unscaled) for plots/SHAP
            X_original = X.copy()
            X[num_sc] = scaler.fit_transform(X[num_sc])
        else:
            X_original = X.copy()

        # Infer task type from converted target
        task_type = (
            "regression"
            if task_mode == "regression" and pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
            else "classification" if task_mode == "classification"
            else "regression"
        )

        # ── Write to shared session state ─────────────────────────────
        set_state("data.cleaned",                   pipeline_data)
        set_state("data.X",                         X)
        set_state("data.X_original",                X_original)   # unscaled for plots
        set_state("data.y",                         y)
        set_state("data.formulation_id",             formulation_id_final)
        set_state("data.oil_identity_id",            oil_identity_id_final)
        set_state("data.brine_identity_id",          brine_identity_id_final)
        set_state("data.feature_names",             feat_cols_final)
        set_state("data.target_name",               selected_target)
        set_state("data.processed_feature_names",   list(X.columns))
        set_state("preprocessing.missing_strategy", chem_nan_strategy)
        set_state("preprocessing.outlier_method",
                  st.session_state.get("prep_outlier_params", {}).get("method", "none")
                  if st.session_state.get("prep_confirmed_outlier") else "none")
        set_state("preprocessing.scaler",           scaler)
        set_state("preprocessing.scaler_strategy",  scaling_strategy)
        set_state("preprocessing.encoder",          encoder_map)
        set_state("preprocessing.rows_before",      df_raw.shape[0])
        set_state("preprocessing.rows_after",       len(pipeline_data))
        set_state("preprocessing.applied",          True)
        set_state("model.task_type",                task_type)
        set_state("preprocessing.feature_engineering_enabled", True)
        set_state("preprocessing.feature_engineering_meta",    {"feature_cols": feat_cols_final, "feat_to_group": fg})
        set_state("preprocessing.feat_to_group",               fg)

        st.success("✅ Preprocessing complete and saved to session!")
        st.info(
            f"Rows: {df_raw.shape[0]:,} → {len(pipeline_data):,}  |  "
            f"Features: {X.shape[1]}  |  "
            f"Task: **{task_type}**  |  "
            f"Scaling: **{scaling_strategy}**"
        )

        with st.expander("📊 Final processed data preview (first 50 rows)", expanded=True):
            st.dataframe(X, use_container_width=True)

        st.info("👉 Proceed to **🤖 Train** to choose your validation strategy and run the model.")