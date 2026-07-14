"""
page_ai_report_validated.py
============================
"Automatic AI Report (Validated)" — same upload / group-selection /
computed-feature / SHAP workflow as Page_auto_ml.py, PLUS the group-aware
validation workflow recommended for this foam dataset:

  • Many rows share an identical chemical formulation and differ only in
    process conditions (dilution ratio, temperature, brine) — a plain
    random split can leak the same formulation into both train and test,
    inflating the reported score.
  • This page builds a "formulation ID" from the selected chemical columns
    and validates with Group K-Fold (each formulation stays in one fold),
    reporting honest out-of-fold metrics BEFORE the final model (used for
    SHAP interpretation) is fit on 100% of the data.

Sections 1-8 (Upload → SHAP Plot Configuration) are unchanged in behaviour
from Page_auto_ml.py — the constants, preprocessing, training and SHAP
plotting logic are imported from there rather than re-implemented, so the
two pages never drift apart. Only the widget keys are namespaced (KEY_PREFIX)
so this page's session state never collides with Page_auto_ml.py's.

New sections added on top of the mirrored workflow:
  9  · Validation Strategy (Group-Aware)
  11 · Group-Aware Validation  (honest held-out metrics)

Run:
    streamlit run app.py   (Extra Tool → Automatic AI Report (Validated))
"""
from __future__ import annotations

import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from utils.data_filter import render_data_filters
from pages_content.page_auto_ml import (
    CHEM_GROUPS,
    COND_GROUPS,
    FIG_DPI,
    GROUPS,
    GRP_CLR,
    SUM_FEATURES,
    TARGET_CANDIDATES,
    _avail,
    _generate_model_performance,
    _normalise,
    _render_target_conversion,
    _render_to_bytes,
    _safe_num,
    _shap_dependence,
    compute_sum_features,
    figures_to_zip,
    generate_shap_plots,
    preprocess,
    save_figures_to_disk,
    train_model,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Namespaces every widget key on this page so it never shares session state
# with Page_auto_ml.py (both pages otherwise use identical widget labels).
KEY_PREFIX = "vml"

# ── Validation-strategy options (new workflow step) ─────────────────────────
VALIDATION_METHODS: list[str] = [
    "Group K-Fold (recommended — keeps each group in one fold)",
    "Random K-Fold (ignores groups — may leak)",
]

# What defines a "group" for Group K-Fold — the axis you want to test
# generalization along. Formulation = same chemistry, different dilution/
# temperature/brine (prevents formulation leakage). Oil/Brine identity =
# tests whether the model generalizes to an oil/brine it has never seen,
# not just a new formulation of a familiar one.
GROUP_BY_OPTIONS: list[str] = [
    "Chemical formulation (prevents formulation leakage)",
    "Oil identity (tests generalization to a new oil)",
    "Brine identity (tests generalization to a new brine)",
]

DEFAULT_N_SPLITS = 5
MIN_N_SPLITS = 2
MAX_N_SPLITS = 10

LOG_TRANSFORM_OPTIONS: list[str] = [
    "log1p (recommended for skewed targets)",
    "None (raw target)",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.title("🛡️ Automatic AI Report (Validated)")
    st.caption(
        "Same workflow as **Automatic AI Report**, plus group-aware validation: "
        "predictions are honestly scored on held-out *formulations*, not just "
        "held-out rows, before the final model is fit for SHAP interpretation."
    )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 1 — Upload
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 1 · Upload Data")
    uploaded = st.file_uploader(
        "CSV (UTF-8) or Excel (.xlsx / .xls)",
        type=["csv", "xlsx", "xls"],
        key=f"{KEY_PREFIX}_uploader",
    )
    if uploaded is None:
        st.info("Upload a CSV or Excel file to continue.")
        st.stop()

    is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
    sheet_name: str | int = 0
    if is_excel:
        peek_bytes = uploaded.read(); uploaded.seek(0)
        try:
            sheets = pd.ExcelFile(io.BytesIO(peek_bytes)).sheet_names
            sheet_name = (st.selectbox("Select sheet", sheets, key=f"{KEY_PREFIX}_sheet")
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

    # ══════════════════════════════════════════════════════════════════════
    # Step 2 — Data Filtering
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 2 · Data Filtering")
    st.caption(
        "Filter rows before training. Filters are non-destructive — adjust any time."
    )
    df_raw = render_data_filters(df_raw, key_prefix=f"{KEY_PREFIX}_filter")
    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 3 — Target
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 3 · Select Target")
    default_target = next((c for c in TARGET_CANDIDATES if c in all_cols), all_cols[-1])
    target_col = st.selectbox(
        "Target column", all_cols,
        index=all_cols.index(default_target),
        key=f"{KEY_PREFIX}_target",
    )
    non_target = [c for c in all_cols if c != target_col]

    # ── Target conversion (numeric targets only) ────────────────────────────
    _y_probe  = pd.to_numeric(df_raw[target_col], errors="coerce")
    _y_dropna = _y_probe.dropna()
    _n_unique    = _y_dropna.nunique()
    _frac_nonint = (_y_dropna % 1 != 0).mean() if len(_y_dropna) > 0 else 0
    _is_numeric_target = (
        len(_y_dropna) >= 5
        and _n_unique > 20
        and (
            _frac_nonint >= 0.02
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

    # ══════════════════════════════════════════════════════════════════════
    # Step 4 — Chemical component groups
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 4 · Chemical Component Groups")
    st.caption("Default columns pre-selected from uploaded file. Add or remove freely.")

    sel_groups: dict[str, list[str]] = {}
    grp_items = [g for g in GROUPS if g not in ("Oil", "Brine", "Process")]

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
                    key=f"{KEY_PREFIX}_grp_{grp}",
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
            sel_groups[grp] = st.multiselect(
                f"Columns for {grp}",
                options=non_target,
                default=_avail(GROUPS[grp], non_target),
                key=f"{KEY_PREFIX}_grp_{grp}",
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
                                key=f"{KEY_PREFIX}_proc_temp")
    with pc2:
        _def_dil = next((c for c in non_target
                        if re.sub(r"\s+", " ", _normalise(c))
                        in ("dilution ratio", "dilution ratio corrected", "dilution")), NONE)
        proc_dil = st.selectbox("💧 Dilution Ratio", none_opt,
                                index=none_opt.index(_def_dil) if _def_dil in none_opt else 0,
                                key=f"{KEY_PREFIX}_proc_dil")
    with pc3:
        _def_oilpct = next((c for c in non_target
                            if re.sub(r"\s+", " ", _normalise(c))
                            in ("oil", "oil percent", "oil pct")), NONE)
        proc_oil_pct = st.selectbox("🛢️ Oil Percent (%)", none_opt,
                                    index=none_opt.index(_def_oilpct) if _def_oilpct in none_opt else 0,
                                    key=f"{KEY_PREFIX}_proc_oil_pct")

    sel_groups["Process"] = [c for c in [proc_temp, proc_dil, proc_oil_pct] if c != NONE]

    _proc_already = set(sel_groups["Process"])
    _all_assigned  = set(c for g, cols in sel_groups.items() for c in cols)

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
        key=f"{KEY_PREFIX}_proc_extra",
        label_visibility="collapsed",
    )
    sel_groups["Process"] = list(dict.fromkeys(sel_groups["Process"] + proc_extra))

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

    # ══════════════════════════════════════════════════════════════════════
    # Step 5 — Computed features
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 5 · Computed Features")

    df_work       = compute_sum_features(df_raw, sel_groups)
    computed_sums = [f for f in SUM_FEATURES if f in df_work.columns]

    if computed_sums:
        pills = " ".join(
            f'<span class="group-pill" style="background:#e3f2fd;color:#1565C0;'
            f'border:1px solid #90caf9">{s}</span>'
            for s in computed_sums
        )
        st.markdown("**Auto-computed sums:**")
        st.markdown(pills, unsafe_allow_html=True)

    user_ratios: list[tuple[str, str, str]] = []
    avail_for_ratio = list(dict.fromkeys(non_target + computed_sums))

    with st.expander("🧪 Custom Engineered Features", expanded=False):
        st.caption("Create new features by combining two columns with + or /")
        n_custom = st.number_input("Number of custom engineered features",
                                    0, 20, 0, 1, key=f"{KEY_PREFIX}_n_custom")
        for ri in range(int(n_custom)):
            st.markdown(f"**Custom Feature #{ri+1}**")
            cc1, cc2, cc3 = st.columns([5, 2, 5])
            c1_sel = cc1.selectbox("Component 1", avail_for_ratio, key=f"{KEY_PREFIX}_cf_c1_{ri}")
            op_sel = cc2.selectbox("Operation",   ["+", "/"],       key=f"{KEY_PREFIX}_cf_op_{ri}")
            c2_sel = cc3.selectbox("Component 2", avail_for_ratio, key=f"{KEY_PREFIX}_cf_c2_{ri}")
            user_ratios.append((c1_sel, op_sel, c2_sel))

    for _c1, _op, _c2 in user_ratios:
        _fname = f"{_c1} {_op} {_c2}"
        if _c1 in df_work.columns and _c2 in df_work.columns and _fname not in df_work.columns:
            _c1_num = _safe_num(df_work[_c1])
            _c2_num = _safe_num(df_work[_c2])
            df_work[_fname] = (_c1_num + _c2_num if _op == "+"
                               else _c1_num / _c2_num.replace(0, float("nan")))

    # ── Must-have ratio / sum features ──────────────────────────────────────
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
            _winner, _loser  = _sum_name, _ratio_name
            _r_win, _r_lose  = _r_sum, _r_ratio
        else:
            _winner, _loser  = _ratio_name, _sum_name
            _r_win, _r_lose  = _r_ratio, _r_sum
        if _winner not in EXCLUDED_IX:
            _best_cols.append(_winner)
        _corr_report.append({
            "Pair":       f"{_num}  ×  {_den}",
            "Winner":     _winner,
            "Winner r":   round(_r_win,  4),
            "Winner |r|": round(abs(_r_win), 4),
            "Loser":      _loser,
            "Loser r":    round(_r_lose, 4),
            "Loser |r|":  round(abs(_r_lose), 4),
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
                f'<span class="group-pill" style="background:#fff8e1;color:#795548;'
                f'border:1px solid #bcaaa4">{c}</span>'
                for c in all_ratio_cols
            )
            st.markdown("**Selected features (winners + custom):**")
            st.markdown(_ratio_pills, unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 6 — NaN Handling
    # ══════════════════════════════════════════════════════════════════════
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
            key=f"{KEY_PREFIX}_chem_nan_strategy",
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

        all_chem_cols_sel = [c for g in CHEM_GROUPS for c in sel_groups.get(g, [])
                            if c in df_work.columns]
        if all_chem_cols_sel:
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

    # ══════════════════════════════════════════════════════════════════════
    # Step 7 — Model settings
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 7 · Model Settings")

    with st.container(border=True):
        ms1, ms2, ms3, ms4 = st.columns(4)
        n_estimators = ms1.number_input("Trees", 50, 1000, 300, 50, key=f"{KEY_PREFIX}_n_est")
        max_depth    = ms2.number_input("Max depth (0=unlimited)", 0, 30, 0, 1, key=f"{KEY_PREFIX}_mdepth")
        max_depth    = None if max_depth == 0 else int(max_depth)
        random_state = ms3.number_input("Random seed", 0, 999, 42, 1, key=f"{KEY_PREFIX}_rseed")

        task_override = ms4.selectbox(
            "Task type",
            ["From target conversion", "regression", "classification"],
            key=f"{KEY_PREFIX}_task_override",
        )
        task = task_from_conversion if task_override == "From target conversion" else task_override
        st.caption(f"From target conversion: **{task_from_conversion}** → Using: **{task}**")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 8 — SHAP plot configuration
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 8 · SHAP Plot Configuration")

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

    all_x_candidates = list(dict.fromkeys(
        individual_selected + must_have_present + computed_sums + all_ratio_cols
    ))
    all_color_candidates = list(dict.fromkeys(
        individual_selected + must_have_present + computed_sums
        + all_ratio_cols + oil_cols_in_data
    ))

    with st.container(border=True):
        st.markdown("##### 📊 SHAP Plot Configuration")

        st.markdown("**X-axis features** (one folder per feature):")
        st.caption("Select any individual, sum, or ratio feature from the final dataset.")
        _X_DEFAULTS = ["APG (%)", "AOS (%)", "CapB (%)", "Divalent", "Monovalent"]
        _x_def = _avail(_X_DEFAULTS, all_x_candidates)
        x_feats_sf1 = st.multiselect(
            "X-axis features",
            options=all_x_candidates,
            default=list(dict.fromkeys(must_have_present + _x_def)),
            key=f"{KEY_PREFIX}_x_feats",
            label_visibility="collapsed",
        )

        st.markdown("**Colour-by features** (peers + these, per folder):")
        st.caption("Select any column — individual, sum, ratio, or oil property.")
        _COLOR_DEFAULTS = ["Dilution Ratio", "Dilution Ratio_Corrected"]
        _c_def          = _avail(_COLOR_DEFAULTS, all_color_candidates)
        color_feats     = st.multiselect(
            "Colour features",
            options=all_color_candidates,
            default=_c_def or all_color_candidates[:min(3, len(all_color_candidates))],
            key=f"{KEY_PREFIX}_color_feats",
            label_visibility="collapsed",
        )

    def _n_cols(x):
        peers  = [c for c in x_feats_sf1 if c != x]
        extras = [c for c in color_feats  if c != x and c not in peers]
        return len(peers) + len(extras)

    n_plots = sum(_n_cols(x) for x in x_feats_sf1 if x in df_work.columns)
    st.info(f"Will generate **{n_plots}** SHAP plots across **{len(x_feats_sf1)}** X-axis folders.")
    st.divider()

    # ── Data overview before training ───────────────────────────────────────
    with st.expander("📋 Data Overview (before training)", expanded=True):
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
        _prev_target_num = _safe_num(_prev[target_col])
        if _prev_target_num.notna().mean() > 0.5:
            _prev[target_col] = _prev_target_num
        _prev = _prev.dropna(subset=[target_col])

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

        _nan_pct = (_prev[_feature_cols_prev].isnull().mean() * 100).round(1)
        _nan_remaining = _nan_pct[_nan_pct > 0]
        if _nan_remaining.empty:
            st.success("✅ No remaining NaN in any feature column.")
        else:
            st.warning(f"⚠️ {len(_nan_remaining)} feature(s) still have NaN — "
                    "will be filled with 0 before scaling.")
            st.dataframe(_nan_remaining.rename("NaN %").to_frame(),
                        use_container_width=True)

        st.dataframe(_prev[_feature_cols_prev], use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 9 — Validation Strategy (Group-Aware)  ── NEW
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 9 · Validation Strategy (Group-Aware)")
    st.caption(
        "Rows that share the same chemical formulation but differ only in "
        "process conditions (dilution ratio, temperature, brine) are near-duplicates. "
        "A random split can leak the same formulation (or the same oil, or the "
        "same brine) into both train and test, inflating scores. Pick which "
        "axis you want an honest generalization estimate for."
    )

    with st.container(border=True):
        group_by = st.radio(
            "Group rows by",
            GROUP_BY_OPTIONS,
            index=0,
            key=f"{KEY_PREFIX}_group_by",
        )

        if group_by == GROUP_BY_OPTIONS[1]:
            _group_cols  = [c for c in sel_groups.get("Oil", []) if c in df_work.columns]
            _group_kind  = "oil"
            _group_label = "unique oils"
        elif group_by == GROUP_BY_OPTIONS[2]:
            _group_cols  = [c for c in sel_groups.get("Brine", []) if c in df_work.columns]
            _group_kind  = "brine"
            _group_label = "unique brines"
        else:
            _group_cols  = [c for g in CHEM_GROUPS for c in sel_groups.get(g, [])
                            if c in df_work.columns]
            _group_kind  = "formulation"
            _group_label = "unique formulations"

        if _group_cols:
            _group_key_df  = df_work[_group_cols].apply(_safe_num).fillna(0).round(6)
            formulation_id = _group_key_df.astype(str).agg("|".join, axis=1)
        else:
            formulation_id = pd.Series(df_work.index, index=df_work.index).astype(str)
        df_work["_formulation_id"] = formulation_id

        n_unique_formulations = formulation_id.nunique()
        _dupe_counts = formulation_id.value_counts()
        n_repeat_formulations = int((_dupe_counts >= 2).sum())

        if not _group_cols:
            st.warning(
                f"⚠️ No columns available for {_group_kind} grouping with the "
                "current selection — falls back to one group per row "
                "(equivalent to a random split)."
            )
        elif _group_kind != "formulation":
            st.caption(
                f"With only **{n_unique_formulations}** {_group_label} in this "
                f"data, Group K-Fold here is effectively leave-one-{_group_kind}-out "
                "when the fold count is set to match — the most rigorous test of "
                f"generalization to a new {_group_kind}, but based on very few groups."
            )

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Rows", f"{len(df_work):,}")
        vc2.metric(f"Unique {_group_kind} groups", f"{n_unique_formulations:,}")
        vc3.metric("Groups w/ repeats", f"{n_repeat_formulations:,}")

        _default_method_idx = 0 if n_repeat_formulations > 0 else 1
        validation_method = st.radio(
            "Validation method",
            VALIDATION_METHODS,
            index=_default_method_idx,
            key=f"{KEY_PREFIX}_val_method",
            help="Group K-Fold keeps every row of a group in the same fold.",
        )

        _max_splits = min(MAX_N_SPLITS, n_unique_formulations)
        if _max_splits <= MIN_N_SPLITS:
            n_splits = MIN_N_SPLITS
            st.caption(f"Only {n_unique_formulations} unique {_group_kind} group(s) "
                       f"available — using {n_splits} folds.")
        else:
            n_splits = st.slider(
                "Number of folds", MIN_N_SPLITS, _max_splits,
                min(DEFAULT_N_SPLITS, _max_splits),
                key=f"{KEY_PREFIX}_n_splits",
                help=(f"Set to {n_unique_formulations} for a full "
                      f"leave-one-{_group_kind}-out test." if _group_kind != "formulation" else None),
            )

        log_transform = None
        if task == "regression":
            log_transform = st.radio(
                "Target transform",
                LOG_TRANSFORM_OPTIONS,
                index=0,
                key=f"{KEY_PREFIX}_log_transform",
                horizontal=True,
                help="The target is right-skewed — log1p usually improves fit and residuals.",
            )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 10 — Save destination
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 10 · Save Destination")
    save_mode = st.radio("How to get the plots?",
                        ["⬇️ Download as ZIP", "💾 Save to disk path"],
                        horizontal=True, key=f"{KEY_PREFIX}_save_mode")
    disk_path = ""
    if "disk" in save_mode:
        disk_path = st.text_input(
            "Destination folder", value=str(Path.home() / "foam_shap_plots"),
            key=f"{KEY_PREFIX}_disk_path",
        )
    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 11 — Group-Aware Validation (honest held-out metrics)  ── NEW
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 11 · Group-Aware Validation")
    st.caption(
        "Runs k-fold cross-validation using the settings from Step 9, honestly "
        "estimating how well the model generalises to a *new group* along the "
        "chosen axis (formulation, oil, or brine) — not just re-scoring "
        "training data. Does not affect the final model below."
    )

    if st.button("🔎 Run Group-Aware Validation",
                key=f"{KEY_PREFIX}_val_btn", use_container_width=True):

        all_feature_cols = list(dict.fromkeys(
            c for g, cols in sel_groups.items() for c in cols
        ))
        df_for_val = df_work.copy()
        df_for_val[target_col] = y_converted.values

        chem_cols_v = [c for g in CHEM_GROUPS for c in sel_groups.get(g, []) if c in df_for_val.columns]
        cond_cols_v = [c for g in COND_GROUPS for c in sel_groups.get(g, []) if c in df_for_val.columns]
        feature_cols_v = list(dict.fromkeys(all_feature_cols + computed_sums + all_ratio_cols))
        feature_cols_v = [c for c in feature_cols_v if c in df_for_val.columns and c != target_col]

        with st.spinner("Preprocessing…"):
            try:
                X_val, y_val, feat_names_val, _ = preprocess(
                    df_for_val, feature_cols_v, target_col, chem_cols_v, cond_cols_v, task,
                    chem_nan_strategy=chem_nan_strategy,
                )
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                st.stop()

        groups_val = formulation_id.reindex(X_val.index)
        use_log = task == "regression" and log_transform == LOG_TRANSFORM_OPTIONS[0]
        y_fit = np.log1p(y_val.clip(lower=0)) if use_log else y_val

        try:
            if validation_method == VALIDATION_METHODS[0]:
                splitter  = GroupKFold(n_splits=int(n_splits))
                split_idx = list(splitter.split(X_val, y_fit, groups=groups_val))
            elif task == "regression":
                splitter  = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
                split_idx = list(splitter.split(X_val, y_fit))
            else:
                splitter  = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
                split_idx = list(splitter.split(X_val, y_fit))
        except ValueError as e:
            st.error(f"Could not build folds: {e}. Try fewer folds.")
            st.stop()

        oof_pred  = pd.Series(index=X_val.index, dtype=float)
        fold_rows: list[dict] = []

        with st.spinner(f"Running {n_splits}-fold validation…"):
            for fold_i, (tr_idx, te_idx) in enumerate(split_idx, 1):
                X_tr, X_te = X_val.iloc[tr_idx], X_val.iloc[te_idx]
                y_tr, y_te = y_fit.iloc[tr_idx], y_fit.iloc[te_idx]

                if task == "regression":
                    fold_model = RandomForestRegressor(
                        n_estimators=int(n_estimators), max_depth=max_depth,
                        min_samples_leaf=3, random_state=int(random_state), n_jobs=-1,
                    )
                else:
                    fold_model = RandomForestClassifier(
                        n_estimators=int(n_estimators), max_depth=max_depth,
                        min_samples_leaf=3, random_state=int(random_state), n_jobs=-1,
                    )
                fold_model.fit(X_tr, y_tr)
                pred = fold_model.predict(X_te)
                oof_pred.iloc[te_idx] = pred

                if task == "regression":
                    y_te_orig   = np.expm1(y_te) if use_log else y_te
                    pred_orig   = np.expm1(pred) if use_log else pred
                    fold_rows.append({
                        "Fold": fold_i, "Rows": len(te_idx),
                        "R2":   r2_score(y_te_orig, pred_orig),
                        "MAE":  mean_absolute_error(y_te_orig, pred_orig),
                        "RMSE": float(np.sqrt(mean_squared_error(y_te_orig, pred_orig))),
                    })
                else:
                    fold_rows.append({
                        "Fold": fold_i, "Rows": len(te_idx),
                        "Accuracy":      accuracy_score(y_te, pred),
                        "F1 (weighted)": f1_score(y_te, pred, average="weighted", zero_division=0),
                    })

        fold_df = pd.DataFrame(fold_rows)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        if task == "regression":
            y_all_orig = np.expm1(y_fit) if use_log else y_fit
            oof_orig   = np.expm1(oof_pred) if use_log else oof_pred
            overall_r2   = r2_score(y_all_orig, oof_orig)
            overall_mae  = mean_absolute_error(y_all_orig, oof_orig)
            overall_rmse = float(np.sqrt(mean_squared_error(y_all_orig, oof_orig)))

            m1, m2, m3 = st.columns(3)
            m1.metric("Held-out R² (pooled)", f"{overall_r2:.3f}",
                    help=f"Fold mean ± std: {fold_df['R2'].mean():.3f} ± {fold_df['R2'].std():.3f}")
            m2.metric("Held-out MAE (pooled)",  f"{overall_mae:.2f}")
            m3.metric("Held-out RMSE (pooled)", f"{overall_rmse:.2f}")

            _scatter_df = pd.DataFrame({
                "Actual": y_all_orig.values,
                "Predicted (out-of-fold)": oof_orig.values,
            })
            fig = px.scatter(
                _scatter_df, x="Actual", y="Predicted (out-of-fold)",
                opacity=0.6, template="plotly_white",
                title="Out-of-Fold Predicted vs Actual",
            )
            _lims = [float(_scatter_df.min().min()), float(_scatter_df.max().max())]
            fig.add_shape(type="line", x0=_lims[0], y0=_lims[0], x1=_lims[1], y1=_lims[1],
                        line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            overall_acc = accuracy_score(y_fit, oof_pred)
            overall_f1  = f1_score(y_fit, oof_pred, average="weighted", zero_division=0)
            m1, m2 = st.columns(2)
            m1.metric("Held-out Accuracy (pooled)",       f"{overall_acc:.3f}")
            m2.metric("Held-out F1 weighted (pooled)",    f"{overall_f1:.3f}")

        st.caption(
            "⚠️ Compare this to the **train-set R² / Accuracy** shown after "
            "clicking 'Train Final Model' below — a much higher train score "
            "than this held-out score means the model is memorising groups "
            "along the chosen axis, not generalising to new ones."
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # Step 12 — Train Final Model & Generate SHAP Plots
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("### 12 · Train Final Model & Generate SHAP Plots")
    st.caption(
        "Fit on 100% of the data for interpretation/export. Trust the "
        "**Group-Aware Validation** metrics above for real-world performance, "
        "not the train-set R² / Accuracy shown here."
    )

    if not x_feats_sf1:
        st.warning("Select at least one X-axis feature.")
        st.stop()
    if not color_feats:
        st.warning("Select at least one colour feature.")
        st.stop()

    if st.button("🚀 Train Final Model & Generate Plots",
                key=f"{KEY_PREFIX}_train_btn", type="primary", use_container_width=True):

        all_feature_cols = list(dict.fromkeys(
            c for g, cols in sel_groups.items() for c in cols
        ))
        df_for_model = df_work.copy()
        df_for_model[target_col] = y_converted.values

        chem_cols = [c for g in CHEM_GROUPS for c in sel_groups.get(g, []) if c in df_for_model.columns]
        cond_cols = [c for g in COND_GROUPS for c in sel_groups.get(g, []) if c in df_for_model.columns]

        feature_cols = list(dict.fromkeys(all_feature_cols + computed_sums + all_ratio_cols))
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

        X_orig = df_for_model.loc[X_scaled.index, [c for c in feat_names
                                                    if c in df_for_model.columns]].copy()
        for col in feat_names:
            if col not in X_orig.columns:
                X_orig[col] = X_scaled[col]

        import hashlib as _hl
        _hash_src = str(X_scaled.shape) + str(list(X_scaled.columns)) + str(int(n_estimators))
        X_hash = _hl.md5(_hash_src.encode()).hexdigest()
        model, X_shap, shap_vals = train_model(
            X_hash, X_scaled, y, task,
            int(n_estimators), max_depth, int(random_state),
        )
        X_shap_orig = X_orig.loc[X_shap.index].copy()

        y_pred = model.predict(X_scaled)
        if task == "regression":
            score = r2_score(y, y_pred)
            st.metric("R² (train — 100% of data)", f"{score:.4f}")
        else:
            score = accuracy_score(y, y_pred)
            st.metric("Accuracy (train — 100% of data)", f"{score:.4f}")

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        imp_df = pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_abs_shap})
        imp_df = imp_df.sort_values("Mean |SHAP|", ascending=False).head(20)
        with st.expander("📊 Top 20 Features by Mean |SHAP|", expanded=True):
            st.bar_chart(imp_df.set_index("Feature"))

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

        with st.spinner("Generating Model Performance plots…"):
            perf_figs = _generate_model_performance(
                model, X_scaled, y, task, feat_names, shap_vals, X_shap_orig, target_col
            )
            figures.update({f"{SF_PERF}/{k}": v for k, v in perf_figs.items()})

        SF_TOP = "Most Important Features"
        with st.spinner("Generating Most Important Features SHAP plots…"):
            _n_top    = min(20, len(feat_names))
            _mean_abs = np.abs(shap_vals).mean(axis=0)
            _top_idx  = np.argsort(_mean_abs)[::-1][:_n_top]
            _top_feats = [feat_names[i] for i in _top_idx]

            def _best_color(x_feat, x_idx):
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
                title = f"Top #{rank}  |  X={x_feat}  |  Colour={col_feat}"
                fig = _shap_dependence(X_shap_orig, shap_vals, feat_names,
                                       x_feat, col_feat, title)
                fname = f"{rank:02d}_{x_feat}_Color_{col_feat}.png".replace("/", "_")
                top_figs[fname] = _render_to_bytes(fig)

            figures.update({f"{SF_TOP}/{k}": v for k, v in top_figs.items()})

        st.success(
            f"✅ Generated **{len(f1)}** SHAP + "
            f"**{len(perf_figs)}** performance + "
            f"**{len(top_figs)}** top-feature = **{len(figures)}** plots total."
        )

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
                file_name="foam_shap_plots_validated.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"{KEY_PREFIX}_download_zip",
            )

        st.divider()
        st.markdown("#### Preview (first 4 plots)")
        prev_cols = st.columns(2)
        for pi, (path_key, png_bytes) in enumerate(list(figures.items())[:4]):
            with prev_cols[pi % 2]:
                st.image(io.BytesIO(png_bytes), caption=path_key.split("/")[-1],
                        use_column_width=True)
