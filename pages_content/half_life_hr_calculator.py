import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from core.data.loader import load_csv

# =========================================
# Config
# =========================================
POLY_DEGREE_DEFAULT = 2  # you can expose in UI if you want

# =========================================
# 1) Identify & parse time columns (-> minutes)
# =========================================
def parse_time_to_minutes(col):
    inside = re.search(r'\((.*?)\)', col)
    token = inside.group(1) if inside else col
    m = re.search(r'^\s*([\d.]+)\s*([a-zA-Z]*)\s*$', token)
    if not m:
        return np.nan
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ["s", "sec", "secs", "second", "seconds"]:
        return val / 60.0
    elif unit in ["h", "hr", "hrs", "hour", "hours"]:
        return val * 60.0
    return val  # minutes

import re
import numpy as np

import re
import numpy as np

def parse_time_to_minutes(col_name):
    """
    Robust time parser:
    Handles:
    - Time (30m), Time(1h), time 0.5 h
    - 0h, 1h, 30m
    - Time-30m, t=30m
    """

    col = col_name.lower().strip()

    # Strong pattern: number + explicit unit
    match = re.search(r'(\d+\.?\d*)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)\b', col)
    
    if match:
        value = float(match.group(1))
        unit = match.group(2)

        if unit.startswith('h'):
            return value * 60.0
        else:
            return value

    # Fallback ONLY if column clearly looks like time
    if re.search(r'\btime\b|\bt\s*=', col):
        match_num = re.search(r'(\d+\.?\d*)', col)
        if match_num:
            return float(match_num.group(1))  # assume minutes

    return np.nan


import re
import pandas as pd
import numpy as np
import streamlit as st

def find_time_columns(df):
    """
    Identifies time-related columns, cleans asterisk markings, 
    converts to numeric, and sorts them chronologically.
    """
    tcols = []
    
    for c in df.columns:
        c_low = c.lower()
        if (
            re.search(r'\btime\b', c_low) or
            re.match(r'^\s*\d+\.?\d*\s*(h|m|hr|mins|minutes|min)\b', c_low)
        ):
            tcols.append(c)

    times_min, keep_cols = [], []

    for col in tcols:
        # 1. Clean the column: Remove '*' and convert to numeric
        # We use regex=True to replace all asterisks with empty strings
        # Removes any character that isn't a digit (0-9) or a decimal point (.)
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), 
            errors='coerce'
        )
        
        tm = parse_time_to_minutes(col)
        if np.isfinite(tm):
            keep_cols.append(col)
            times_min.append(tm)

    order = np.argsort(times_min)
    sorted_cols = [keep_cols[i] for i in order]
    sorted_times = np.array([times_min[i] for i in order], dtype=float)

    # ✅ Streamlit info message with Type Check
    if sorted_cols:
        status_lines = []
        for col, t in zip(sorted_cols, sorted_times):
            # Check if the column is successfully converted to a numeric type
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            type_label = "Number (Pass)" if is_numeric else "Text (Fail)"
            
            status_lines.append(f"- {col} → {int(t)} min - Type: {type_label}")

        info_msg = f"⏱️ **Detected {len(sorted_cols)} time columns (sorted ascending):**\n" + "\n".join(status_lines)
        st.info(info_msg)
    else:
        st.warning("⚠️ No valid time columns detected.")
        raise ValueError("No time-like columns found. Make sure headers contain 'Time (...)'.")

    return sorted_cols, sorted_times

# =========================================
# 2) Fit helpers  (return R², yhat, params, eq, extra)
# =========================================
def fit_linear(t, y):
    """y = a*t + b"""
    model = LinearRegression().fit(t.reshape(-1,1), y)
    yhat = model.predict(t.reshape(-1,1))
    r2 = r2_score(y, yhat) if len(y) >= 2 else np.nan
    a = float(model.coef_[0]); b = float(model.intercept_)
    eq = f"y = {a:.6g} * t + {b:.6g}"
    params = (a, b)
    return r2, yhat, params, eq, None  # extra=None

def fit_exponential(t, y):
    """
    y = A * exp(B*t)
    ln(y) = ln(A) + B*t (y>0 required)
    """
    mask = y > 0
    if mask.sum() < 2:
        return np.nan, np.full_like(y, np.nan, dtype=float), (np.nan, np.nan), "y = A * exp(B*t)", None
    t2 = t[mask].reshape(-1,1)
    ly = np.log(y[mask])
    model = LinearRegression().fit(t2, ly)
    B = float(model.coef_[0])
    lnA = float(model.intercept_)
    A = np.exp(lnA)
    yhat_full = A * np.exp(B * t)
    r2 = r2_score(y[mask], yhat_full[mask]) if mask.sum() >= 2 else np.nan
    eq = f"y = {A:.6g} * exp({B:.6g} * t)"
    params = (A, B)
    return r2, yhat_full, params, eq, None

def fit_polynomial(t, y, degree=3):
    """
    y = c0 + c1*t + c2*t^2 + ... + c_deg*t^deg
    np.polyfit returns coeffs in descending order: [c_deg, ..., c1, c0]
    """
    if len(y) < degree + 1:
        return np.nan, np.full_like(y, np.nan, dtype=float), None, f"y = poly(deg={degree})", None
    coeffs_desc = np.polyfit(t, y, degree)  # [c_deg ... c0]
    yhat = np.polyval(coeffs_desc, t)
    r2 = r2_score(y, yhat) if len(y) >= 2 else np.nan

    # Build eq string
    terms = []
    for i, c in enumerate(coeffs_desc):
        p = degree - i
        if p == 0:
            terms.append(f"{c:.6g}")
        elif p == 1:
            terms.append(f"{c:.6g}*t")
        else:
            terms.append(f"{c:.6g}*t^{p}")
    eq = "y = " + " + ".join(terms)

    params = coeffs_desc  # keep descending for root solving
    return r2, yhat, params, eq, {"degree": degree}

# =========================================
# 3) Evaluate rows (returns result DF + per-row params)
# =========================================
def evaluate_rows(df, r2_threshold=0.80, poly_degree=POLY_DEGREE_DEFAULT):
    time_cols, times_min = find_time_columns(df)

    r2_lin_list, r2_exp_list, r2_poly_list = [], [], []
    eq_lin_list, eq_exp_list, eq_poly_list = [], [], []
    best_model_list = []

    # store params for half-life calc
    lin_params_list, exp_params_list, poly_params_list = [], [], []
    poly_degree_list = []

    for _, row in df.iterrows():
        y_all = pd.to_numeric(row[time_cols], errors='coerce').values.astype(float)
        t_all = times_min
        mask = np.isfinite(y_all) & np.isfinite(t_all)
        t = t_all[mask]
        y = y_all[mask]

        if len(y) < 3:
            r2_lin = r2_exp = r2_poly = np.nan
            eq_lin  = "y = a*t + b"
            eq_exp  = "y = A * exp(B*t)"
            eq_poly = f"y = poly(deg={poly_degree})"
            best_model = "none"
            lin_params = (np.nan, np.nan)
            exp_params = (np.nan, np.nan)
            poly_params = None
            this_poly_degree = poly_degree
        else:
            # Linear
            r2_lin, _, lin_params, eq_lin, _ = fit_linear(t, y)

            # Exponential
            r2_exp, _, exp_params, eq_exp, _ = fit_exponential(t, y)

            # Polynomial
            r2_poly, _, poly_params, eq_poly, extra = fit_polynomial(t, y, degree=poly_degree)
            this_poly_degree = extra["degree"] if extra else poly_degree

            # Best by R² with threshold
            r2s = {
                "linear": r2_lin if np.isfinite(r2_lin) else -np.inf,
                "exp":    r2_exp if np.isfinite(r2_exp) else -np.inf,
                "poly":   r2_poly if np.isfinite(r2_poly) else -np.inf,
            }
            best_name = max(r2s, key=r2s.get)
            best_r2 = r2s[best_name]
            best_model = best_name if (np.isfinite(best_r2) and best_r2 >= r2_threshold) else "none"

        r2_lin_list.append(r2_lin); r2_exp_list.append(r2_exp); r2_poly_list.append(r2_poly)
        eq_lin_list.append(eq_lin); eq_exp_list.append(eq_exp); eq_poly_list.append(eq_poly)
        best_model_list.append(best_model)

        lin_params_list.append(lin_params)
        exp_params_list.append(exp_params)
        poly_params_list.append(poly_params)
        poly_degree_list.append(this_poly_degree)

    out = df.copy()
    out["r2_linear"] = r2_lin_list
    out["r2_exp"]    = r2_exp_list
    out["r2_poly"]   = r2_poly_list

    out["equ_linear"] = eq_lin_list
    out["equ_exp"]    = eq_exp_list
    out["equ_poly"]   = eq_poly_list

    out["best_model"] = best_model_list

    # also return params/times to compute half-life later
    params = {
        "time_cols": time_cols,
        "times_min": times_min,
        "lin_params": lin_params_list,
        "exp_params": exp_params_list,
        "poly_params_desc": poly_params_list,  # descending
        "poly_degree": poly_degree_list
    }
    return out, params

# =========================================
# 4) Half-life utilities
# =========================================
def initial_volume(row, time_cols):
    """Use the earliest time column (already sorted in evaluate_rows) as initial volume."""
    for c in time_cols:
        v = pd.to_numeric(row.get(c), errors='coerce')
        if pd.notna(v):
            return float(v)
    return np.nan

def half_life_linear(y0, params):
    # y = a*t + b => t = (0.5*y0 - b)/a
    a, b = params
    if not np.isfinite(a) or a == 0 or not np.isfinite(b) or not np.isfinite(y0):
        return np.nan
    t = (0.5*y0 - b) / a
    return t if t >= 0 else np.nan

def half_life_exp(y0, params):
    # y = A*exp(B*t), target = 0.5*y0 -> t = (ln(0.5*y0) - ln(A))/B
    A, B = params
    if not np.isfinite(A) or not np.isfinite(B) or A <= 0 or B == 0 or not np.isfinite(y0) or y0 <= 0:
        return np.nan
    target = 0.5 * y0
    if target <= 0:
        return np.nan
    t = (np.log(target) - np.log(A)) / B
    return t if t >= 0 else np.nan

def half_life_poly(y0, coeffs_desc):
    # Solve p(t) = 0.5*y0  =>  p(t) - 0.5*y0 = 0  (coeffs_desc are descending)
    if coeffs_desc is None or not np.isfinite(y0):
        return np.nan
    coeffs = coeffs_desc.copy()
    coeffs[-1] -= 0.5 * y0  # subtract constant term
    roots = np.roots(coeffs)  # complex allowed
    # choose smallest non-negative real root
    real_roots = [r.real for r in roots if np.isreal(r) and r.real >= 0]
    if not real_roots:
        return np.nan
    return float(np.min(real_roots))

def compute_half_life_column(result_df, params, method_choice, unit="hours"):
    """
    method_choice: 'best', 'linear', 'exp', 'poly'
    Times are in minutes; output is in hours by default.
    """
    time_cols = params["time_cols"]
    lin_params_list = params["lin_params"]
    exp_params_list = params["exp_params"]
    poly_params_list = params["poly_params_desc"]
    timescale = 60.0 if unit == "hours" else 1.0  # minutes->hours

    hl_minutes = []
    method_used = []

    for i, row in result_df.iterrows():
        y0 = initial_volume(row, time_cols)
        pick = method_choice
        if pick == "best":
            pick = row.get("best_model", "none")

        if pick == "linear":
            t_min = half_life_linear(y0, lin_params_list[i])
        elif pick == "exp":
            t_min = half_life_exp(y0, exp_params_list[i])
        elif pick == "poly":
            t_min = half_life_poly(y0, poly_params_list[i])
        else:
            t_min = np.nan

        hl_minutes.append(t_min)
        method_used.append(pick)

    # Convert to hours if requested
    hl_out = [v / timescale if np.isfinite(v) else np.nan for v in hl_minutes]

    # Make a friendly column name
    label_map = {"best": "row-wise", "linear": "linear", "exp": "expo", "poly": "poly"}
    label = label_map.get(method_choice, str(method_choice))
    col_name = f"Half life ({'best: ' + label if method_choice=='best' else 'method: ' + label})"
    col_name += " [h]" if unit == "hours" else " [min]"

    out = result_df.copy()
    out[col_name] = hl_out
    # keep which was actually used per row (important when 'best')
    out["HL method used"] = method_used
    return out, col_name

# =====================================================
# Streamlit Interface
# =====================================================
def render():
    st.markdown("## ⏳ Foam Half-Life Calculator")

    with st.expander("📢 Help & Instructions", expanded=False):
        st.markdown("""
        **Half-life** here is the time for foam volume to drop to **half of the initial volume** (earliest time).
        - Time columns like `Time (0.0)`, `Time (30m)`, `Time (3h)` are auto-parsed and converted to minutes.
        - Fits done per row: **Linear**, **Exponential**, **Polynomial** (degree configurable in code).
        - Choose R² threshold to accept the best model; otherwise the row shows **none**.
        - Half-life is reported in **hours**.
        """)

    st.divider()
    uploaded_file = st.file_uploader("📂 Upload dataset (CSV or Excel)", type=["csv", "xlsx"])

    # --- Controls ---
    c1, c2 = st.columns([1,1])
    with c1:
        r2_threshold = st.number_input("R² threshold for best model", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
    with c2:
        method_choice_ui = st.selectbox(
            "Method to use for Half-Life",
            ["Best per-row (uses best_model)", "Linear", "Exponential", "Polynomial (deg=2)"],
            index=0
        )

    # normalize method key
    method_key_map = {
        "Best per-row (uses best_model)": "best",
        "Linear": "linear",
        "Exponential": "exp",
        "Polynomial (deg=2)": "poly",
    }

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = load_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # clean asterisks if present
            df = df.replace(r"\*", "", regex=True)

            st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} cols")
            st.dataframe(df.head())

            if st.button("▶️ Run fitting & compute Half-Life"):
                # 1) evaluate fits with chosen threshold
                result, params = evaluate_rows(df, r2_threshold=r2_threshold, poly_degree=POLY_DEGREE_DEFAULT)

                # 2) compute half-life using chosen method
                method_choice = method_key_map[method_choice_ui]
                result_with_hl, hl_col = compute_half_life_column(result, params, method_choice, unit="hours")

                # round the Results for better readability in the downloaded CSV
                result_with_hl[hl_col] = (
                    result_with_hl[hl_col]
                    .replace([np.inf, -np.inf], np.nan)
                    .round(2)
                    .astype("float64")   # 👈 allows NaN safely
                )

                st.write("### 📈 Fit Summary")
                st.dataframe(result_with_hl)

                # Download
                cols_to_remove = ["r2_linear", "r2_exp", "r2_poly",
                                "equ_linear", "equ_exp", "equ_poly" , "HL method used"]

                result_with_hl = result_with_hl.drop(columns=cols_to_remove, errors="ignore")
                # utf-8-sig (BOM) so Excel renders accented/non-breaking-space
                # characters correctly instead of as "Â" mojibake.
                csv = result_with_hl.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="💾 Download CSV (R², equations, half-life)",
                    data=csv,
                    file_name="foam_half_life_R2_equ.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.info("👆 Upload a CSV or Excel file to begin.")
