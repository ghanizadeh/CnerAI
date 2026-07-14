"""
core/optimisation.py
====================
Safe-region detection, Bayesian optimisation, mixture sampling,
active learning next-experiment suggestions.
No Streamlit imports. No references to tab state.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import dirichlet

# ── Compatibility shim: inline config objects replacing code0's config module ──
from dataclasses import dataclass

@dataclass
class _OptimCfg:
    dirichlet_alpha: float = 1.0
    quantile_low: float = 0.05
    quantile_high: float = 0.95
    n_bo_calls: int = 50
    n_bo_random_starts: int = 10
    n_active_suggestions: int = 5
    active_uncertainty_method: str = "entropy"

@dataclass
class _UICfg:
    decimal_places: int = 2

OPTIM_CFG = _OptimCfg()
UI_CFG    = _UICfg()

warnings.filterwarnings("ignore")


def _skopt_available() -> bool:
    try:
        import skopt  # noqa: F401
        return True
    except ImportError:
        return False


def _scale_for_model(
    df_orig: pd.DataFrame,
    scaler,
    scaled_numeric_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Scale a DataFrame that is in original units into the model's input units.

    The model is trained on whatever Preprocessing produced (scaled, if a
    scaler was configured). Every place in this module that calls
    model.predict / model.predict_proba on candidate rows built in original
    units MUST go through this first, or predictions are computed on the
    wrong scale and are meaningless.

    No-op (returns a copy) when scaler is None, i.e. no scaling was used
    during training.
    """
    if scaler is None:
        return df_orig.copy()
    df_sc = df_orig.copy()
    cols = [c for c in (scaled_numeric_cols or df_sc.columns) if c in df_sc.columns]
    if cols:
        df_sc[cols] = scaler.transform(df_sc[cols])
    return df_sc


# ---------------------------------------------------------------------------
# Synthetic space generation
# ---------------------------------------------------------------------------

def sample_uniform(
    X_ref: pd.DataFrame,
    n_samples: int,
    feature_bounds: Optional[dict] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Uniform sampling inside observed or user-specified feature bounds."""
    rng = np.random.default_rng(random_state)
    synth = {}
    for c in X_ref.columns:
        col = pd.to_numeric(X_ref[c], errors="coerce")
        lo, hi = float(col.min()), float(col.max())
        if feature_bounds and c in feature_bounds:
            lo, hi = feature_bounds[c]
        synth[c] = np.full(n_samples, lo) if np.isclose(lo, hi) else rng.uniform(lo, hi, n_samples)
    return pd.DataFrame(synth)


def sample_dirichlet_mixture(
    features: list[str],
    feature_bounds: dict,
    mixture_total: float,
    n_samples: int,
    alpha: float = OPTIM_CFG.dirichlet_alpha,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample compositions that sum to `mixture_total` using Dirichlet."""
    alphas = np.full(len(features), alpha)
    raw = dirichlet.rvs(alphas, size=n_samples, random_state=int(random_state))
    df = pd.DataFrame(raw * mixture_total, columns=features)
    for c in features:
        if c in feature_bounds:
            lo, hi = feature_bounds[c]
            df[c] = df[c].clip(lo, hi)
    return df


def apply_constraints(
    df: pd.DataFrame,
    feature_bounds: Optional[dict] = None,
    sum_constraint: Optional[Tuple[list[str], float]] = None,
    sum_tolerance: float = 0.5,
) -> pd.DataFrame:
    """Filter rows violating feature bounds or sum constraint."""
    mask = pd.Series(True, index=df.index)
    if feature_bounds:
        for col, (lo, hi) in feature_bounds.items():
            if col in df.columns:
                mask &= df[col].between(lo, hi)
    if sum_constraint:
        cols, target = sum_constraint
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            mask &= (df[valid_cols].sum(axis=1) - target).abs() <= sum_tolerance
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_to_model(synth: pd.DataFrame, model) -> pd.DataFrame:
    """
    Reorder (and subset) synth columns to exactly match the feature order
    the model was trained on.  Works for CatBoost and sklearn (RF etc.).

    Resolution order for model feature names:
      1. model.feature_names_      — CatBoost
      2. model.feature_names_in_   — sklearn (set automatically when fit
                                     on a DataFrame)
      3. fall-through: return synth unchanged (model has no stored names)

    If the model expects a feature not present in synth, we add it as
    zeros rather than crashing — this keeps the app running even when
    the session state holds a stale model from a previous session.
    Users will see a warning in the UI (raised in the tab, not here).
    """
    model_features = None

    if hasattr(model, "feature_names_"):          # CatBoost
        try:
            model_features = list(model.feature_names_)
        except Exception:
            pass

    if model_features is None and hasattr(model, "feature_names_in_"):   # sklearn
        model_features = list(model.feature_names_in_)

    if model_features is None:
        return synth   # can't determine order — pass through

    # Add any missing columns as zeros (stale-model safety net)
    for f in model_features:
        if f not in synth.columns:
            synth = synth.copy()
            synth[f] = 0.0

    return synth[model_features]


# ---------------------------------------------------------------------------
# Prediction and safe region scoring
# ---------------------------------------------------------------------------

def score_synthetic_classification(
    model,
    synth: pd.DataFrame,
    class_names: list[str],
    safe_class_idx: int,
) -> pd.DataFrame:
    """Score synthetic samples; align column order to model before predicting."""
    synth_aligned = _align_to_model(synth, model)

    proba    = model.predict_proba(synth_aligned)
    pred_idx = np.argmax(proba, axis=1)

    result = synth.copy()
    result["pred_class_idx"]  = pred_idx
    result["pred_class"]      = [class_names[i] for i in pred_idx]
    result["safe_probability"] = proba[:, safe_class_idx]

    competing = proba.copy()
    competing[:, safe_class_idx] = -np.inf
    result["safety_margin"] = result["safe_probability"] - competing.max(axis=1)
    return result.round(UI_CFG.decimal_places)


def score_synthetic_regression(
    model,
    synth: pd.DataFrame,
    objective: str = "maximize",
) -> pd.DataFrame:
    synth_aligned = _align_to_model(synth, model)

    preds  = model.predict(synth_aligned)
    result = synth.copy()
    result["predicted_value"] = np.round(preds, UI_CFG.decimal_places)
    result["objective_score"] = result["predicted_value"] if objective == "maximize" else -result["predicted_value"]
    return result.round(UI_CFG.decimal_places)


def filter_safe_classification(
    scored: pd.DataFrame,
    safe_class_idx: int,
    confidence_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    safe    = scored[scored["pred_class_idx"] == safe_class_idx].copy()
    safe_hi = safe[safe["safe_probability"] >= confidence_threshold].copy()
    if safe_hi.empty:
        safe_hi = safe.sort_values(
            ["safe_probability", "safety_margin"], ascending=False
        ).head(min(200, len(safe)))
    return safe, safe_hi


def filter_optimal_regression(
    scored: pd.DataFrame,
    top_pct: float = 0.10,
) -> pd.DataFrame:
    n = max(1, int(len(scored) * top_pct))
    return scored.nlargest(n, "objective_score").copy()


# ---------------------------------------------------------------------------
# Recommended ranges
# ---------------------------------------------------------------------------

def build_recommended_ranges(
    df: pd.DataFrame,
    feature_cols: list[str],
    q_low: float  = OPTIM_CFG.quantile_low,
    q_high: float = OPTIM_CFG.quantile_high,
) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "Feature":                    c,
            f"Low ({int(q_low*100)}%)":   round(vals.quantile(q_low), UI_CFG.decimal_places),
            "Median":                     round(vals.quantile(0.50),  UI_CFG.decimal_places),
            f"High ({int(q_high*100)}%)": round(vals.quantile(q_high), UI_CFG.decimal_places),
            "Observed Min":               round(vals.min(),            UI_CFG.decimal_places),
            "Observed Max":               round(vals.max(),            UI_CFG.decimal_places),
        })
    return pd.DataFrame(rows)


def format_recommendation_text(ranges_df: pd.DataFrame, label: str) -> str:
    if ranges_df.empty:
        return f"No robust range could be extracted for '{label}'."
    cols     = ranges_df.columns.tolist()
    low_col  = next(c for c in cols if "Low"  in c)
    high_col = next(c for c in cols if "High" in c)
    lines    = [f"Recommended robust range for '{label}':"]
    for _, r in ranges_df.iterrows():
        lines.append(f"  {r['Feature']}: {r[low_col]} – {r[high_col]}  (median ≈ {r['Median']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bayesian Optimisation
# ---------------------------------------------------------------------------

def bayesian_optimise(
    model,
    feature_cols: list[str],
    X_ref: pd.DataFrame,             # ORIGINAL units — for sampling bounds + display
    feature_bounds: dict,
    safe_class_idx: int,
    class_names: list[str],
    problem_type: str,
    objective: str = "maximize",
    n_calls: int = OPTIM_CFG.n_bo_calls,
    n_random_starts: int = OPTIM_CFG.n_bo_random_starts,
    random_state: int = 42,
    scaler=None,                     # fitted scaler used during training, or None
    scaled_numeric_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Search for the single best formulation according to the trained model.

    X_ref, feature_bounds, and every value in the returned (scored, best)
    pair are in ORIGINAL units — the same scale as the raw uploaded data.
    Internally, every candidate row is scaled with `scaler` (if given)
    before being passed to model.predict / model.predict_proba, since the
    model itself was trained on scaled features. Skipping this step would
    silently score candidates on the wrong scale.
    """
    if not _skopt_available():
        synth_orig   = sample_uniform(X_ref, n_calls * 10, feature_bounds, random_state)
        synth_scaled = _scale_for_model(synth_orig, scaler, scaled_numeric_cols)
        if problem_type == "classification":
            scored = score_synthetic_classification(model, synth_scaled, class_names, safe_class_idx)
        else:
            scored = score_synthetic_regression(model, synth_scaled, objective)
        # Restore original-scale feature values for display (score columns are untouched)
        scored[feature_cols] = synth_orig[feature_cols].values
        sort_col = "safe_probability" if problem_type == "classification" else "objective_score"
        best     = scored.sort_values(sort_col, ascending=False).iloc[0]
        return scored, best[feature_cols].to_dict()

    from skopt import gp_minimize
    from skopt.space import Real

    # A feature is "pinned" when its bound has zero width — either the user
    # deliberately set Min = Max (e.g. fixing a new client's known oil/brine
    # properties so only the formulation is searched) or the column is
    # naturally constant in X_ref. skopt's Real dimension requires low < high
    # and raises ValueError otherwise, so pinned/constant features are held
    # out of the search space entirely and injected as constants instead.
    resolved_bounds = {
        c: feature_bounds.get(c, (float(X_ref[c].min()), float(X_ref[c].max())))
        for c in feature_cols
    }
    fixed_vals  = {c: lo for c, (lo, hi) in resolved_bounds.items() if not hi > lo}
    search_cols = [c for c in feature_cols if c not in fixed_vals]

    def _score_row(row_vals: dict) -> float:
        row_orig   = pd.DataFrame([row_vals])[feature_cols]
        row_scaled = _scale_for_model(row_orig, scaler, scaled_numeric_cols)
        row_scaled = _align_to_model(row_scaled, model)
        if problem_type == "classification":
            return float(model.predict_proba(row_scaled)[0, safe_class_idx])
        return float(model.predict(row_scaled)[0])

    if not search_cols:
        # Every feature is pinned — nothing to search; just score the one point.
        score = _score_row(fixed_vals)
        score_col = "safe_probability" if problem_type == "classification" else "predicted_value"
        return pd.DataFrame([{**fixed_vals, score_col: round(score, 4)}]), fixed_vals

    space    = [Real(*resolved_bounds[c], name=c) for c in search_cols]
    call_log: list[dict] = []

    def objective_fn(params):
        row_vals = {**dict(zip(search_cols, params)), **fixed_vals}
        if problem_type == "classification":
            prob = _score_row(row_vals)
            call_log.append({**row_vals, "safe_probability": round(prob, 4)})
            return -prob
        else:
            pred = _score_row(row_vals)
            call_log.append({**row_vals, "predicted_value": round(pred, 4)})
            return -pred if objective == "maximize" else pred

    result    = gp_minimize(objective_fn, space, n_calls=n_calls,
                            n_random_starts=n_random_starts, random_state=random_state, verbose=False)
    best_point = {**dict(zip(search_cols, result.x)), **fixed_vals}
    best_point = {c: best_point[c] for c in feature_cols}  # restore original column order
    return pd.DataFrame(call_log).round(UI_CFG.decimal_places), best_point


# ---------------------------------------------------------------------------
# Active Learning
# ---------------------------------------------------------------------------

def suggest_next_experiments(
    model,
    X_ref: pd.DataFrame,             # ORIGINAL units — for sampling bounds + display
    feature_bounds: Optional[dict],
    problem_type: str,
    class_names: list[str],
    safe_class_idx: int,
    n_suggestions: int = OPTIM_CFG.n_active_suggestions,
    n_candidates: int  = 5_000,
    method: str        = OPTIM_CFG.active_uncertainty_method,
    random_state: int  = 42,
    scaler=None,                     # fitted scaler used during training, or None
    scaled_numeric_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Suggest the next experiments to run, ranked by model uncertainty.

    X_ref and the returned suggestions are in ORIGINAL units so they read as
    real formulation recipes. Candidates are scaled with `scaler` (if given)
    only for the internal predict / predict_proba call.
    """
    candidates_orig    = sample_uniform(X_ref, n_candidates, feature_bounds, random_state)
    candidates_scaled  = _scale_for_model(candidates_orig, scaler, scaled_numeric_cols)
    candidates_aligned = _align_to_model(candidates_scaled, model)

    if problem_type == "regression":
        preds       = model.predict(candidates_aligned)
        uncertainty = np.abs(preds - preds.mean())
    else:
        proba = model.predict_proba(candidates_aligned)
        if method == "entropy":
            eps         = 1e-9
            uncertainty = -np.sum(proba * np.log(proba + eps), axis=1)
        elif method == "margin":
            s           = np.sort(proba, axis=1)[:, ::-1]
            uncertainty = 1.0 - (s[:, 0] - s[:, 1])
        else:
            uncertainty = 1.0 - proba.max(axis=1)

    candidates_orig = candidates_orig.copy()
    candidates_orig["uncertainty_score"] = np.round(uncertainty, UI_CFG.decimal_places)
    out = candidates_orig.nlargest(n_suggestions, "uncertainty_score").reset_index(drop=True)
    out.insert(0, "Suggestion #", range(1, len(out) + 1))
    return out.round(UI_CFG.decimal_places)
