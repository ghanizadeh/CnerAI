"""
pages_content/page_safe_region.py
===================================
🟢 Safe Region & Optimizer

Original-scale policy
---------------------
ALL feature values shown to the user (bounds, recommended ranges, top-K table,
plots) come from original (unscaled) data.

The model only ever sees scaled data internally.
We store TWO versions of the synthetic data:
  - synth_orig  → original scale (for display, ranges, plots)
  - synth_scaled → scaled (for model scoring)
We NEVER use inverse_transform on scored DataFrames because the score columns
(safe_probability, objective_score, etc.) are not feature columns and must not
be transformed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from state.session import init_state, get_value, set_state
from core.models.optimisation import (
    sample_uniform,
    sample_dirichlet_mixture,
    apply_constraints,
    score_synthetic_classification,
    score_synthetic_regression,
    filter_safe_classification,
    filter_optimal_regression,
    build_recommended_ranges,
    format_recommendation_text,
    bayesian_optimise,
    suggest_next_experiments,
)
from utils.plots_safe_region import (
    plot_safe_region_2d,
    plot_safe_region_3d,
    plot_bo_history,
)

_MIN_SYNTH      = 100
_DECIMAL_PLACES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_X_train_orig(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Return unscaled version of X_train.
    Uses saved data.X_original (pre-scaling DataFrame).
    Falls back to X_train when no scaler was applied.
    """
    X_orig_full = get_value("data.X_original")
    if X_orig_full is None:
        return X_train.copy()
    try:
        out = X_orig_full.loc[X_train.index].copy()
    except KeyError:
        out = X_orig_full.iloc[: len(X_train)].copy()
    return out.reindex(columns=X_train.columns)


def _scale(df_orig: pd.DataFrame, X_train_scaled: pd.DataFrame,
           scaler) -> pd.DataFrame:
    """
    Scale a DataFrame that is in original units → model-input units.
    Only transforms numeric columns that the scaler was fitted on.
    Returns a new DataFrame; does not mutate the input.
    """
    if scaler is None:
        return df_orig.copy()
    df_sc = df_orig.copy()
    num_cols = X_train_scaled.select_dtypes(include="number").columns.tolist()
    cols_present = [c for c in num_cols if c in df_sc.columns]
    if cols_present:
        df_sc[cols_present] = scaler.transform(df_sc[cols_present])
    return df_sc


def _attach_orig_features(scored: pd.DataFrame,
                           orig_df: pd.DataFrame,
                           feature_names: list[str]) -> pd.DataFrame:
    """
    Replace the (possibly scaled) feature columns in `scored` with the
    original-scale values from `orig_df`, keeping all score columns intact.
    Both DataFrames must have the same row order.
    """
    result = scored.copy()
    for f in feature_names:
        if f in orig_df.columns:
            result[f] = orig_df[f].values
    return result


def _render_constraint_controls(features: list[str],
                                 X_train_orig: pd.DataFrame) -> dict:
    """
    Render feature-bound inputs. X_train_orig MUST be in original units.
    Returns feature_bounds dict with (lo, hi) per feature in original units.
    """
    st.markdown("### Constraint configuration")

    with st.expander("Feature bounds (optional overrides)", expanded=False):
        st.caption(
            "Bounds are in **original units** — the same scale as your raw data. "
            "Change any value and results update automatically."
        )
        feature_bounds: dict = {}
        cols_ui = st.columns(2)
        for i, feat in enumerate(features):
            col_vals = pd.to_numeric(X_train_orig[feat], errors="coerce")
            data_min = float(col_vals.min(skipna=True))
            data_max = float(col_vals.max(skipna=True))
            with cols_ui[i % 2]:
                st.markdown(f"**{feat}**")
                st.caption(f"Data range: {data_min:.3g} – {data_max:.3g}")
                bc1, bc2 = st.columns(2)
                lo = bc1.number_input(
                    "Min", value=round(data_min, 3),
                    key=f"lo_{feat}",
                    label_visibility="collapsed",
                    format="%.3f",
                )
                hi = bc2.number_input(
                    "Max", value=round(data_max, 3),
                    key=f"hi_{feat}",
                    label_visibility="collapsed",
                    format="%.3f",
                )
                if lo > hi:
                    st.error(f"Min > Max for {feat} — values swapped.")
                    lo, hi = hi, lo
                feature_bounds[feat] = (lo, hi)

    use_mixture   = False
    mixture_cols  = None
    mixture_total = 100.0
    use_dirichlet = False

    with st.expander("Mixture / sum constraint (optional)", expanded=False):
        use_mixture = st.checkbox("Enable sum-to-constant constraint")
        if use_mixture:
            mixture_cols  = st.multiselect("Columns that must sum to constant", features)
            mixture_total = st.number_input("Target sum", value=100.0, step=1.0)
            use_dirichlet = st.checkbox(
                "Use Dirichlet sampling (ensures exact sum)", value=True,
            )

    return {
        "feature_bounds": feature_bounds,
        "use_mixture":    use_mixture,
        "mixture_cols":   mixture_cols,
        "mixture_total":  mixture_total,
        "use_dirichlet":  use_dirichlet,
    }


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    init_state()

    st.title("🟢 Safe Region & Optimizer")
    st.divider()

    # ── Guards ────────────────────────────────────────────────────────────
    model     = get_value("model.object")
    X_train   = get_value("split.X_train")   # SCALED — model input only
    y         = get_value("data.y")
    task_type = get_value("model.task_type")
    scaler    = get_value("preprocessing.scaler")

    if model is None or X_train is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Train** first.")
        st.stop()

    # Unscaled training split — used for bound defaults, sampling reference
    X_train_orig = _get_X_train_orig(X_train)

    feature_names = (
        get_value("data.processed_feature_names")
        or get_value("data.feature_names")
        or list(X_train.columns)
    )

    class_names: list[str] = []
    if task_type == "classification" and y is not None:
        class_names = [str(c) for c in sorted(y.unique())]

    random_state = 42

    # ── Target class / objective ──────────────────────────────────────────
    if task_type == "classification":
        safe_class_idx = st.selectbox(
            "Target (safe) class",
            options=list(range(len(class_names))),
            format_func=lambda i: f"{i} – {class_names[i]}",
        )
        safe_label = class_names[safe_class_idx] if class_names else "safe"
        objective  = "maximize"
    else:
        safe_class_idx = 0
        safe_label     = "optimal"
        objective      = st.radio("Objective", ["maximize", "minimize"], horizontal=True)

    # ── Scoring parameters ────────────────────────────────────────────────
    st.markdown("### Scoring parameters")
    conf_thr = st.slider(
        "Confidence threshold (classification)",
        0.50, 0.99, 0.75, 0.01,
        key="safe_conf_thr",
        disabled=(task_type != "classification"),
    )

    # ── Design space ──────────────────────────────────────────────────────
    st.markdown("### Design space exploration")
    use_synth = st.toggle("Use synthetic design space", value=True)

    n_synth = 0
    if use_synth:
        n_synth = max(_MIN_SYNTH, st.number_input(
            "Number of synthetic samples",
            value=5000, min_value=100, max_value=50000, step=500, key="n_synth",
        ))
    else:
        st.info("Synthetic sampling disabled — scoring real training data only.")

    # ── Constraints (shown in original units) ─────────────────────────────
    constraint_cfg = _render_constraint_controls(feature_names, X_train_orig)
    feature_bounds = constraint_cfg["feature_bounds"]
    mixture_cols   = constraint_cfg["mixture_cols"]
    mixture_total  = constraint_cfg["mixture_total"]
    use_dirichlet  = constraint_cfg["use_dirichlet"]
    sum_constraint = (
        (mixture_cols, mixture_total)
        if constraint_cfg["use_mixture"] and mixture_cols
        else None
    )

    # ── Build data to score ───────────────────────────────────────────────
    # We keep TWO parallel DataFrames:
    #   data_orig   — original scale (for display, ranges, bounds filtering)
    #   data_scaled — model-input scale (for scoring)
    # They have identical row order throughout.

    if use_synth:
        # Cache key includes bounds so any bound change busts the cache
        synth_key = (
            f"synth|{id(model)}|{n_synth}"
            f"|{str(sorted(feature_bounds.items()))}"
            f"|{str(mixture_cols)}|{mixture_total}|{use_dirichlet}|{random_state}"
        )
        cache_key_orig   = synth_key + "|orig"
        cache_key_scaled = synth_key + "|scaled"

        if cache_key_orig not in st.session_state:
            # Clear any stale synth caches
            stale = [k for k in list(st.session_state.keys())
                     if k.startswith("synth|")]
            for k in stale:
                del st.session_state[k]

            with st.spinner(f"Generating {n_synth:,} synthetic samples…"):
                # Sample in ORIGINAL space using original-scale bounds
                if use_dirichlet and mixture_cols:
                    base = sample_dirichlet_mixture(
                        mixture_cols, feature_bounds, mixture_total,
                        n_synth, 1.0, random_state,
                    )
                    remaining = [f for f in feature_names if f not in mixture_cols]
                    if remaining:
                        uni = sample_uniform(
                            X_train_orig[remaining], n_synth,
                            {k: v for k, v in feature_bounds.items() if k in remaining},
                            random_state,
                        )
                        for c in remaining:
                            base[c] = uni[c].values
                    gen_orig = base[feature_names]
                else:
                    gen_orig = sample_uniform(
                        X_train_orig, n_synth, feature_bounds, random_state
                    )
                    gen_orig = apply_constraints(gen_orig, feature_bounds, sum_constraint)

                # Store original-scale version
                st.session_state[cache_key_orig] = gen_orig

                # Scale for model scoring — stored separately
                gen_scaled = _scale(gen_orig, X_train, scaler)
                st.session_state[cache_key_scaled] = gen_scaled

        data_orig   = st.session_state[cache_key_orig]
        data_scaled = st.session_state[cache_key_scaled]
        source_label = "synthetic"

    else:
        # Real training data — filter by bounds in original scale
        data_orig = apply_constraints(
            X_train_orig.copy(), feature_bounds, sum_constraint
        )
        if data_orig.empty:
            st.error(
                "No real experimental rows satisfy your feature bounds (see "
                "'Feature bounds' above). Showing results anyway would "
                "silently ignore the bounds you set, so nothing is shown "
                "instead. Try widening the bounds, or turn on "
                "**'Use synthetic design space'** above to generate "
                "candidates directly within your bounds."
            )
            st.stop()

        # Scale for model
        data_scaled = _scale(data_orig, X_train, scaler)
        source_label = "experimental"

    if data_orig.empty:
        st.error("No data available to score. Check your constraints.")
        return

    # ── Score (always uses scaled data for model) ─────────────────────────
    if task_type == "classification":
        scored_raw = score_synthetic_classification(
            model, data_scaled, class_names, safe_class_idx
        )
        safe_all_raw, safe_hi_raw = filter_safe_classification(
            scored_raw, safe_class_idx, conf_thr
        )
        score_col  = "safe_probability"
        orig_labels = [str(v) for v in y.values] if y is not None else []
    else:
        scored_raw = score_synthetic_regression(model, data_scaled, objective)
        safe_hi_raw = filter_optimal_regression(scored_raw, top_pct=0.10)
        safe_all_raw = scored_raw
        score_col    = "predicted_value"
        orig_labels  = [str(round(float(v), 2)) for v in y.values] if y is not None else []

    # ── Replace scaled feature columns with original-scale values ─────────
    # Strategy: attach orig features to the FULL scored_raw first (same length
    # as data_orig, so alignment is trivial), then re-filter to get safe subsets.
    # This avoids any index alignment issue between subsets of different lengths.
    data_orig_r  = data_orig.reset_index(drop=True)
    scored_raw_r = scored_raw.reset_index(drop=True)

    # Full scored df with original-scale feature columns
    scored = _attach_orig_features(scored_raw_r, data_orig_r, feature_names)

    # Re-filter safe_all and safe_hi from `scored` (now with orig-scale features).
    # Column names match what score_synthetic_* actually produces:
    #   classification: "pred_class_idx", "safe_probability", "safety_margin"
    #   regression:     "predicted_value", "objective_score"
    if task_type == "classification":
        # safe_all = rows predicted as the target class
        if "pred_class_idx" in scored.columns:
            safe_all = scored[scored["pred_class_idx"] == safe_class_idx].copy()
        else:
            safe_all = scored.copy()
        # safe_hi = safe_all rows above confidence threshold
        if "safe_probability" in scored.columns:
            safe_hi = safe_all[safe_all["safe_probability"] >= conf_thr].copy()
        else:
            safe_hi = safe_all.copy()
    else:
        safe_all = scored.copy()
        # top 10 % by objective_score (mirrors filter_optimal_regression)
        if "objective_score" in scored.columns and len(scored) > 0:
            thresh  = scored["objective_score"].quantile(0.90)
            safe_hi = scored[scored["objective_score"] >= thresh].copy()
        else:
            safe_hi = scored.copy()

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("### Search summary")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Scored ({source_label})", f"{len(scored):,}")
    c2.metric("Predicted safe / optimal",  f"{len(safe_all):,}")
    c3.metric("High-confidence",           f"{len(safe_hi):,}")

    if safe_hi.empty:
        st.warning(
            "No high-confidence safe samples found. "
            "Try lowering the confidence threshold, widening bounds, "
            "or enabling synthetic sampling."
        )

    # ── Filter candidates by the user's bounds (shared by every section below) ─
    source_df = safe_hi if not safe_hi.empty else safe_all
    filtered_source = source_df.copy()
    for feat, (lo, hi) in feature_bounds.items():
        if feat in filtered_source.columns:
            col_num = pd.to_numeric(filtered_source[feat], errors="coerce")
            filtered_source = filtered_source[(col_num >= lo) & (col_num <= hi)]

    if filtered_source.empty:
        st.error(
            f"None of the **{len(source_df):,}** candidates predicted "
            f"'{safe_label}' satisfy your feature bounds (see 'Feature "
            "bounds' above). The sections below are left empty rather than "
            "silently showing results that ignore your bounds. Try "
            "widening the bounds, increasing the number of synthetic "
            "samples, or lowering the confidence threshold."
        )
        # Deliberately NOT falling back to unfiltered data here — showing
        # out-of-bounds results as if they respected the bounds is worse
        # than showing nothing (this was the original bug).

    sort_by = (
        ["safe_probability", "safety_margin"]
        if task_type == "classification"
        else ["objective_score"]
    )
    ranked_source = filtered_source.sort_values(sort_by, ascending=False)

    st.divider()

    # ── Bayesian Optimisation ─────────────────────────────────────────────
    # Runs first (if enabled) so its result — the single best formulation
    # the model can find across the whole continuous design space — can be
    # promoted into the "Optimized Formulation" headline below, instead of
    # only ever showing the best among the sampled/scored candidates.
    st.markdown("### 🎯 Bayesian Optimisation")
    run_bo = st.checkbox("Run Bayesian Optimisation", value=False, key="run_bo")
    bo_results, bo_best = None, None

    if run_bo:
        n_bo_calls = st.number_input(
            "BO evaluations", value=50, min_value=10, max_value=300,
            step=10, key="n_bo_calls",
        )
        bo_key = f"bo_{id(model)}_{n_bo_calls}_{random_state}"
        if bo_key not in st.session_state:
            with st.spinner(f"Running {n_bo_calls} Bayesian evaluations…"):
                bo_results, bo_best = bayesian_optimise(
                    model=model,
                    feature_cols=feature_names,
                    X_ref=X_train_orig,   # reference in original scale
                    feature_bounds=feature_bounds,
                    safe_class_idx=safe_class_idx,
                    class_names=class_names,
                    problem_type=task_type,
                    objective=objective,
                    n_calls=n_bo_calls,
                    random_state=random_state,
                    scaler=scaler,
                    scaled_numeric_cols=X_train.select_dtypes(include="number").columns.tolist(),
                )
            st.session_state[bo_key] = (bo_results, bo_best)
        bo_results, bo_best = st.session_state[bo_key]

        st.success("Bayesian optimisation complete — see the Optimized Formulation below.")
        if bo_results is not None and score_col in bo_results.columns:
            st.plotly_chart(plot_bo_history(bo_results, score_col), use_container_width=True)
    else:
        st.caption(
            "Optional: searches the full continuous design space (not just the "
            "sampled candidates below) for the single best formulation. "
            "Slower, but the most rigorous single-point answer."
        )

    st.divider()

    # ── Optimized Formulation (headline result) ────────────────────────────
    st.markdown("### 🏆 Optimized Formulation")
    if bo_best is not None:
        st.success(
            "Best formulation found by **Bayesian Optimisation** — searched across "
            "the full continuous design space, not just the sampled candidates."
        )
        st.json({k: round(v, _DECIMAL_PLACES) for k, v in bo_best.items()})
    elif not ranked_source.empty:
        best_row = ranked_source.iloc[0]
        st.info(
            f"Best formulation among the **{len(scored):,}** scored candidates "
            f"({source_label}). Run **Bayesian Optimisation** above for a more "
            "rigorous continuous-space search."
        )
        best_display = {f: round(float(best_row[f]), _DECIMAL_PLACES)
                         for f in feature_names if f in best_row.index}
        st.json(best_display)
        score_label = "Predicted safe probability" if task_type == "classification" else "Predicted value"
        st.caption(f"{score_label}: **{best_row.get(score_col, float('nan')):.3f}**")
    else:
        st.info("No candidates available yet — widen bounds or enable synthetic sampling.")

    st.divider()

    # ── Recommended ranges (secondary: the neighbourhood around the optimum) ─
    st.markdown("### Safe neighbourhood (recommended ranges)")
    st.caption(
        "This is **not** a single formulation — it is the spread of values seen "
        "across all high-confidence candidates, one feature at a time. Use the "
        "**Optimized Formulation** above as your primary recommendation; use "
        "this table to see how much flexibility you have around it."
    )
    ranges_df = build_recommended_ranges(filtered_source, feature_names)
    if not ranges_df.empty:
        st.dataframe(ranges_df.round(_DECIMAL_PLACES), use_container_width=True)
        st.code(format_recommendation_text(ranges_df, safe_label))
    else:
        st.info("Not enough data to compute recommended ranges.")
        ranges_df = pd.DataFrame()

    # ── Top candidates ────────────────────────────────────────────────────
    top_k = st.slider("Top-K formulations to display", 5, 100, 20, key="safe_top_k")
    st.markdown(f"### Top {top_k} recommended formulations")
    if ranked_source.empty:
        st.info("No candidates satisfy your bounds — see the message above.")
        top_df = ranked_source
    else:
        top_df = ranked_source.head(top_k).copy()
        top_df.insert(0, "Rank", range(1, len(top_df) + 1))
        st.dataframe(top_df.round(_DECIMAL_PLACES), use_container_width=True)

    # ── 2D region map ─────────────────────────────────────────────────────
    _X_orig_val = get_value("data.X_original")
    X_orig_display = X_train_orig if _X_orig_val is None else _X_orig_val
    if len(feature_names) >= 2:
        st.markdown("### 2D region map")
        r1, r2 = st.columns(2)
        x2 = r1.selectbox("X axis", feature_names, key="map2_x")
        y2 = r2.selectbox("Y axis", [f for f in feature_names if f != x2], key="map2_y")
        color_col = "pred_class" if task_type == "classification" else "predicted_value"
        try:
            fig = plot_safe_region_2d(
                scored, X_orig_display, orig_labels, x2, y2,
                color_col=color_col,
                class_names=class_names if task_type == "classification" else None,
                sample_n=n_synth if use_synth else len(scored),
                random_state=random_state,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"2D map failed: {exc}")

    # ── 3D region map ─────────────────────────────────────────────────────
    if len(feature_names) >= 3:
        st.markdown("### 3D region map")
        r1, r2, r3 = st.columns(3)
        x3 = r1.selectbox("X", feature_names, key="map3_x")
        y3 = r2.selectbox("Y", [f for f in feature_names if f != x3], key="map3_y")
        z3 = r3.selectbox("Z", [f for f in feature_names if f not in [x3, y3]], key="map3_z")
        try:
            fig = plot_safe_region_3d(
                scored, x3, y3, z3,
                color_col=color_col,
                class_names=class_names if task_type == "classification" else None,
                sample_n=n_synth if use_synth else len(scored),
                random_state=random_state,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"3D map failed: {exc}")

    st.divider()

    # ── Active Learning ───────────────────────────────────────────────────
    st.markdown("### 🔬 Active Learning — Next Experiment Suggestions")
    run_al = st.checkbox("Run Active Learning", value=False, key="run_al")

    if run_al:
        n_al = st.number_input(
            "Suggestions", value=5, min_value=1, max_value=50, step=1, key="n_al"
        )
        al_method = st.selectbox(
            "Uncertainty method",
            ["entropy", "margin", "least_confident"],
            key="al_method",
        )
        al_key = f"al_{id(model)}_{n_al}_{al_method}_{random_state}"
        if al_key not in st.session_state:
            with st.spinner("Identifying most informative experiment candidates…"):
                al_suggestions = suggest_next_experiments(
                    model=model,
                    X_ref=X_train_orig,     # original units — suggestions read as real recipes
                    feature_bounds=feature_bounds or None,
                    problem_type=task_type,
                    class_names=class_names,
                    safe_class_idx=safe_class_idx,
                    n_suggestions=n_al,
                    method=al_method,
                    random_state=random_state,
                    scaler=scaler,
                    scaled_numeric_cols=X_train.select_dtypes(include="number").columns.tolist(),
                )
            st.session_state[al_key] = al_suggestions
        al_suggestions = st.session_state[al_key]

        st.info(
            "These formulations have the **highest model uncertainty** — "
            "running these experiments will most efficiently improve accuracy."
        )
        st.dataframe(al_suggestions.round(_DECIMAL_PLACES), use_container_width=True)

    # ── Industrial interpretation ─────────────────────────────────────────
    st.divider()
    st.markdown("### Industrial interpretation")
    if not ranges_df.empty:
        cols_r   = ranges_df.columns.tolist()
        low_col  = next((c for c in cols_r if "Low"  in c), cols_r[1])
        high_col = next((c for c in cols_r if "High" in c), cols_r[3])
        insights = [
            f"- **{row['Feature']}**: "
            f"{row[low_col]:.3g} – {row[high_col]:.3g}  "
            f"(median ≈ {row['Median']:.3g})"
            for _, row in ranges_df.iterrows()
        ]
        mode_note = (
            "real experimental data" if not use_synth
            else f"{len(scored):,} synthetic samples"
        )
        st.markdown(
            f"For **'{safe_label}'**, use the **Optimized Formulation** above as the "
            f"single best point. This is the surrounding safe neighbourhood, for "
            f"flexibility around it:\n\n"
            + "\n".join(insights)
            + f"\n\nRanges from **5th–95th percentile** of high-confidence safe region "
              f"({mode_note}), filtered to your specified bounds."
        )
    else:
        st.info(
            "No interpretation available — no candidates satisfied your "
            "feature bounds. See the message above 'Optimized Formulation'."
        )