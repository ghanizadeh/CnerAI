import streamlit as st
import numpy as np
import pandas as pd
from itertools import product as iter_product


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide two series, replacing zero-denominators with NaN."""
    return a / b.replace(0, np.nan)


def _one_hot_interaction(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """
    Create binary interaction columns for every (cat_a, cat_b) combination.
    Returns a DataFrame of new columns only.
    """
    dummies_a = pd.get_dummies(df[col_a], prefix=col_a)
    dummies_b = pd.get_dummies(df[col_b], prefix=col_b)
    new_cols = {}
    for ca in dummies_a.columns:
        for cb in dummies_b.columns:
            name = f"{ca}_x_{cb}"
            new_cols[name] = (dummies_a[ca] * dummies_b[cb]).astype(int)
    return pd.DataFrame(new_cols, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def render_surfactant_np_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render the Surfactant-Nanoparticle Interaction feature-engineering panel.
    Supports multi-selection: all (Surfactant x NP) combinations are computed.
    """
    df = df.copy()

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    st.markdown(
        """
        <style>
        .sni-header {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #7c8fa6;
            margin-bottom: 0.25rem;
        }
        .feature-pill {
            display: inline-block;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin: 0.15rem 0.15rem 0 0;
        }
        .pill-blue  { background:#dbeafe; color:#1d4ed8; }
        .pill-green { background:#dcfce7; color:#166534; }
        .pill-amber { background:#fef9c3; color:#854d0e; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Surfactant-Nanoparticle Interaction Features")
        st.caption(
            "Select **multiple** Surfactant and NP columns — every combination will be computed. "
            "Engineers physically meaningful cross-features that capture "
            "co-adsorption, surface coverage, and type-specific synergies."
        )

        # Column selectors (multi-select)
        st.markdown('<p class="sni-header">Column Mapping</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            surf_conc_cols = st.multiselect(
                "Surfactant Concentration columns",
                options=numeric_cols,
                help="Select one or more numeric columns for surfactant concentrations. "
                     "All combinations with selected NP columns will be computed.",
            )
            surf_type_cols = st.multiselect(
                "Surfactant Type columns  (categorical interaction)",
                options=categorical_cols + numeric_cols,
                help="Select one or more columns identifying surfactant type.",
            )

        with col2:
            np_conc_cols = st.multiselect(
                "Nanoparticle Concentration columns",
                options=numeric_cols,
                help="Select one or more numeric columns for nanoparticle concentrations. "
                     "All combinations with selected Surfactant columns will be computed.",
            )
            np_type_cols = st.multiselect(
                "Nanoparticle Type columns  (categorical interaction)",
                options=categorical_cols + numeric_cols,
                help="Select one or more columns identifying nanoparticle type.",
            )

        # Combination preview
        if surf_conc_cols and np_conc_cols:
            n_combos = len(surf_conc_cols) * len(np_conc_cols)
            st.info(
                f"**{len(surf_conc_cols)} Surfactant** x **{len(np_conc_cols)} NP** columns "
                f"-> **{n_combos} combination(s)** will be generated per enabled interaction type."
            )

        st.divider()

        # Option toggles
        st.markdown('<p class="sni-header">Select Interactions to Generate</p>', unsafe_allow_html=True)

        opt_col1, opt_col2, opt_col3 = st.columns(3)

        with opt_col1:
            enable_product = st.toggle(
                "Surf x NP  (product)",
                value=False,
                help="Each surfactant conc. multiplied by each NP conc.",
            )
            st.caption("`surf_conc x np_conc`")

        with opt_col2:
            enable_ratio = st.toggle(
                "Surf / NP  (ratio)",
                value=False,
                help="Each surfactant-to-particle ratio.",
            )
            st.caption("`surf_conc / np_conc`")

        with opt_col3:
            enable_type = st.toggle(
                "Surf type x NP type  (one-hot)",
                value=False,
                help="One-hot product encoding across every surfactant-type / NP-type combination.",
            )
            st.caption("`onehot(surf_type) x onehot(np_type)`")

        # Generation logic
        st.divider()

        if not any([enable_product, enable_ratio, enable_type]):
            st.info("Enable at least one interaction above to generate features.")
            return df

        created_features: list = []
        warnings: list = []

        # 1 -- Product features (all surf x np combinations)
        if enable_product:
            if not surf_conc_cols or not np_conc_cols:
                st.warning("Surf x NP: please select at least one Surfactant AND one NP concentration column.")
            else:
                for s_col, n_col in iter_product(surf_conc_cols, np_conc_cols):
                    feat_name = f"{s_col}_x_{n_col}"
                    if feat_name in df.columns:
                        warnings.append(f"`{feat_name}` already exists — skipped.")
                    else:
                        df[feat_name] = df[s_col] * df[n_col]
                        created_features.append(feat_name)

        # 2 -- Ratio features (all surf / np combinations)
        if enable_ratio:
            if not surf_conc_cols or not np_conc_cols:
                st.warning("Surf / NP: please select at least one Surfactant AND one NP concentration column.")
            else:
                for s_col, n_col in iter_product(surf_conc_cols, np_conc_cols):
                    feat_name = f"{s_col}_div_{n_col}"
                    if feat_name in df.columns:
                        warnings.append(f"`{feat_name}` already exists — skipped.")
                    else:
                        zero_count = (df[n_col] == 0).sum()
                        df[feat_name] = _safe_divide(df[s_col], df[n_col])
                        created_features.append(feat_name)
                        if zero_count:
                            warnings.append(
                                f"{feat_name}: {zero_count} row(s) had NP conc = 0 -> filled with NaN."
                            )

        # 3 -- Categorical one-hot interactions (all surf_type x np_type combinations)
        if enable_type:
            if not surf_type_cols or not np_type_cols:
                st.warning("Surf type x NP type: please select at least one Surfactant AND one NP type column.")
            else:
                for s_col, n_col in iter_product(surf_type_cols, np_type_cols):
                    interaction_df = _one_hot_interaction(df, s_col, n_col)
                    new_type_cols = [c for c in interaction_df.columns if c not in df.columns]
                    df = pd.concat([df, interaction_df[new_type_cols]], axis=1)
                    created_features.extend(new_type_cols)
                    skipped = [c for c in interaction_df.columns if c not in new_type_cols]
                    if skipped:
                        warnings.append(
                            f"{len(skipped)} one-hot column(s) for `{s_col} x {n_col}` already existed — skipped."
                        )

        # Results summary
        if created_features:
            st.success(f"Created **{len(created_features)}** new interaction feature(s):")

            pills_html = ""
            for i, feat in enumerate(created_features):
                css = ["pill-blue", "pill-green", "pill-amber"][i % 3]
                pills_html += f'<span class="feature-pill {css}">{feat}</span>'
            st.markdown(pills_html, unsafe_allow_html=True)

            st.markdown("**Preview of new columns:**")
            st.dataframe(
                df[created_features].head(8),
                use_container_width=True,
                hide_index=False,
            )

            numeric_created = [c for c in created_features if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_created:
                with st.expander("Descriptive statistics for new features"):
                    st.dataframe(
                        df[numeric_created].describe().T.style.format("{:.4f}"),
                        use_container_width=True,
                    )

        for w in warnings:
            st.warning(w)

        if not created_features and not warnings:
            st.info("No new features were generated. Check your column selections.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Ratio feature panel (preserved from original)
# ─────────────────────────────────────────────────────────────────────────────

def render_custom_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render the custom ratio (numerator / denominator) feature panel.
    """
    df = df.copy()

    with st.container(border=True):
        st.subheader("Optional: Create Custom Ratio Features (e.g., APG/TDS)")

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

        st.markdown("Select numerators and denominators to automatically create ratio features:")

        col1, col2 = st.columns(2)

        numerators   = col1.multiselect("Select Numerator Columns",   options=numeric_columns)
        denominators = col2.multiselect("Select Denominator Columns", options=numeric_columns)

        ratio_warnings: list = []
        new_ratio_cols: dict = {}

        if numerators and denominators:
            for num in numerators:
                for den in denominators:
                    ratio_name = f"{num}/{den}"
                    if ratio_name in df.columns:
                        continue
                    df[ratio_name] = df[num] / df[den].replace(0, np.nan)
                    zero_count = (df[den] == 0).sum()
                    if zero_count > 0:
                        ratio_warnings.append(
                            f"Ratio {ratio_name}: {zero_count} rows have denominator = 0 -> filled with NaN."
                        )
                    new_ratio_cols[ratio_name] = ratio_name

            if new_ratio_cols:
                st.success(f"Created {len(new_ratio_cols)} ratio features:")
                st.write(list(new_ratio_cols.keys()))

            for warning in ratio_warnings:
                st.warning(warning)
        else:
            st.info("Select at least one numerator and one denominator to generate ratio features.")

    return df