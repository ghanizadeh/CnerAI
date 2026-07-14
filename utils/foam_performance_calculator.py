"""
performance_index_calculator.py
--------------------------------
Streamlit page: Foam Performance Calculator.

This file is responsible only for UI concerns:
  - file upload widgets
  - user option controls
  - triggering the pipeline on button click
  - delegating all rendering to ui_helpers

All business logic lives in ``foam_performance_calc_utility``.
"""

import streamlit as st

from foam_performance_calc import PipelineConfig, run_pipeline
from foam_performance_calc.constants import (
    DAY_WEIGHT_COLUMNS,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_R2_THRESHOLD,
    MAX_NUM_DAYS,
    MIN_NUM_DAYS,
    MODEL_CHOICES,
)
from foam_performance_calc.ui_helpers import render_pipeline_results


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def show_performance_index_calculator() -> None:
    """
    Render the Foam Performance Calculator page.

    Call this function from the parent Streamlit app's routing logic, e.g.:

        if menu_selection == "Performance Calculator":
            show_performance_index_calculator()
    """
    st.title("🧪 Foam Performance Calculator")
    st.write("---")

    _render_about_expander()

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------
    with st.container(border=True):
        foam_file = st.file_uploader("📑 Upload **Foam Data** (CSV)", type=["csv"])

    with st.container(border=True):
        texture_file = st.file_uploader(
            "📑 Upload **Texture Weights** (CSV)", type=["csv"]
        )

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("Options")
        config = _render_options()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if st.button("Run", type="primary"):
        if not foam_file or not texture_file:
            st.info(
                "Please upload both a **Foam Data** CSV and a "
                "**Texture Weights** CSV before running."
            )
            return

        with st.spinner("Running pipeline…"):
            try:
                result = run_pipeline(
                    foam_source=foam_file,
                    texture_weights_source=texture_file,
                    config=config,
                )
            except ValueError as exc:
                st.error(f"Pipeline error: {exc}")
                return

        st.write("---")
        render_pipeline_results(result)


# ---------------------------------------------------------------------------
# Private UI helpers (local to this page only)
# ---------------------------------------------------------------------------

def _render_options() -> PipelineConfig:
    """
    Render the options section and return a populated ``PipelineConfig``.
    """
    num_days: int = st.slider(
        "📅 **Last Day (inclusive)** (0–N)",
        min_value=MIN_NUM_DAYS,
        max_value=MAX_NUM_DAYS,
        value=MAX_NUM_DAYS,
        help="If N=14, days 0 through 14 (15 days total) will be used.",
    )

    day_weights: dict[int, float] = {}
    with st.expander("⚖️ **Day Weights**", expanded=False):
        st.info("Assign a custom weight for each day from 0 to N (default = 1.0).")
        cols = st.columns(DAY_WEIGHT_COLUMNS)
        for d in range(num_days + 1):
            with cols[d % DAY_WEIGHT_COLUMNS]:
                day_weights[d] = st.number_input(
                    f"Day {d}",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"day_weight_{d}",
                )

    with st.expander("🔧 Missing Foam Volume Filling", expanded=True):
        selected_model: str = st.selectbox(
            "Foam Volume Imputation Model (row-wise):",
            options=MODEL_CHOICES,
            index=0,
            help=(
                "`best` selects the highest R² model per sample. "
                "Other options force a single model for all samples."
            ),
        )

    return PipelineConfig(
        num_days=num_days,
        selected_model=selected_model,
        r2_threshold=DEFAULT_R2_THRESHOLD,
        day_weights=day_weights,
    )


def _render_about_expander() -> None:
    """Render the collapsible 'About' section."""
    with st.expander("ℹ️ About This App", expanded=False):
        st.markdown(
            """
### 📂 Upload Input Files
- **Foam Data (CSV):** daily foam volumes and textures.
  Column headers may be in the legacy deduplicated format
  (`Foam (cc)`, `Foam (cc).1`, `Texture`, `Texture.1`, …)
  or the pre-labelled format (`Day 0 - Foam (cc)`, `Day 1 - Foam Texture`, …).
- **Texture Weights (CSV):** must contain columns:
  - `Normalized_Texture` – texture label
  - `Weight` – numeric weight

---

### ⚙️ Workflow
1. **Filter valid samples** — based on first/last day completeness and no
   two consecutive missing days.
2. **Impute missing foam volumes** — row-wise regression (linear / exp /
   polynomial / random forest).
3. **Impute missing textures** — average of the nearest previous and next
   valid texture weights.
4. **Rename columns** to `Day d - Foam (cc)` / `Day d - Foam Texture`.
5. **Compute scores:**
   - `Score_Volume`
   - `Score_Texture`
   - `Performance_Index = Score_Volume + Score_Texture`
6. **Download** the full original dataset with scores merged in (invalid
   rows receive NaN for score columns).

---
"""
        )
