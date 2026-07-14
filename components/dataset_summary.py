"""
components/dataset_summary.py
Reusable Streamlit widget: compact dataset overview card.

Usage
-----
from components.dataset_summary import render_dataset_summary
render_dataset_summary(df)
"""

import pandas as pd
import streamlit as st
from core.data.loader import extended_describe


def render_dataset_summary(df: pd.DataFrame) -> None:
    """
    Render a compact overview: shape, dtypes breakdown, missing values.
    Designed for use on any page that has a DataFrame in scope.
    """
    if df is None or df.empty:
        st.info("No dataset available.")
        return

    n_rows, n_cols = df.shape
    n_numeric = len(df.select_dtypes(include="number").columns)
    n_cat = len(df.select_dtypes(include=["object", "category"]).columns)
    n_missing = int(df.isnull().sum().sum())
    missing_pct = round(100 * n_missing / (n_rows * n_cols), 2) if n_rows * n_cols else 0

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{n_rows:,}")
        c2.metric("Columns", n_cols)
        c3.metric("Numeric", n_numeric)
        c4.metric("Categorical", n_cat)
        c5.metric("Missing", f"{missing_pct} %")

    #with st.expander("📋 Detailed Summary Table", expanded=False):
    #    st.dataframe(extended_describe(df), use_container_width=True)
