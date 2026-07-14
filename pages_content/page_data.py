"""
pages_content/page_data.py
Upload, preview, and validate the dataset.
All heavy logic lives in core/data/loader.py.
"""

import streamlit as st
from pandas.errors import EmptyDataError

from state.session import clear_state, init_state, set_state, get_value
from core.data.loader import (
    load_csv, load_excel, list_excel_sheets,
    validate_df, extended_describe,
)
from components.dataset_summary import render_dataset_summary
from config.settings import SUPPORTED_FILE_TYPES, MAX_ROWS_PREVIEW


def render():
    init_state()

    st.title("📂 Data Loading")
    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────
    st.subheader("📥 Import Dataset")
    uploaded_file = st.file_uploader(
        "**Upload dataset (CSV or Excel)**",
        type=SUPPORTED_FILE_TYPES,
        key="data_uploader",
    )

    if uploaded_file is not None:
        fname    = uploaded_file.name
        fname_lc = fname.lower()
        is_excel = fname_lc.endswith((".xls", ".xlsx"))

        sheet = None
        if is_excel:
            sheets = list_excel_sheets(uploaded_file)
            sheet = st.selectbox("📄 Select Excel sheet", sheets, key="data_sheet_select")

        # Only re-load (and wipe downstream Preprocessing/Model state) when
        # the file or sheet actually changed — otherwise every unrelated
        # rerun of this page (e.g. clicking anywhere) would silently reset
        # the whole pipeline, since the uploader still reports the same file.
        is_new = (fname != get_value("data.file_name")
                  or sheet != get_value("data.sheet_name"))

        if is_new:
            clear_state()
            try:
                if fname_lc.endswith(".csv"):
                    df = load_csv(uploaded_file)
                    set_state("data.raw", df)
                    st.success("✅ CSV loaded successfully.")

                elif is_excel:
                    df = load_excel(uploaded_file, sheet)
                    set_state("data.raw", df)
                    st.success(f"✅ Sheet **'{sheet}'** loaded successfully.")

                set_state("data.file_name", fname)
                set_state("data.sheet_name", sheet)

            except EmptyDataError:
                st.error("❌ The file is empty or has no readable columns.")
            except UnicodeDecodeError:
                st.error("❌ Could not decode the file even after trying multiple encodings. Try re-saving as UTF-8.")
            except Exception as e:
                st.error("❌ Failed to load file.")
                st.exception(e)

    # ── Always read from session state ───────────────────────────────────
    st.divider()
    df = get_value("data.raw")

    if df is None:
        st.info("No dataset loaded yet. Upload a file above.")
        st.stop()

    # File/sheet name persisted in session state — shown regardless of
    # whether the file_uploader widget still reports the upload (it can
    # come back empty after navigating to another page and back).
    file_name  = get_value("data.file_name")
    sheet_name = get_value("data.sheet_name")
    if file_name:
        label = f"📄 **{file_name}**"
        if sheet_name:
            label += f"  |  Sheet: **{sheet_name}**"
        st.caption(f"Currently loaded: {label}")

    # ── Validation warnings ───────────────────────────────────────────────
    warnings = validate_df(df)
    if warnings:
        with st.expander("⚠️ Data Quality Warnings", expanded=True):
            for w in warnings:
                st.warning(w)

    # ── Dataset summary card ──────────────────────────────────────────────
    render_dataset_summary(df)

    # ── Full preview ──────────────────────────────────────────────────────
    st.subheader("💻 Dataset Preview")
    st.dataframe(df.head(MAX_ROWS_PREVIEW), use_container_width=True)

    st.divider()

    # ── Extended describe ─────────────────────────────────────────────────
    st.subheader("📝 Statistical Summary")
    summary_df = extended_describe(df)
    st.dataframe(summary_df, use_container_width=True)

    st.info("👉 Proceed to **📊 EDA** or **⚙️ Preprocessing**.")