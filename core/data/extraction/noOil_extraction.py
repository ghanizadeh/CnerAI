import numpy as np
import pandas as pd
import streamlit as st

from core.data.extraction import noOil_extraction_helper as utl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLS_TO_DROP_MULTI = ["Initial Foam Volume (cc)", "Pilot", "Water (cc)", "Tube Volume (mL)"]
COLS_TO_DROP_SINGLE = COLS_TO_DROP_MULTI + ["Day", "Foam (cc)", "Foam Texture"]
DILUTION_APPLY_COLS = [
    "Pilot", "Temp Foam Monitoring", "Initial Foam Volume (cc)",
    "Dilution Ratio", "Concentrate manufacturing method (Ratio)",
    "Sonicated", "Sample Description", "Date",
]


def render():
    st.title("🧪 Foam Sample Data Extraction (No Oil)")

    _render_help()
    st.divider()

    with st.container(border=True):
        st.markdown("##### 📂 Import Dataset")
        uploaded_file = st.file_uploader("📤 Upload Data (CSV)", type=["csv"])

    if not uploaded_file:
        return

    st.success("Parsing complete!")

    df_multiple, df_single_final, rows_without_dilution = _process_file(uploaded_file)

    st.subheader("📋 Preview: Extracted Samples (without Oil)")
    st.dataframe(df_single_final)
    st.success(f"Total number of extracted samples: {len(df_single_final)}")

    df_multiple_final = df_multiple.drop(columns=COLS_TO_DROP_MULTI, errors="ignore")

    _render_download_buttons(df_multiple_final, df_single_final)
    st.divider()
    _render_search(df_multiple_final, df_single_final)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process_file(uploaded_file):
    # Try UTF-8 first so genuinely UTF-8 files aren't silently mis-decoded
    # (cp1252 alone accepts almost any byte sequence without raising, so a
    # cp1252-only read would produce garbled text instead of an error).
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            uploaded_file.seek(0)
            df_input = pd.read_csv(uploaded_file, header=None, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        uploaded_file.seek(0)
        df_input = pd.read_csv(uploaded_file, header=None, encoding="latin-1")
    samples, formulations = utl.extract_samples_complete_fixed(df_input)

    df_samples = pd.DataFrame(samples)
    df_formulations = pd.DataFrame.from_dict(formulations, orient="index")
    df_formulations["SampleID"] = df_formulations.index

    df_multiple = df_samples.merge(df_formulations, on="SampleID", how="left")

    # Add columns with defaults
    df_multiple["Initial Foam Volume (cc)"] = utl.DEFAULT_INITIAL_FOAM
    df_multiple["Pilot"] = np.nan
    df_multiple["Temp Foam Monitoring"] = np.nan
    df_multiple["Initial Foam Temp (dilution Temp)"] = np.nan
    df_multiple["Water (cc)"] = np.nan
    df_multiple["Sonicated"] = np.nan
    df_multiple["Brine Type"] = utl.DEFAULT_BRINE_TYPE
    df_multiple["Sample Description"] = pd.NA

    # Rows without dilution data — handled separately
    rows_without_dilution = df_multiple[df_multiple["Dilution Ratio"].isna()].copy()

    # Apply dilution parsing
    df_multiple[DILUTION_APPLY_COLS] = df_multiple.apply(
        lambda row: pd.Series(utl.process_dilution(row["Dilution Ratio"], row["Date"])),
        axis=1,
    )

    df_multiple["Date"] = df_multiple["Date"].fillna(utl.FALLBACK_DATE)
    df_multiple["Dilution Ratio"] = df_multiple["Dilution Ratio"].fillna(utl.FALLBACK_DILUTION)
    df_multiple = utl.make_sampleid_unique(df_multiple)
    df_multiple["Tube Volume (mL)"] = (
        df_multiple["Tube Volume (mL)"]
        .astype(str)
        .str.replace(r"mL\s*tube", "", case=False, regex=True)
        .str.strip()
    )

    df_single = utl.assign_pilot_column(df_multiple)
    df_single = df_single.replace({None: np.nan}).infer_objects(copy=False)
    df_single_final = pd.DataFrame(utl.clean_dilution(df_single))

    # Merge in rows that had no dilution data
    df_single_final = pd.concat([df_single_final, rows_without_dilution], ignore_index=True)

    # Drop unneeded columns
    df_single_final = df_single_final.drop(columns=COLS_TO_DROP_SINGLE, errors="ignore")
    df_single_final = utl.clean_dilution_ratio(df_single_final)
    df_single_final["Sample Description"] = (
        df_single_final["Sample Description"]
        .str.replace(r"\b\d{2,3}X\b", "", regex=True)
        .str.strip()
    )

    df_single_final = utl.sort_columns_custom(df_single_final)

    # Replace "-1" suffix with "- New Trial" annotation
    mask = df_single_final["SampleID"].astype(str).str.contains("-1")
    df_single_final.loc[mask, "SampleID"] = (
        df_single_final.loc[mask, "SampleID"].str.replace("-1", "", regex=False).str.strip()
    )
    df_single_final.loc[mask, "Sample Description"] = (
        df_single_final.loc[mask, "Sample Description"].astype(str) + " - New Trial"
    )

    return df_multiple, df_single_final, rows_without_dilution


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_help():
    with st.expander("🛟 Help & Instructions", expanded=False):

        st.markdown("""
        ### 📘 Overview
        This module parses and processes foam sample data **without oil** from CSV files.
        It extracts key parameters, computes dilution ratios, and produces a cleaned,
        analysis-ready dataset.

        ---
        ### 🧭 Step-by-Step Guide
        1. **Upload your file** — click **"Browse files"** to select your CSV.
        2. **Automatic processing** — extraction starts immediately after upload.
        3. **Review results** — the parsed dataset is shown directly in the app.
        4. **Download outputs** — use the download buttons to save the processed files.
        5. **Search** — use the SampleID search to locate specific samples quickly.

        ---
        ### 📄 Expected File Structure
        """)

        st.image(
            "assets/noOil_extraction.png",
            caption="Example of expected file structure",
            use_container_width=True
        )

        st.markdown("""
        **Supported dilution metadata** (any order, any column after col A):
        - Pilot: `AFC`
        - Temperature: `45C` (degrees) or `RT` (room temp)
        - Sonication: `Sonicated` / `Not Sonicated`
        - Tube volume: e.g. `10mL tube`
        - Mix ratio: e.g. `(2:1) ratio`
        - Foam volume: e.g. `5cc`

        **Stability notes** (col B onward on the sample row):
        - `Unstable concentrate at RT` / `Stable concentrate at RT`
        - `Unstable concentrate at 8C` / `Stable concentrate at 8C`
        - `Unstable concentrate at 4C` / `Stable concentrate at 4C`
        - `Unstable dilution`

        **Baseline marker:** place a `*` in any cell after the Foam Texture column on a Day row
        to mark that observation as a baseline measurement.

        ---
        💡 *Tip:* Ensure your CSV uses consistent column names and units for accurate parsing.
        """)


def _render_download_buttons(df_multiple_final, df_single_final):
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "💾 Download Multi Row Samples.csv",
            df_multiple_final.to_csv(index=False).encode("utf-8-sig"),
            file_name="1_Parser_No_Oil_No_Process.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "💾 Download Single Row Samples.csv",
            df_single_final.to_csv(index=False).encode("utf-8-sig"),
            file_name="2_Parser_No_Oil_Single.csv",
            mime="text/csv",
        )


def _render_search(df_multiple_final, df_single_final):
    st.subheader("Search by SampleID")
    search_id = st.text_input("Enter SampleID:")
    search_type = st.radio("Search type", ["Exact Match", "Contains"])
    view_option = st.radio("Choose view mode:", ["Multi Row Samples", "Single Row Samples"])

    if not search_id:
        return

    if search_type == "Exact Match":
        result_multi = df_multiple_final[df_multiple_final["SampleID"].str.lower() == search_id.lower()]
        result_single = df_single_final[df_single_final["SampleID"].str.lower() == search_id.lower()]
    else:
        result_multi = df_multiple_final[
            df_multiple_final["SampleID"].str.lower().str.contains(search_id.lower(), na=False)
        ]
        result_single = df_single_final[
            df_single_final["SampleID"].str.lower().str.contains(search_id.lower(), na=False)
        ]

    if view_option == "Multi Row Samples":
        st.markdown("### Multi Row Samples")
        if result_multi.empty:
            st.warning("No Multi Row Sample found for this SampleID.")
        else:
            st.success(f"Found {len(result_multi)} row(s) in Multi Row Samples.")
            st.dataframe(result_multi, use_container_width=True)
    else:
        st.markdown("### Single Row Samples")
        if result_single.empty:
            st.warning("No Single Row Sample found for this SampleID.")
        else:
            st.success("Found matching sample in Single Row Samples.")
            st.dataframe(result_single, use_container_width=True)
