import streamlit as st
import pandas as pd
import numpy as np
import re

from core.data.loader import load_csv

# ==============================
# Core Function (preserves row order + normalization preview)
# ==============================
def update_dilution_stability_14_30(df: pd.DataFrame):
    df = df.copy()
    df["shelf_life_days"] = np.nan
    # Identify Observation columns (case-insensitive)
    obs_cols = [c for c in df.columns if "observation" in c.lower()]

    # If no observation columns found, stop here
    if not obs_cols:
        return None, None, "⚠️ No columns containing 'Observation' were found in your dataset."

    # Ensure stability columns exist (case-insensitive check)
    col_map = {c.lower(): c for c in df.columns}
    if "dilution stability (14 days)" not in col_map:
        df["Dilution Stability (14 Days)"] = np.nan
        col_map["dilution stability (14 days)"] = "Dilution Stability (14 Days)"
    if "dilution stability (30 days)" not in col_map:
        df["Dilution Stability (30 Days)"] = np.nan
        col_map["dilution stability (30 days)"] = "Dilution Stability (30 Days)"

    # Regex patterns for instability
    instab_obs_pattern = re.compile(
        r"(?<!no\s)\b(?:precip\w*|gel|unstable)\b",
        re.IGNORECASE
    )
    instab_desc_pattern = re.compile(r"(precipitate dilution|gel dilution|unstable dilution)", re.IGNORECASE)

    # ✅ Normalization preview
    preview_data = []

    for c in obs_cols:
        for val in df[c].dropna().unique():
            val_str = str(val).strip()
            norm_val = re.sub(r"[\.\_\-\s]+", " ", val_str).strip().lower()
            preview_data.append({
                "Observation Column": c,
                "Original Text": val_str,
                "Normalized Text": norm_val
            })

        # Clean & normalize the actual data (keep column names intact)
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"[\.\_\-\s]+", " ", regex=True)
            .str.strip()
            .str.lower()
        )
        df[c] = df[c].replace(["", "nan"], np.nan)

    preview_df = pd.DataFrame(preview_data).drop_duplicates().reset_index(drop=True)

    # ----- Row-wise stability logic -----
    for i in df.index:
        row = df.loc[i]

        # --- Sort observation columns by day ---
        obs_with_days = []
        for c in obs_cols:
            match = re.search(r"\d+", c)
            if match:
                obs_with_days.append((int(match.group()), c))

        obs_with_days = sorted(obs_with_days, key=lambda x: x[0])

        first_failure_day = None
        available_days = []

        for day, col in obs_with_days:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                available_days.append(day)

                if first_failure_day is None:
                    if instab_obs_pattern.search(str(val)):
                        first_failure_day = day

        max_day = max(available_days) if available_days else 0

        # Save shelf_life_days
        df.loc[i, "shelf_life_days"] = first_failure_day
 
        # --- 14 Days ---
        # Determine instability by threshold
        instab_14 = False
        instab_30 = False

        for day, col in obs_with_days:
            val = row[col]
            if pd.notna(val):
                if instab_obs_pattern.search(str(val)):
                    if day <= 14:
                        instab_14 = True
                    if day <= 30:
                        instab_30 = True

        # --- 14 Days ---
        if max_day < 14:
            df.loc[i, col_map["dilution stability (14 days)"]] = (
                False if instab_14 else np.nan
            )
        else:
            df.loc[i, col_map["dilution stability (14 days)"]] = (
                False if instab_14 else True
            )

        # --- 30 Days ---
        if max_day < 30:
            df.loc[i, col_map["dilution stability (30 days)"]] = (
                False if instab_30 else np.nan
            )
        else:
            df.loc[i, col_map["dilution stability (30 days)"]] = (
                False if instab_30 else True
            )
    return df, preview_df, None

# ==============================
# Streamlit UI
# ==============================
def render():
    st.title("🧫 Foam Stability (14/30 Days)")
    st.write("---")

    with st.expander("❓ Help & Instructions", expanded=False):
        st.markdown("""
        This tool updates the **Dilution Stability (14 Days)** and **(30 Days)** columns automatically  
        — without changing the order of rows in your dataset.

        ### ⚙️ Data Requirements
        - Your dataset **must include columns** containing the word **“Observation”**  
          (e.g., `Day 1 - Observation`, `Observation 7`, etc.).
        - If no such columns exist, the process will stop and display an info message.

        ### 🔍 How It Works
        - Cleans each *Observation* cell by removing `.`, `_`, `-`, and extra spaces, converting text to lowercase.  
        - Example:  
          `Unstable-Gel Formation` → `unstable gel formation`  
          `precipitate.` → `precipitate`
        - Determines the latest observation day with valid text and scans for instability keywords:
            - **False** if text contains *precipitate*, *gel*, or *unstable*.
            - **NaN (Unknown)** if all are clear but monitoring stopped before 14 or 30 days.
            - **True** if stable after the full observation period.
        - If the columns `Dilution Stability (14 Days)` and `Dilution Stability (30 Days)` already exist,
                              they will be **updated** instead of recreated.
        """)

    st.markdown("<hr style='border:1px solid #1E90FF;'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload dataset (csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = load_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ File loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head())

            if st.button("▶️ Run Dilution Stability Calculation"):
                updated_df, preview_df, warning_msg = update_dilution_stability_14_30(df)

                if warning_msg:
                    st.warning(warning_msg)
                    return

                updated_df = updated_df.loc[df.index]

                st.success("✅ Dilution stability columns updated successfully (row order preserved).")

                st.write("### 🧩 Text Normalization Preview")
                st.dataframe(preview_df.head(20))

                st.write("### 🧪 Updated Data Preview")
                st.dataframe(updated_df.head())

                # --- Download Buttons ---
                # utf-8-sig (BOM) so Excel renders accented/non-breaking-space
                # characters correctly instead of as "Â" mojibake.
                csv = updated_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="💾 Download Updated CSV",
                    data=csv,
                    file_name="updated_dilution_stability.csv",
                    mime="text/csv"
                )

                norm_csv = preview_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📄 Download Normalization Preview CSV",
                    data=norm_csv,
                    file_name="normalized_observation_texts.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
    else:
        st.info("👆 Please upload a CSV or Excel file to begin.")
 
