import re
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_FOAM = "5cc"
DEFAULT_BRINE_TYPE = "Field Brine"
FALLBACK_DATE = "No Date"
FALLBACK_DILUTION = "No Dilution Info"
NO_DILUTION_LABEL = "Only Formulation - No Dilution Data"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_samples_complete_fixed(df):
    samples = []
    formulations = {}
    row = 0
    last_formulation = None
    last_dilution = None
    last_tube_volume = None
    column_map = {}
    current_dilution_rows = []

    concentrate_has_RT = True
    concentrate_has_8c = np.nan
    concentrate_has_4c = np.nan
    concentrate_has_HT = np.nan
    dilution_stability = True

    def flush_current_dilution():
        for row_data in current_dilution_rows:
            row_data["Concentrate Stability (8C)"] = concentrate_has_8c
            row_data["Concentrate Stability (4C)"] = concentrate_has_4c
            row_data["Concentrate Stability (RT)"] = concentrate_has_RT
            row_data["Concentrate Stability (HT)"] = concentrate_has_HT
            row_data["Dilution Stability"] = dilution_stability
            row_data["Tube Volume (mL)"] = last_tube_volume
            samples.append(row_data)
        current_dilution_rows.clear()

    def parse_formulation(text):
        data = {
            "SampleID": re.search(r"\((.*?)\)", text).group(1).strip()
            if re.search(r"\((.*?)\)", text) else None
        }
        seen_keys_lower = set()
        for part in re.split(r"[,-]", text):
            part = part.strip()
            ppm_match = re.match(r"(\d+\.?\d*)\s*ppm\s*(.*)", part, re.IGNORECASE)
            if ppm_match:
                val, chem = ppm_match.groups()
                chem = re.sub(r"\(.*?\)", "", chem).strip()
                key = f"{chem} (ppm)"
                if key.lower() not in seen_keys_lower:
                    data[key] = float(val)
                    seen_keys_lower.add(key.lower())
                continue
            pct_match = re.match(r"(\d+\.?\d*)%\s*(.*)", part, re.IGNORECASE)
            if pct_match:
                val, chem = pct_match.groups()
                chem = re.sub(r"\(.*?\)", "", chem).strip()
                key = f"{chem} (%)"
                if key.lower() not in seen_keys_lower:
                    data[key] = float(val)
                    seen_keys_lower.add(key.lower())
        return data

    def _parse_stability_flags(row_idx):
        """Return (s8, s4, RT, HT, ds) stability flags from the formulation row."""
        formulation_text = str(df.iat[row_idx, 0]).strip().lower()
        s8, s4, HT, ds = np.nan, np.nan, np.nan, True
        RT = False if "unstable concentrate" in formulation_text else True

        side_values = [
            str(df.iat[row_idx, col]).lower().strip()
            for col in range(1, min(11, df.shape[1]))
            if pd.notna(df.iat[row_idx, col])
        ]

        for val in side_values:
            is_unstable = "unstable" in val or "cloudy" in val
            is_concentrate = "concentrate" in val

            if is_unstable and is_concentrate and not re.search(r"\b\d{1,2}C\b", val, re.IGNORECASE):
                RT = False
            if is_unstable and is_concentrate and re.search(r"8\s*[cC]", val):
                s8 = False
            if is_unstable and is_concentrate and re.search(r"4\s*[cC]", val):
                s4 = False
            if (is_unstable or "cloudy" in val) and "dilution" in val:
                ds = False

            if re.search(r"\bunstable\b", val, re.IGNORECASE) and re.search(r"4\s*[cC]\b", val, re.IGNORECASE):
                s4 = False
            elif re.search(r"\bstable\b", val, re.IGNORECASE) and re.search(r"4\s*[cC]\b", val, re.IGNORECASE):
                if s4 is not False:
                    s4 = True

            if re.search(r"\bunstable\b", val, re.IGNORECASE) and re.search(r"8\s*[cC]\b", val, re.IGNORECASE):
                s8 = False
            elif re.search(r"\bstable\b", val, re.IGNORECASE) and re.search(r"8\s*[cC]\b", val, re.IGNORECASE):
                if s8 is not False:
                    s8 = True

            if re.search(r"\bunstable\b", val, re.IGNORECASE) and re.search(r"\bRT\b", val, re.IGNORECASE):
                RT = False
            elif re.search(r"\bstable\b", val, re.IGNORECASE) and re.search(r"\bRT\b", val, re.IGNORECASE):
                if RT is not False:
                    RT = True

        return s8, s4, RT, HT, ds

    while row < df.shape[0]:
        cell = str(df.iat[row, 0]).strip()

        # --- Formulation row ---
        if any(sym in cell.lower() for sym in ["%", "ppm"]) and "(" in cell:
            flush_current_dilution()

            s8, s4, RT, HT, ds = _parse_stability_flags(row)
            concentrate_has_8c = s8
            concentrate_has_4c = s4
            concentrate_has_RT = RT
            concentrate_has_HT = HT
            dilution_stability = ds

            last_formulation = parse_formulation(cell)
            if not last_formulation.get("SampleID"):
                last_formulation["SampleID"] = f"Sample_{len(formulations) + 1}"
            formulations[last_formulation["SampleID"]] = last_formulation

            next_row_text = (
                ",".join(df.iloc[row + 1].fillna("").astype(str)).lower()
                if row + 1 < df.shape[0] else ""
            )
            if not re.search(r"\d+\s*x", next_row_text):
                samples.append({
                    "SampleID": last_formulation["SampleID"],
                    "Dilution Ratio": None,
                    "Day": None,
                    "Foam (cc)": None,
                    "Foam Texture": None,
                    "Water (cc)": None,
                    "Zeta": None,
                    "Conductivity": None,
                    "Size": None,
                    "PI": None,
                    "Baseline": None,
                    "Date": None,
                    "Concentrate Stability (8C)": s8,
                    "Concentrate Stability (4C)": s4,
                    "Concentrate Stability (RT)": RT,
                    "Concentrate Stability (HT)": HT,
                    "Tube Volume (mL)": None,
                })

            row += 1
            continue

        # --- Day row ---
        if re.match(r"Day\s*\d+", cell, re.IGNORECASE):
            row_data = {"SampleID": last_formulation["SampleID"]} if last_formulation else {}
            row_data["Dilution Ratio"] = last_dilution
            row_data["Day"] = cell.strip()

            stars = [
                "*"
                for i in range(column_map.get("Foam Texture", 0) + 1, df.shape[1])
                if "*" in str(df.iat[row, i])
            ]
            row_data["Baseline"] = ", ".join(stars) if stars else None

            for label in ["Date", "Foam (cc)", "Foam Texture", "Water (cc)", "Zeta", "Conductivity", "Size", "PI"]:
                col_idx = column_map.get(label)
                if label == "Date" and col_idx is None:
                    col_idx = 1
                val = (
                    str(df.iat[row, col_idx]).strip()
                    if col_idx is not None and col_idx < df.shape[1] else None
                )
                if not val or val.lower() == "nan":
                    row_data[label] = None
                elif label in ["Foam (cc)", "Water (cc)", "Zeta", "Conductivity", "Size", "PI"]:
                    num = re.search(r"[-+]?\d+\.?\d*", val)
                    row_data[label] = float(num.group()) if num else None
                else:
                    row_data[label] = val

            if row + 1 < df.shape[0]:
                next_row = df.iloc[row + 1]
                non_empty = next_row.dropna()
                if len(non_empty) == 1 and column_map.get("Foam Texture") in non_empty.index:
                    extra_texture = str(non_empty.values[0]).strip()
                    if extra_texture:
                        existing = row_data.get("Foam Texture", "")
                        row_data["Foam Texture"] = f"{existing}, {extra_texture}".strip(", ")
                    row += 1

            current_dilution_rows.append(row_data)
            row += 1
            continue

        # --- Dilution row ---
        row_text_combined = ",".join(df.iloc[row].fillna("").astype(str)).lower()
        dilution_search = re.search(r"(\d+\s*X)", row_text_combined, re.IGNORECASE)
        if dilution_search:
            flush_current_dilution()
            base_dilution = dilution_search.group(1).replace(" ", "").upper()
            extra_label = []
            tube_volume = ""

            for col in range(1, 12):
                if col < df.shape[1]:
                    val = str(df.iat[row, col]).strip()
                    if val and val.lower() != "nan":
                        if "ml" in val.lower():
                            tube_volume = val
                        else:
                            extra_label.append(val)

            last_dilution = base_dilution + (" " + " ".join(extra_label) if extra_label else "")
            last_tube_volume = tube_volume

            if "foam" in row_text_combined:
                header_row = df.iloc[row]
                row += 1
            elif row + 1 < df.shape[0] and "foam" in ",".join(df.iloc[row + 1].fillna("").astype(str)).lower():
                header_row = df.iloc[row + 1]
                row += 2
            else:
                row += 1
                continue

            column_map = {}
            for i, val in header_row.items():
                val_lower = str(val).strip().lower()
                if "foam amount" in val_lower or ("foam" in val_lower and "cc" in val_lower):
                    column_map["Foam (cc)"] = i
                elif "foam texture" in val_lower or "texture" in val_lower:
                    column_map["Foam Texture"] = i
                elif "zeta" in val_lower:
                    column_map["Zeta"] = i
                elif "pi" in val_lower:
                    column_map["PI"] = i
                elif "conductivity" in val_lower:
                    column_map["Conductivity"] = i
                elif "size" in val_lower:
                    column_map["Size"] = i
                elif "water" in val_lower and "cc" in val_lower:
                    column_map["Water (cc)"] = i
                elif "date" in val_lower:
                    column_map["Date"] = i
            continue

        row += 1

    flush_current_dilution()
    return samples, formulations


# ---------------------------------------------------------------------------
# Dilution processing
# ---------------------------------------------------------------------------

def process_dilution(dilution, date):
    pilot = np.nan
    temp_foam = np.nan
    ini_foam = DEFAULT_INITIAL_FOAM
    ratio = np.nan
    sonic = np.nan
    sample_description = np.nan

    if pd.isna(dilution):
        return pilot, temp_foam, ini_foam, NO_DILUTION_LABEL, ratio, sonic, sample_description, date

    text = str(dilution)

    if "AFC" in text:
        pilot = "AFC"

    if re.search(r"[-\s]{0,10}(no|not)[-\s]{0,10}\w*sonic[\w-]*", text, re.IGNORECASE):
        sonic = False
    elif re.search(r"[-\s]{0,10}sonic[\w-]*", text, re.IGNORECASE):
        sonic = True

    match_ratio = re.search(r"\(?(\d):(\d)\)?\s*ratio", text, re.IGNORECASE)
    if match_ratio:
        ratio = f"{match_ratio.group(1)}::{match_ratio.group(2)}"

    xcc_match = re.search(r"(\d+)\s*cc", text, re.IGNORECASE)
    if xcc_match:
        ini_foam = xcc_match.group(1).strip()

    xc_match = re.search(r"(\d+)\s*c(?!c)", text, re.IGNORECASE)
    if xc_match:
        temp_foam = int(xc_match.group(1))

    match_stream = re.search(r"\b(\d+)\s*stream\b", text, re.IGNORECASE)
    if match_stream:
        ratio = f"{match_stream.group(1)} stream"

    if re.search(r"RT", text, re.IGNORECASE):
        temp_foam = "RT"

    if re.search(r"synthetic brine", text, re.IGNORECASE):
        date = date  # brine_type handled elsewhere; no-op retained for traceability

    cleaned_text = text.strip(" ,;-").strip()

    return pilot, temp_foam, ini_foam, cleaned_text, ratio, sonic, sample_description, date


# ---------------------------------------------------------------------------
# DataFrame transformations
# ---------------------------------------------------------------------------

def clean_dilution(df):
    df = df[df["Day"].notna()].copy()
    df["Day_Num"] = df["Day"].str.extract(r"(\d+)").astype(int)
    max_day = df["Day_Num"].max()

    formulation_cols = [
        col for col in df.columns
        if col not in ["SampleID", "Day", "Day_Num", "Foam (cc)", "Foam Texture", "Date", "Baseline", "Pilot"]
    ]

    output_rows = []

    for (sample_id, dilution), group in df.groupby(["SampleID", "Dilution Ratio"]):
        row = {"SampleID": sample_id, "Dilution Ratio": dilution}

        for col in formulation_cols:
            if col in group.columns:
                non_null = group[col].dropna()
                row[col] = non_null.iloc[0] if not non_null.empty else np.nan

        day0_row = group[group["Day_Num"] == 0]
        row["Date"] = day0_row["Date"].iloc[0] if not day0_row.empty else np.nan

        pilot_val = group["Pilot"].dropna()
        row["Pilot"] = pilot_val.iloc[0] if not pilot_val.empty else ""

        baseline_marker = {
            r["Day_Num"]: ("*" in str(r.get("Baseline", "")))
            for _, r in group.iterrows()
        }

        for day in range(max_day + 1):
            day_row = group[group["Day_Num"] == day]
            if not day_row.empty:
                row[f"Day {day} - Foam (cc)"] = day_row["Foam (cc)"].values[0]
                foam_texture = (
                    day_row["Foam Texture"].values[0]
                    if not pd.isna(day_row["Foam Texture"].values[0]) else ""
                )
                if baseline_marker.get(day, False):
                    foam_texture = f"{foam_texture}, *" if foam_texture else "*"
                row[f"Day {day} - Foam Texture"] = foam_texture
            else:
                row[f"Day {day} - Foam (cc)"] = np.nan
                row[f"Day {day} - Foam Texture"] = ""

        output_rows.append(row)

    return output_rows


def make_sampleid_unique(df):
    df = df.copy()
    df["SampleID"] = df["SampleID"].astype(str)
    df.sort_values(by=["SampleID", "Dilution Ratio", "Date", "Day"], inplace=True)

    updated_rows = []
    for (sample_id, _dilution), group in df.groupby(["SampleID", "Dilution Ratio"]):
        suffix = 0
        current_suffix = ""
        for _idx, row in group.iterrows():
            if pd.notna(row["Day"]) and row["Day"].strip().lower() == "day 0":
                current_suffix = "" if suffix == 0 else f"-{suffix}"
                suffix += 1
            new_row = row.copy()
            new_row["SampleID"] = f"{sample_id}{current_suffix}"
            updated_rows.append(new_row)

    return pd.DataFrame(updated_rows)


def assign_pilot_column(df):
    df["Pilot"] = df["Dilution Ratio"].apply(
        lambda x: "AFC" if pd.notna(x) and "AFC" in str(x).upper() else None
    )
    return df


def clean_dilution_ratio(df, dilution_col="Dilution Ratio", desc_col="Sample Description"):
    if desc_col not in df.columns:
        df[desc_col] = None

    extracted = df[dilution_col].astype(str).str.extract(
        r"^(?P<valid>\d+[A-Z])\s*(?P<rest>.*)", expand=True
    )
    df[dilution_col] = extracted["valid"]

    new_desc = extracted["rest"].fillna("").str.strip()
    old_desc = df[desc_col].fillna("").astype(str).str.strip()
    df[desc_col] = (old_desc + " " + new_desc).str.strip()
    df[desc_col].replace(["", "nan", "None"], pd.NA, inplace=True)
    df[desc_col] = df[desc_col].fillna("")
    df[dilution_col].replace("", pd.NA, inplace=True)

    return df


def sort_columns_custom(df):
    percent_cols = [col for col in df.columns if "%" in col or "ppm" in col]
    fixed_order = ["SampleID", "Date", "Dilution Ratio", "Concentrate manufacturing method (Ratio)", "Brine Type"]
    performance_cols = [
        "Concentrate Stability (4C)", "Concentrate Stability (8C)",
        "Concentrate Stability (RT)", "Concentrate Stability (HT)",
        "Dilution Stability", "Zeta", "Conductivity", "Size", "PI",
        "Initial Foam Temp (dilution Temp)", "Temp Foam Monitoring", "Baseline", "Sonicated",
    ]
    day_cols = [col for col in df.columns if "Day" in col]

    desired_order = percent_cols + fixed_order + performance_cols + day_cols
    existing_cols = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in set(existing_cols)]

    return df[existing_cols + remaining_cols]


def update_sampleid_with_sonicated_status(df):
    df = df.copy()

    def _update(row):
        sampleid = str(row["SampleID"]).strip() if pd.notna(row["SampleID"]) else ""
        if pd.isna(row["Sonicated"]):
            return sampleid
        if row["Sonicated"] is False and "not sonicated" not in sampleid.lower():
            return f"{sampleid} - Not Sonicated"
        if row["Sonicated"] is True and "sonicated" not in sampleid.lower():
            return f"{sampleid} - Sonicated"
        return sampleid

    df["SampleID"] = df.apply(_update, axis=1)
    return df


def extract_half_life_samples(df):
    df = df.applymap(lambda x: x.replace("*", "") if isinstance(x, str) else x)
    foam_cols = [col for col in df.columns if re.match(r"Day\s+\d+\s+-\s+Foam\s+\(cc\)", col)]
    if not foam_cols:
        raise ValueError("No columns found matching 'Day N - Foam (cc)'")

    day_pattern = re.compile(r"Day\s+(\d+)\s+-\s+Foam\s+\(cc\)")
    foam_cols_sorted = sorted(foam_cols, key=lambda x: int(day_pattern.search(x).group(1)))

    half_life_rows = []
    for idx, row in df.iterrows():
        initial = row[foam_cols_sorted[0]]
        if pd.isna(initial) or initial == 0:
            continue
        for col in foam_cols_sorted[1:]:
            half = 0.5 * initial
            tolerance = 0.3 * half
            if pd.notna(row[col]) and (half - tolerance) < row[col] < (half + tolerance):
                half_life_rows.append(idx)
                break

    return df.loc[half_life_rows]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def check_password(password: str = "2025"):
    st.title("Foam Parser - No Oil")

    def _on_submit():
        st.session_state["password_correct"] = (
            st.session_state["password"] == password
        )
        del st.session_state["password"]

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password", type="password", on_change=_on_submit, key="password")
        return False
    if not st.session_state["password_correct"]:
        st.error("Incorrect password")
        return False
    return True
