"""
pages_content/page_predict.py
Single-row manual prediction or batch CSV prediction.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

from state.session import init_state, get_value
from core.data.loader import load_csv


def render():
    init_state()

    st.title("🔮 Predictions")
    st.divider()

    # ── Guards ────────────────────────────────────────────────────────────
    model         = get_value("model.object")
    model_name    = get_value("model.name")
    task_type     = get_value("model.task_type")
    _pfn = get_value("data.processed_feature_names")
    feature_names = _pfn if _pfn is not None else get_value("data.feature_names")
    scaler        = get_value("preprocessing.scaler")
    encoder_map   = get_value("preprocessing.encoder") or {}

    if model is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Train** first.")
        st.stop()

    st.info(f"**Model:** {model_name} &nbsp;|&nbsp; **Task:** {task_type.capitalize()}")

    # ─────────────────────────────────────────────────────────────────────
    # Helper: apply same preprocessing pipeline to new data
    # ─────────────────────────────────────────────────────────────────────
    def preprocess_input(df_in: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoder + scaler to new input rows."""
        X = df_in.copy()
        # Encoding
        for col, enc in encoder_map.items():
            if col == "__ordinal__":
                continue
            if col in X.columns:
                X[col] = enc.transform(X[col].astype(str))
        # One-hot (if get_dummies was used — reindex to match training columns)
        if feature_names:
            X = pd.get_dummies(X)
            X = X.reindex(columns=feature_names, fill_value=0)
        # Scaling
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        return X

    # ─────────────────────────────────────────────────────────────────────
    # Mode selector
    # ─────────────────────────────────────────────────────────────────────
    mode = st.radio(
        "Prediction mode",
        ["🖊️ Single-row (manual)", "📂 Batch (upload CSV)"],
        horizontal=True,
    )
    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # A. Single-row prediction
    # ─────────────────────────────────────────────────────────────────────
    if mode.startswith("🖊️"):
        st.subheader("🖊️ Enter Feature Values")

        if not feature_names:
            st.warning("Feature names not found in session. Re-run preprocessing.")
            st.stop()

        _raw = get_value("data.raw")
        raw_df = _raw if _raw is not None else get_value("data.cleaned")
        input_vals: dict = {}

        cols = st.columns(min(3, len(feature_names)))
        for i, feat in enumerate(feature_names):
            with cols[i % len(cols)]:
                if raw_df is not None and feat in raw_df.columns:
                    dtype = raw_df[feat].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        input_vals[feat] = st.number_input(feat, value=int(raw_df[feat].median()), step=1)
                    elif pd.api.types.is_float_dtype(dtype):
                        input_vals[feat] = st.number_input(feat, value=float(raw_df[feat].median()), format="%.1f")
                    else:
                        options = raw_df[feat].dropna().unique().tolist()
                        input_vals[feat] = st.selectbox(feat, options)
                else:
                    input_vals[feat] = st.text_input(feat, value="0")

        if st.button("🔮 Predict", type="primary"):
            try:
                row_df = pd.DataFrame([input_vals])
                X_in = preprocess_input(row_df)
                prediction = model.predict(X_in)[0]
                st.success(f"**Prediction: {prediction}**")

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_in)[0]
                    classes = model.classes_
                    prob_df = pd.DataFrame({"Class": classes, "Probability": proba.round(4)})
                    st.dataframe(prob_df, use_container_width=True)

            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)

    # ─────────────────────────────────────────────────────────────────────
    # B. Batch prediction
    # ─────────────────────────────────────────────────────────────────────
    else:
        st.subheader("📂 Upload CSV for Batch Predictions")
        st.caption("The file should contain the same feature columns used during training (no target column needed).")

        batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_predict")
        if batch_file is not None:
            try:
                batch_df = load_csv(batch_file)
                st.write(f"Loaded **{len(batch_df):,}** rows.")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🔮 Run Batch Prediction", type="primary"):
                    X_batch = preprocess_input(batch_df)
                    preds = model.predict(X_batch)
                    result_df = batch_df.copy()
                    result_df["__prediction__"] = preds

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_batch)
                        for i, cls in enumerate(model.classes_):
                            result_df[f"prob_{cls}"] = proba[:, i].round(4)

                    st.success(f"✅ {len(result_df):,} predictions made.")
                    st.dataframe(result_df, use_container_width=True)

                    # Download button
                    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error("Failed to process batch file.")
                st.exception(e)