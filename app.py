import streamlit as st
import onnxruntime as ort
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="UAV IMU Predictor", layout="centered")

# ==============================================================
# LOAD ASSETS (NO TORCH)
# ==============================================================
@st.cache_resource
def load_assets():
    with open("uav_runtime.pkl", "rb") as f:
        runtime = pickle.load(f)

    session = ort.InferenceSession(
        "uav_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    return runtime, session

runtime, session = load_assets()

seq_length   = runtime["seq_length"]
feature_cols = runtime["feature_cols"]
target_cols  = runtime["target_cols"]
scaler_X     = runtime["scaler_X"]
scaler_y     = runtime["scaler_y"]

# ==============================================================
# UI
# ==============================================================
st.title("UAV IMU Prediction (ONNX)")
st.write("CPU-only inference | No PyTorch")

uploaded_file = st.file_uploader(
    "Upload IMU CSV file",
    type=["csv"]
)

# ==============================================================
# CSV HANDLING
# ==============================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # ---- Validate columns ----
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    if len(df) < seq_length:
        st.error(f"CSV must have at least {seq_length} rows")
        st.stop()

    # ---- Prepare input ----
    X_raw = df[feature_cols].values
    X_scaled = scaler_X.transform(X_raw)

    sequences = []
    for i in range(len(X_scaled) - seq_length):
        sequences.append(X_scaled[i:i+seq_length])

    X_seq = np.array(sequences, dtype=np.float32)

    st.write(f"Total sequences created: {len(X_seq)}")

    # ==============================================================
    # PREDICTION (SAFE BATCH=1 LOOP)
    # ==============================================================
    if st.button("Run Prediction"):
        preds = []

        with st.spinner("Running ONNX inference..."):
            for i in range(len(X_seq)):
                single_input = X_seq[i:i+1]
                pred = session.run(None, {"input": single_input})[0]
                preds.append(pred)

        preds = np.vstack(preds)
        preds_inv = scaler_y.inverse_transform(preds)

        pred_df = pd.DataFrame(preds_inv, columns=target_cols)

        # ==============================================================
        # OUTPUT TABLE
        # ==============================================================
        st.subheader("Prediction Output (First Rows)")
        st.dataframe(pred_df.head())

        # ==============================================================
        # VISUALIZATION 1: LINE PLOTS
        # ==============================================================
        st.subheader("Prediction Trends")

        plot_targets = st.multiselect(
            "Select targets to visualize",
            target_cols,
            default=target_cols[:3]
        )

        for col in plot_targets:
            fig, ax = plt.subplots()
            ax.plot(pred_df[col].values, label=col)
            ax.set_title(f"{col} Prediction")
            ax.set_xlabel("Time Step")
            ax.set_ylabel(col)
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

        # ==============================================================
        # VISUALIZATION 2: ERROR DISTRIBUTION
        # ==============================================================
        st.subheader("Prediction Distribution")

        fig, ax = plt.subplots()
        ax.hist(pred_df.values.flatten(), bins=50)
        ax.set_title("Prediction Value Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True)
        st.pyplot(fig)

        # ==============================================================
        # VISUALIZATION 3: STATISTICS
        # ==============================================================
        st.subheader("Prediction Statistics")
        st.dataframe(pred_df.describe())

        # ==============================================================
        # DOWNLOAD
        # ==============================================================
        csv_out = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            csv_out,
            "predictions.csv",
            "text/csv"
        )

else:
    st.info("Please upload a CSV file to start prediction.")
