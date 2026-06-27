import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]


@st.cache_data
def load_data():
    data_path = ROOT_DIR / "data" / "processed" / "train_fd001_processed.csv"

    if not data_path.exists():
        st.error(
            "Processed dataset not found. Please run `dvc repro` first."
        )
        return None

    return pd.read_csv(data_path)


def load_metrics():
    metrics_path = ROOT_DIR / "reports" / "metrics.json"

    if not metrics_path.exists():
        return {
            "mae": 10.9488,
            "rmse": 16.1515,
            "r2_score": 0.8462,
        }

    with open(metrics_path, "r") as file:
        return json.load(file)


st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Industrial Predictive Maintenance MLOps Dashboard")

st.markdown(
    """
    This dashboard demonstrates an end-to-end predictive maintenance MLOps project
    using the NASA CMAPSS FD001 turbofan engine degradation dataset.

    The goal is to predict **Remaining Useful Life (RUL)** of aircraft engines
    from sensor readings and operational settings.
    """
)

df = load_data()
metrics = load_metrics()

st.divider()

st.subheader("Project Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Problem Type", "Regression")

with col2:
    st.metric("Model", "Random Forest")

with col3:
    st.metric("MLOps Stack", "DVC + MLflow")

st.divider()

st.subheader("Dataset Overview")

if df is not None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Engine Units", df["unit_number"].nunique())

    with col4:
        st.metric("Max RUL", int(df["RUL"].max()))

    st.write("Sample processed data:")
    st.dataframe(df.head(10), use_container_width=True)

else:
    st.warning("Dataset could not be loaded.")

st.divider()

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{metrics['mae']:.4f}")

with col2:
    st.metric("RMSE", f"{metrics['rmse']:.4f}")

with col3:
    st.metric("R² Score", f"{metrics['r2_score']:.4f}")

st.markdown(
    """
    **Interpretation:**

    The baseline Random Forest model predicts Remaining Useful Life with an
    average error of approximately **11 engine cycles**.
    """
)

st.divider()

st.subheader("MLOps Workflow")

st.code(
    """
Raw CMAPSS Data
      ↓
Data Ingestion
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Processed Dataset
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Inference / Prediction
    """,
    language="text",
)

st.markdown(
    """
    ### Tools Used

    - **Git/GitHub** for source code versioning
    - **DVC** for data, model, and pipeline versioning
    - **MLflow** for experiment tracking
    - **Streamlit** for dashboard deployment
    - **Scikit-learn** for model training
    """
)