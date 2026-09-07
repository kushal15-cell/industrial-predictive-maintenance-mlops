from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.agent.retrain import load_compatible_model
from src.models.explain import explain_prediction
import plotly.express as px
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]


@st.cache_data
def load_data():
    data_path = ROOT_DIR / "data" / "processed" / "train_fd001_processed.csv"

    if not data_path.exists():
        st.error("Processed dataset not found. Please run `dvc repro` first.")
        return None

    return pd.read_csv(data_path)


@st.cache_resource
def load_model():
    model_path = ROOT_DIR / "models" / "random_forest_rul_model.pkl"

    if not model_path.exists():
        st.error("Model file not found. Please run `dvc repro` first.")
        return None

    try:
        return load_compatible_model(model_path)
    except Exception as error:
        st.error(f"Model unavailable: {error}")
        return None


st.set_page_config(
    page_title="Predict RUL | Predictive Maintenance",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Predict Remaining Useful Life")

st.markdown(
    """
    This page uses the trained Random Forest model to predict the
    **Remaining Useful Life (RUL)** of an engine based on sensor readings
    and operational settings.

    For the first dashboard version, prediction is done by selecting an
    existing engine record from the processed dataset.
    """
)

st.caption("Historical batch demo, not live telemetry. A calibrated prediction interval is not available.")

df = load_data()
model = load_model()

if df is None or model is None:
    st.stop()

st.divider()

# -----------------------------
# User selection
# -----------------------------

st.subheader("1. Select Engine Record")

engine_units = sorted(df["unit_number"].unique())

selected_engine = st.selectbox(
    "Select Engine Unit",
    engine_units,
)

engine_df = df[df["unit_number"] == selected_engine]

selected_cycle = st.selectbox(
    "Select Time Cycle",
    sorted(engine_df["time_in_cycles"].unique()),
)

selected_row = engine_df[engine_df["time_in_cycles"] == selected_cycle].iloc[0]

st.markdown("### Selected Engine Record")

display_columns = [
    "unit_number",
    "time_in_cycles",
    "RUL",
]

st.dataframe(
    selected_row[display_columns].to_frame().T,
    use_container_width=True,
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------

st.subheader("2. Model Prediction")

drop_columns = ["RUL", "unit_number"]

input_data = selected_row.drop(labels=drop_columns).to_frame().T

prediction = model.predict(input_data)[0]
actual_rul = selected_row["RUL"]
error = abs(actual_rul - prediction)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Estimated cycles until failure", f"{prediction:.2f} cycles")

with col2:
    st.metric("Dataset target (clipped cycles)", f"{actual_rul:.2f} cycles")

with col3:
    st.metric("Absolute Error", f"{error:.2f} cycles")

st.divider()

# -----------------------------
# Risk level
# -----------------------------

st.subheader("3. Maintenance Risk Level")

if prediction <= 30:
    risk_level = "High Risk"
    recommendation = (
        "Engine is close to failure. Maintenance inspection should be prioritized."
    )
    st.error(f"🚨 {risk_level}: {recommendation}")

elif prediction <= 60:
    risk_level = "Medium Risk"
    recommendation = (
        "Engine is degrading. Schedule maintenance planning and monitor closely."
    )
    st.warning(f"⚠️ {risk_level}: {recommendation}")

else:
    risk_level = "Low Risk"
    recommendation = (
        "Engine appears healthy. Continue regular monitoring."
    )
    st.success(f"✅ {risk_level}: {recommendation}")

st.divider()

# -----------------------------
# Feature values
# -----------------------------

st.subheader("4. Input Features Used for Prediction")

st.dataframe(input_data, use_container_width=True)

st.markdown(
    """
    **Industry interpretation:**  
    In production, this input would come from live engine sensor data or a batch
    of recently collected operational records. The trained model would estimate
    Remaining Useful Life and support maintenance decision-making.
    """
)

st.subheader("Why this estimate?")
try:
    with st.spinner("Calculating sensor contributions..."):
        baseline, explained_prediction, contributions = explain_prediction(model, input_data)
    st.write(f"Reference estimate: {baseline:.1f} cycles. This observation shifts the estimate to {explained_prediction:.1f} cycles.")
    top = contributions.loc[contributions["Contribution (cycles)"].abs().nlargest(10).index].copy()
    top["Effect"] = top["Contribution (cycles)"].apply(lambda value: "Raises estimate" if value >= 0 else "Lowers estimate")
    st.plotly_chart(px.bar(top.sort_values("Contribution (cycles)"), x="Contribution (cycles)", y="Feature", color="Effect", orientation="h", hover_data=["Reading"]), use_container_width=True)
    st.caption("Top 10 SHAP contributions: positive adds cycles; negative subtracts cycles. These explain model behavior, not physical causes or confidence. Omitted features also contribute.")
    with st.expander("All sensor contributions"):
        st.dataframe(contributions, use_container_width=True)
except Exception as error:
    st.warning(f"Explanation unavailable: {error}")
