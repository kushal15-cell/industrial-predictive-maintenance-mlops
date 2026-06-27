from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
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

    return joblib.load(model_path)


def calculate_drift_summary(reference_df, current_df, feature_columns, threshold=0.10):
    drift_results = []

    for feature in feature_columns:
        ref_mean = reference_df[feature].mean()
        cur_mean = current_df[feature].mean()

        if abs(ref_mean) < 1e-8:
            relative_change = 0
        else:
            relative_change = abs(cur_mean - ref_mean) / abs(ref_mean)

        is_drifted = relative_change > threshold

        drift_results.append(
            {
                "feature": feature,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "relative_change": relative_change,
                "drift_detected": is_drifted,
            }
        )

    return pd.DataFrame(drift_results)


st.set_page_config(
    page_title="Monitoring | Predictive Maintenance",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Monitoring Reports")

st.markdown(
    """
    This page simulates a basic production monitoring report for the predictive
    maintenance model.

    In real production, monitoring helps answer:

    - Has incoming sensor data changed compared to training data?
    - Are model predictions shifting over time?
    - Are certain sensors showing abnormal behavior?
    - Should the model be retrained?
    """
)

df = load_data()
model = load_model()

if df is None or model is None:
    st.stop()

st.divider()

# -----------------------------
# Reference vs Current Split
# -----------------------------

st.subheader("1. Reference vs Current Data")

st.markdown(
    """
    For this demo:

    - **Reference data** = earlier engine records
    - **Current data** = later engine records

    This simulates comparing historical training data with newer production data.
    """
)

df_sorted = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

split_index = int(len(df_sorted) * 0.7)

reference_df = df_sorted.iloc[:split_index].copy()
current_df = df_sorted.iloc[split_index:].copy()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Reference Rows", f"{len(reference_df):,}")

with col2:
    st.metric("Current Rows", f"{len(current_df):,}")

with col3:
    st.metric("Total Rows", f"{len(df_sorted):,}")

st.divider()

# -----------------------------
# Dataset Drift Summary
# -----------------------------

st.subheader("2. Dataset Drift Summary")

feature_columns = [
    col for col in df.columns
    if col not in ["RUL", "unit_number"]
]

drift_threshold = st.slider(
    "Relative mean change threshold for drift detection",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01,
)

drift_df = calculate_drift_summary(
    reference_df,
    current_df,
    feature_columns,
    threshold=drift_threshold,
)

drifted_features = drift_df[drift_df["drift_detected"] == True]
drift_share = len(drifted_features) / len(feature_columns)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Monitored Features", len(feature_columns))

with col2:
    st.metric("Drifted Features", len(drifted_features))

with col3:
    st.metric("Share of Drifted Features", f"{drift_share:.2%}")

if len(drifted_features) > 0:
    st.warning("⚠️ Dataset drift detected in one or more features.")
else:
    st.success("✅ No major dataset drift detected using the selected threshold.")

st.markdown(
    """
    **Interpretation:**  
    This simple drift check compares feature means between reference and current data.
    A production system would use more advanced statistical drift tests, but this gives
    a clear first monitoring layer.
    """
)

st.divider()

# -----------------------------
# Drift Table
# -----------------------------

st.subheader("3. Feature Drift Details")

drift_display = drift_df.copy()
drift_display["reference_mean"] = drift_display["reference_mean"].round(4)
drift_display["current_mean"] = drift_display["current_mean"].round(4)
drift_display["relative_change"] = drift_display["relative_change"].round(4)

st.dataframe(
    drift_display.sort_values("relative_change", ascending=False),
    use_container_width=True,
)

top_drift = drift_display.sort_values("relative_change", ascending=False).head(10)

fig_drift = px.bar(
    top_drift.sort_values("relative_change"),
    x="relative_change",
    y="feature",
    orientation="h",
    title="Top 10 Features by Relative Mean Change",
    labels={
        "relative_change": "Relative Change",
        "feature": "Feature",
    },
)

st.plotly_chart(fig_drift, use_container_width=True)

st.divider()

# -----------------------------
# RUL Distribution Drift
# -----------------------------

st.subheader("4. RUL Distribution Comparison")

reference_rul = reference_df[["RUL"]].copy()
reference_rul["dataset"] = "Reference"

current_rul = current_df[["RUL"]].copy()
current_rul["dataset"] = "Current"

rul_compare = pd.concat([reference_rul, current_rul], axis=0)

fig_rul = px.histogram(
    rul_compare,
    x="RUL",
    color="dataset",
    nbins=50,
    barmode="overlay",
    opacity=0.6,
    title="Reference vs Current RUL Distribution",
)

st.plotly_chart(fig_rul, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    If the current RUL distribution shifts strongly toward low values, the system may
    be seeing more near-failure engines than expected.
    """
)

st.divider()

# -----------------------------
# Prediction Drift
# -----------------------------

st.subheader("5. Prediction Drift")

drop_columns = ["RUL", "unit_number"]

X_reference = reference_df.drop(columns=drop_columns)
X_current = current_df.drop(columns=drop_columns)

reference_predictions = model.predict(X_reference)
current_predictions = model.predict(X_current)

prediction_compare = pd.DataFrame(
    {
        "predicted_rul": list(reference_predictions) + list(current_predictions),
        "dataset": ["Reference"] * len(reference_predictions)
        + ["Current"] * len(current_predictions),
    }
)

ref_pred_mean = reference_predictions.mean()
cur_pred_mean = current_predictions.mean()

prediction_change = abs(cur_pred_mean - ref_pred_mean) / abs(ref_pred_mean)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Reference Mean Prediction", f"{ref_pred_mean:.2f}")

with col2:
    st.metric("Current Mean Prediction", f"{cur_pred_mean:.2f}")

with col3:
    st.metric("Prediction Shift", f"{prediction_change:.2%}")

fig_pred = px.histogram(
    prediction_compare,
    x="predicted_rul",
    color="dataset",
    nbins=50,
    barmode="overlay",
    opacity=0.6,
    title="Reference vs Current Prediction Distribution",
)

st.plotly_chart(fig_pred, use_container_width=True)

if prediction_change > drift_threshold:
    st.warning("⚠️ Prediction drift detected.")
else:
    st.success("✅ No major prediction drift detected.")

st.markdown(
    """
    **Industry interpretation:**  
    Prediction drift means the model's output distribution has changed. This may happen
    because incoming engines are healthier, more degraded, or because sensor behavior
    has changed.

    Drift does not always mean the model is wrong, but it is a signal to investigate.
    """
)

st.divider()

# -----------------------------
# Monitoring Conclusion
# -----------------------------

st.subheader("6. Monitoring Conclusion")

if len(drifted_features) > 0 or prediction_change > drift_threshold:
    st.warning(
        """
        Monitoring detected possible drift. In a production system, this would trigger
        further investigation, model evaluation on recent labeled data, or retraining.
        """
    )
else:
    st.success(
        """
        No major drift was detected using the current simple monitoring rules.
        The model appears stable under this simulated reference/current comparison.
        """
    )

st.markdown(
    """
    ### Next Production Improvements

    - Add Evidently AI reports for statistical drift testing.
    - Store daily prediction logs.
    - Monitor real-time model performance after labels become available.
    - Trigger retraining when drift or performance degradation crosses a threshold.
    - Send alerts through email, Slack, or dashboard notifications.
    """
)