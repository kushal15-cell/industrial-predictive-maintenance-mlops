from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import mannwhitneyu


ROOT_DIR = Path(__file__).resolve().parents[2]


@st.cache_data
def load_data():
    data_path = ROOT_DIR / "data" / "processed" / "train_fd001_processed.csv"

    if not data_path.exists():
        st.error("Processed dataset not found. Please run `dvc repro` first.")
        return None

    return pd.read_csv(data_path)


st.set_page_config(
    page_title="EDA | Predictive Maintenance",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Exploratory Data Analysis")

st.markdown(
    """
    This page explores the processed NASA CMAPSS FD001 dataset.
    The goal is to understand engine degradation behavior, sensor patterns,
    and the target variable: **Remaining Useful Life (RUL)**.
    """
)

df = load_data()

if df is None:
    st.stop()

st.divider()

# -----------------------------
# Dataset Summary
# -----------------------------

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Engine Units", df["unit_number"].nunique())

with col4:
    st.metric("Max Cycle", int(df["time_in_cycles"].max()))

st.dataframe(df.head(10), use_container_width=True)

st.divider()

# -----------------------------
# RUL Distribution
# -----------------------------

st.subheader("1. Distribution of Remaining Useful Life")

fig_rul = px.histogram(
    df,
    x="RUL",
    nbins=50,
    title="RUL Distribution",
    labels={"RUL": "Remaining Useful Life"},
)

st.plotly_chart(fig_rul, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    RUL is capped at 125 cycles. This is why many early-life engine records
    appear at the maximum RUL value. Near failure, RUL approaches 0.
    """
)

st.divider()

# -----------------------------
# Engine Cycle Distribution
# -----------------------------

st.subheader("2. Engine Cycle Distribution")

engine_life = (
    df.groupby("unit_number")["time_in_cycles"]
    .max()
    .reset_index()
    .rename(columns={"time_in_cycles": "max_cycles"})
)

fig_cycles = px.histogram(
    engine_life,
    x="max_cycles",
    nbins=30,
    title="Distribution of Engine Lifetimes",
    labels={"max_cycles": "Maximum Cycles Before Failure"},
)

st.plotly_chart(fig_cycles, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    Each engine fails after a different number of cycles. This variation makes
    Remaining Useful Life prediction more realistic and challenging.
    """
)

st.divider()

# -----------------------------
# Sensor Trend Over Time
# -----------------------------

st.subheader("3. Sensor Trend Over Time")

sensor_columns = [col for col in df.columns if col.startswith("sensor_")]

selected_engine = st.selectbox(
    "Select Engine Unit",
    sorted(df["unit_number"].unique()),
    index=0,
)

selected_sensor = st.selectbox(
    "Select Sensor",
    sensor_columns,
    index=1,
)

engine_df = df[df["unit_number"] == selected_engine]

fig_sensor = px.line(
    engine_df,
    x="time_in_cycles",
    y=selected_sensor,
    title=f"{selected_sensor} Trend for Engine {selected_engine}",
    labels={
        "time_in_cycles": "Time in Cycles",
        selected_sensor: selected_sensor,
    },
)

st.plotly_chart(fig_sensor, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    Sensor trends over time help us understand how degradation appears before
    failure. Some sensors change clearly as the engine approaches end of life,
    while others may remain almost constant.
    """
)

st.divider()

# -----------------------------
# RUL vs Sensor
# -----------------------------

st.subheader("4. RUL vs Selected Sensor")

selected_sensor_scatter = st.selectbox(
    "Select Sensor for RUL Relationship",
    sensor_columns,
    index=2,
)

sample_df = df.sample(min(5000, len(df)), random_state=42)

fig_scatter = px.scatter(
    sample_df,
    x=selected_sensor_scatter,
    y="RUL",
    opacity=0.5,
    title=f"RUL vs {selected_sensor_scatter}",
    labels={
        selected_sensor_scatter: selected_sensor_scatter,
        "RUL": "Remaining Useful Life",
    },
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    This plot helps identify whether a sensor has a relationship with Remaining
    Useful Life. Strong patterns may indicate that the sensor is useful for
    prediction.
    """
)

st.divider()

# -----------------------------
# Correlation Heatmap
# -----------------------------

st.subheader("5. Sensor Correlation Heatmap")

corr_columns = sensor_columns + ["RUL"]
corr = df[corr_columns].corr()

fig_corr = px.imshow(
    corr,
    text_auto=False,
    aspect="auto",
    title="Correlation Heatmap of Sensors and RUL",
)

st.plotly_chart(fig_corr, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    Correlation helps identify sensors that move together and sensors that are
    more related to RUL. Highly correlated features may contain overlapping
    information.
    """
)

st.divider()

# -----------------------------
# Low Variance Sensors
# -----------------------------

st.subheader("6. Low-Variance Sensor Check")

sensor_variance = (
    df[sensor_columns]
    .var()
    .sort_values()
    .reset_index()
    .rename(columns={"index": "sensor", 0: "variance"})
)

st.dataframe(sensor_variance, use_container_width=True)

low_variance_sensors = sensor_variance[sensor_variance["variance"] < 1e-6]["sensor"].tolist()

if low_variance_sensors:
    st.warning(
        f"Low-variance sensors detected: {', '.join(low_variance_sensors)}"
    )
else:
    st.success("No extremely low-variance sensors detected.")

st.markdown(
    """
    **Industry note:**  
    Sensors with very low variance may not add useful predictive signal. In a
    production project, these features can be reviewed for removal or monitoring.
    """
)
st.divider()

# -----------------------------
# Boxplots for Sensor Outlier Analysis
# -----------------------------

st.subheader("7. Sensor Boxplot and Outlier Analysis")

selected_box_sensor = st.selectbox(
    "Select Sensor for Boxplot",
    sensor_columns,
    index=3,
)

fig_box = px.box(
    df,
    y=selected_box_sensor,
    points="outliers",
    title=f"Boxplot of {selected_box_sensor}",
    labels={selected_box_sensor: selected_box_sensor},
)

st.plotly_chart(fig_box, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    Boxplots help identify spread, skewness, and potential outliers in sensor readings.

    In predictive maintenance, outliers should not be removed automatically because
    extreme sensor values may represent real degradation or abnormal machine behavior.
    """
)

st.divider()

# -----------------------------
# Boxplot by Engine Health Stage
# -----------------------------

st.subheader("8. Sensor Distribution by Engine Health Stage")

df_health = df.copy()

df_health["health_stage"] = pd.cut(
    df_health["RUL"],
    bins=[-1, 30, 100, 125],
    labels=["Near Failure", "Degrading", "Healthy"],
)

selected_stage_sensor = st.selectbox(
    "Select Sensor for Health Stage Comparison",
    sensor_columns,
    index=4,
)

fig_stage_box = px.box(
    df_health,
    x="health_stage",
    y=selected_stage_sensor,
    color="health_stage",
    title=f"{selected_stage_sensor} Distribution Across Engine Health Stages",
    labels={
        "health_stage": "Engine Health Stage",
        selected_stage_sensor: selected_stage_sensor,
    },
)

st.plotly_chart(fig_stage_box, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    This plot compares sensor readings across different RUL stages.  
    If a sensor changes clearly from Healthy to Near Failure, it may be useful
    for predicting Remaining Useful Life.
    """
)

st.divider()

# -----------------------------
# Hypothesis Testing
# -----------------------------

st.subheader("9. Hypothesis Testing: Healthy vs Near-Failure Engines")

st.markdown(
    """
    We test whether each sensor has a statistically significant difference between:

    - **Healthy engines:** RUL > 100  
    - **Near-failure engines:** RUL <= 30  

    **Null Hypothesis H0:** Sensor distribution is the same for healthy and near-failure engines.  
    **Alternative Hypothesis H1:** Sensor distribution is different between healthy and near-failure engines.

    Since sensor values may not be normally distributed, we use the **Mann-Whitney U test**.
    """
)

healthy_df = df[df["RUL"] > 100]
near_failure_df = df[df["RUL"] <= 30]

test_results = []

for sensor in sensor_columns:
    healthy_values = healthy_df[sensor]
    near_failure_values = near_failure_df[sensor]

    try:
        stat, p_value = mannwhitneyu(
            healthy_values,
            near_failure_values,
            alternative="two-sided"
        )

        decision = "Reject H0" if p_value < 0.05 else "Fail to Reject H0"

        test_results.append(
            {
                "sensor": sensor,
                "test_statistic": round(stat, 4),
                "p_value": round(p_value, 6),
                "decision": decision,
            }
        )

    except ValueError:
        test_results.append(
            {
                "sensor": sensor,
                "test_statistic": None,
                "p_value": None,
                "decision": "Test not applicable",
            }
        )

test_results_df = pd.DataFrame(test_results)

st.dataframe(test_results_df, use_container_width=True)

significant_sensors = test_results_df[
    test_results_df["decision"] == "Reject H0"
]["sensor"].tolist()

st.success(
    f"{len(significant_sensors)} sensors show statistically significant differences "
    "between healthy and near-failure engine states."
)

st.markdown(
    """
    **Industry interpretation:**  
    A statistically significant result does not automatically mean the sensor is important
    for the model, but it gives useful evidence that the sensor behavior changes as the
    engine approaches failure.
    """
)