import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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


def load_metrics():
    metrics_path = ROOT_DIR / "reports" / "metrics.json"

    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as file:
        return json.load(file)


@st.cache_data
def load_feature_importance():
    importance_path = ROOT_DIR / "reports" / "feature_importance.csv"

    if not importance_path.exists():
        return None

    return pd.read_csv(importance_path)

@st.cache_data
def load_model_comparison():
    comparison_path = ROOT_DIR / "reports" / "model_comparison.csv"

    if not comparison_path.exists():
        return None

    return pd.read_csv(comparison_path)

st.set_page_config(
    page_title="Performance | Predictive Maintenance",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Performance Measures")

st.markdown(
    """
    This page evaluates the baseline Random Forest model used to predict
    **Remaining Useful Life (RUL)**.

    Since RUL prediction is a regression problem, the main evaluation metrics are:

    - **MAE:** average absolute prediction error
    - **RMSE:** penalizes large errors more strongly
    - **R² Score:** explains how much variation in RUL is captured by the model
    """
)

df = load_data()
model = load_model()
metrics = load_metrics()
feature_importance = load_feature_importance()
model_comparison = load_model_comparison()

if df is None or model is None:
    st.stop()

target = "RUL"
drop_columns = ["RUL", "unit_number"]

X = df.drop(columns=drop_columns)
y = df[target]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

st.divider()

# -----------------------------
# Metric Cards
# -----------------------------

st.subheader("1. Regression Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{mae:.4f}")

with col2:
    st.metric("RMSE", f"{rmse:.4f}")

with col3:
    st.metric("R² Score", f"{r2:.4f}")

st.markdown(
    """
    **Interpretation:**  
    The model predicts Remaining Useful Life with an average error of about
    11 cycles. RMSE is higher than MAE, which means some predictions have larger
    errors.
    """
)

st.divider()

# -----------------------------
# Model Comparison
# -----------------------------

st.subheader("2. Model Comparison")

if model_comparison is not None:
    st.markdown(
        """
        Multiple regression models were trained and compared using the same
        train-test split. This helps identify whether the baseline Random Forest
        is actually a strong choice or whether another model performs better.
        """
    )

    st.dataframe(model_comparison, use_container_width=True)

    best_model = model_comparison.sort_values(
        by=["mae", "rmse"],
        ascending=[True, True],
    ).iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Best Model", best_model["model_name"])

    with col2:
        st.metric("Best MAE", f"{best_model['mae']:.4f}")

    with col3:
        st.metric("Best RMSE", f"{best_model['rmse']:.4f}")

    with col4:
        st.metric("Best R²", f"{best_model['r2_score']:.4f}")

    fig_mae = px.bar(
        model_comparison.sort_values("mae"),
        x="model_name",
        y="mae",
        title="Model Comparison by MAE",
        labels={
            "model_name": "Model",
            "mae": "Mean Absolute Error",
        },
    )

    st.plotly_chart(fig_mae, use_container_width=True)

    fig_rmse = px.bar(
        model_comparison.sort_values("rmse"),
        x="model_name",
        y="rmse",
        title="Model Comparison by RMSE",
        labels={
            "model_name": "Model",
            "rmse": "Root Mean Squared Error",
        },
    )

    st.plotly_chart(fig_rmse, use_container_width=True)

    fig_r2 = px.bar(
        model_comparison.sort_values("r2_score", ascending=False),
        x="model_name",
        y="r2_score",
        title="Model Comparison by R² Score",
        labels={
            "model_name": "Model",
            "r2_score": "R² Score",
        },
    )

    st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown(
        """
        **Industry interpretation:**  
        Model comparison should happen before deployment. The selected production
        model should not only have good overall metrics but should also perform
        well in critical low-RUL ranges where maintenance decisions matter most.
        """
    )

else:
    st.warning(
        "Model comparison file not found. Run `python -m src.models.compare_models` first."
    )

st.divider()

# -----------------------------
# Actual vs Predicted
# -----------------------------

st.subheader("3. Actual vs Predicted RUL")

results_df = pd.DataFrame(
    {
        "actual_rul": y_test.values,
        "predicted_rul": predictions,
    }
)

fig_actual_pred = px.scatter(
    results_df,
    x="actual_rul",
    y="predicted_rul",
    opacity=0.5,
    title="Actual vs Predicted RUL",
    labels={
        "actual_rul": "Actual RUL",
        "predicted_rul": "Predicted RUL",
    },
)

fig_actual_pred.add_shape(
    type="line",
    x0=results_df["actual_rul"].min(),
    y0=results_df["actual_rul"].min(),
    x1=results_df["actual_rul"].max(),
    y1=results_df["actual_rul"].max(),
    line=dict(dash="dash"),
)

st.plotly_chart(fig_actual_pred, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    Points closer to the diagonal line indicate better predictions.
    Wider spread means higher prediction error.
    """
)

st.divider()

# -----------------------------
# Residual Distribution
# -----------------------------

st.subheader("4. Residual Error Distribution")

results_df["residual"] = results_df["actual_rul"] - results_df["predicted_rul"]
results_df["absolute_error"] = results_df["residual"].abs()

fig_residual = px.histogram(
    results_df,
    x="residual",
    nbins=50,
    title="Residual Distribution",
    labels={"residual": "Actual RUL - Predicted RUL"},
)

st.plotly_chart(fig_residual, use_container_width=True)

st.markdown(
    """
    **Interpretation:**  
    Residuals show whether the model tends to overpredict or underpredict RUL.
    A distribution centered near zero is generally better.
    """
)

st.divider()

# -----------------------------
# Error by RUL Range
# -----------------------------

st.subheader("5. Error by RUL Range")

results_df["rul_range"] = pd.cut(
    results_df["actual_rul"],
    bins=[-1, 30, 60, 100, 125],
    labels=["0-30 Near Failure", "31-60 Critical", "61-100 Degrading", "101-125 Healthy"],
)

error_by_range = (
    results_df.groupby("rul_range", observed=False)["absolute_error"]
    .mean()
    .reset_index()
)

fig_error_range = px.bar(
    error_by_range,
    x="rul_range",
    y="absolute_error",
    title="Average Absolute Error by RUL Range",
    labels={
        "rul_range": "RUL Range",
        "absolute_error": "Average Absolute Error",
    },
)

st.plotly_chart(fig_error_range, use_container_width=True)

st.markdown(
    """
    **Industry interpretation:**  
    Errors near low RUL ranges are more important because maintenance decisions
    are critical when an engine is close to failure.
    """
)

st.divider()

# -----------------------------
# Feature Importance
# -----------------------------

st.subheader("6. Feature Importance")

if feature_importance is not None:
    top_n = st.slider("Select number of top features", 5, 25, 15)

    top_features = feature_importance.head(top_n)

    fig_importance = px.bar(
        top_features.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Important Features",
        labels={
            "importance": "Importance",
            "feature": "Feature",
        },
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    st.dataframe(feature_importance, use_container_width=True)

else:
    st.warning(
        "Feature importance file not found. Run `dvc repro` after updating train_model.py."
    )

st.markdown(
    """
    **Interpretation:**  
    Feature importance helps identify which sensor readings or operational values
    contributed most to RUL prediction.

    In an industrial setting, this can help maintenance teams understand which
    sensors are most useful for monitoring engine degradation.
    """
)

st.divider()

# -----------------------------
# Business Summary
# -----------------------------

st.subheader("7. Business Interpretation")

st.markdown(
    f"""
    The baseline model achieved:

    - **MAE:** {mae:.2f} cycles
    - **RMSE:** {rmse:.2f} cycles
    - **R² Score:** {r2:.3f}

    This means the model can estimate engine Remaining Useful Life with a practical
    level of accuracy for a first baseline.

    However, in predictive maintenance, the most important region is the
    **near-failure zone**. Future improvements should focus on reducing error
    when actual RUL is below 30 cycles.
    """
)