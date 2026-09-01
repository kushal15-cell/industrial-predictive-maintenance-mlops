import json

import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import ROOT_DIR, load_params


def load_monitoring_data():
    """
    Load reference and production monitoring datasets.
    """

    params = load_params()

    reference_path = (
        ROOT_DIR
        / params["monitoring"]["reference_path"]
    )

    production_path = (
        ROOT_DIR
        / params["monitoring"]["production_path"]
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference data not found: {reference_path}"
        )

    if not production_path.exists():
        raise FileNotFoundError(
            f"Production data not found: {production_path}"
        )

    reference = pd.read_csv(reference_path)
    production = pd.read_csv(production_path)

    print(f"Reference data: {reference.shape}")
    print(f"Production data: {production.shape}")

    return reference, production


def get_drift_datasets(reference, production):
    """
    Prepare equal-schema datasets for Evidently drift monitoring.

    We only compare features that exist in BOTH datasets.

    Columns excluded:
    - unit_number: identifier, not a model feature
    - RUL: target
    - production-only prediction monitoring columns
    """

    excluded_columns = {
        "unit_number",
        "RUL",
        "predicted_RUL",
        "prediction_error",
        "absolute_error",
        "status",
    }

    reference_columns = set(reference.columns)
    production_columns = set(production.columns)

    common_columns = (
        reference_columns
        .intersection(production_columns)
        .difference(excluded_columns)
    )

    feature_columns = sorted(common_columns)

    if not feature_columns:
        raise ValueError(
            "No common feature columns found for drift analysis."
        )

    reference_features = (
        reference[feature_columns]
        .copy()
    )

    production_features = (
        production[feature_columns]
        .copy()
    )

    print(
        f"Features monitored for drift: "
        f"{len(feature_columns)}"
    )

    return (
        reference_features,
        production_features,
        feature_columns,
    )


def calculate_performance(production):
    """
    Calculate current model performance using
    production ground-truth RUL and model predictions.

    In real production, this would only be possible
    after labels become available.
    """

    required_columns = {
        "RUL",
        "predicted_RUL",
    }

    missing = (
        required_columns
        - set(production.columns)
    )

    if missing:
        raise ValueError(
            f"Production data missing required "
            f"performance columns: {missing}"
        )

    y_true = production["RUL"]
    y_pred = production["predicted_RUL"]

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def run_drift_report(
    reference_features,
    production_features,
    report_path,
):
    """
    Run Evidently data drift analysis.
    """

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    evaluation = report.run(
        current_data=production_features,
        reference_data=reference_features,
    )

    evaluation.save_html(
        str(report_path)
    )

    return evaluation


def find_drift_count(result):
    """
    Extract the number of drifted columns from Evidently output.
    """

    metrics = result.get("metrics", [])

    for metric in metrics:

        metric_name = str(
            metric.get("metric_name", "")
        )

        metric_type = str(
            metric.get("config", {}).get("type", "")
        )

        if (
            "DriftedColumnsCount" in metric_name
            or
            "DriftedColumnsCount" in metric_type
        ):
            value = metric.get("value", {})

            if isinstance(value, dict):

                count = value.get("count")

                if count is not None:
                    return int(count)

    return 0

def extract_drifted_features(
    result,
    feature_columns,
):
    """
    Extract individual drifted feature names.

    Evidently's ValueDrift metrics contain:
        - column name
        - drift score
        - configured threshold

    A feature is considered drifted when:
        drift score > threshold
    """

    drifted_features = []

    metrics = result.get("metrics", [])

    for metric in metrics:

        metric_type = str(
            metric.get(
                "config",
                {}
            ).get(
                "type",
                ""
            )
        )

        if "ValueDrift" not in metric_type:
            continue

        config = metric.get(
            "config",
            {}
        )

        feature = config.get(
            "column"
        )

        threshold = config.get(
            "threshold"
        )

        value = metric.get(
            "value"
        )

        if (
            feature in feature_columns
            and
            isinstance(value, (int, float))
            and
            isinstance(threshold, (int, float))
            and
            value > threshold
        ):

            drifted_features.append(
                feature
            )

    return sorted(
        drifted_features
    )

def extract_drift_summary(
    evaluation,
    production,
    feature_columns,
):
    """
    Convert Evidently output and model performance
    into one machine-readable monitoring summary.
    """

    result = evaluation.dict()

    drift_count = find_drift_count(
        result
    )

    drifted_features = (
        extract_drifted_features(
            result,
            feature_columns,
        )
    )

    if drift_count is None:
        drift_count = len(
            drifted_features
        )

    performance = (
        calculate_performance(
            production
        )
    )

    summary = {
        "drift_detected": (
            drift_count > 0
        ),
        "drifted_feature_count": int(
            drift_count
        ),
        "drifted_features": (
            drifted_features
        ),
        "total_monitored_features": len(
            feature_columns
        ),
        "performance": performance,
    }

    if "predicted_RUL" in production.columns:

        summary[
            "prediction_mean"
        ] = float(
            production[
                "predicted_RUL"
            ].mean()
        )

        summary[
            "prediction_std"
        ] = float(
            production[
                "predicted_RUL"
            ].std()
        )

    return summary


def save_raw_evidently_result(
    evaluation,
    output_path,
):
    """
    Save the raw Evidently result for debugging
    and auditability.
    """

    result = evaluation.dict()

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            default=str,
        )


def main():

    params = load_params()

    report_dir = (
        ROOT_DIR
        / params["monitoring"]["report_dir"]
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference, production = (
        load_monitoring_data()
    )

    (
        reference_features,
        production_features,
        feature_columns,
    ) = get_drift_datasets(
        reference,
        production,
    )

    report_path = (
        report_dir
        / "data_drift_report.html"
    )

    raw_result_path = (
        report_dir
        / "evidently_raw_result.json"
    )

    summary_path = (
        report_dir
        / "monitoring_summary.json"
    )

    print()
    print(
        "Running Evidently drift analysis..."
    )

    evaluation = run_drift_report(
        reference_features,
        production_features,
        report_path,
    )

    save_raw_evidently_result(
        evaluation,
        raw_result_path,
    )

    summary = (
        extract_drift_summary(
            evaluation,
            production,
            feature_columns,
        )
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    print()
    print("=" * 60)
    print("MONITORING SUMMARY")
    print("=" * 60)

    print(
        f"Drift detected: "
        f"{summary['drift_detected']}"
    )

    print(
        f"Drifted features: "
        f"{summary['drifted_feature_count']}"
    )

    print(
        f"Total monitored features: "
        f"{summary['total_monitored_features']}"
    )

    print()

    print(
        f"Current MAE: "
        f"{summary['performance']['mae']:.4f}"
    )

    print(
        f"Current RMSE: "
        f"{summary['performance']['rmse']:.4f}"
    )

    print(
        f"Current R2: "
        f"{summary['performance']['r2']:.4f}"
    )

    print()

    print(
        f"Evidently HTML report saved to:"
        f"\n{report_path}"
    )

    print(
        f"\nRaw Evidently result saved to:"
        f"\n{raw_result_path}"
    )

    print(
        f"\nMonitoring summary saved to:"
        f"\n{summary_path}"
    )


if __name__ == "__main__":
    main()