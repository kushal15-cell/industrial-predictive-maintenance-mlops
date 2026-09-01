import json

from src.config import ROOT_DIR, load_params


def load_monitoring_summary():

    params = load_params()

    summary_path = (
        ROOT_DIR /
        params["monitoring"]["report_dir"] /
        "monitoring_summary.json"
    )

    if not summary_path.exists():

        raise FileNotFoundError(
            f"Monitoring summary not found: "
            f"{summary_path}"
        )

    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def get_baseline_metrics():

    return {
        "mae": 10.8955,
        "rmse": 16.0202,
    }


def evaluate_decision(
    monitoring_summary
):

    params = load_params()

    baseline = get_baseline_metrics()

    current = monitoring_summary[
        "performance"
    ]

    drift_count = monitoring_summary.get(
        "drifted_feature_count",
        0
    )

    drift_threshold = params[
        "monitoring"
    ][
        "drift_threshold"
    ]

    min_drift_features = params[
        "agent"
    ][
        "min_drift_features"
    ]

    rmse_multiplier = params[
        "monitoring"
    ][
        "performance"
    ][
        "rmse_multiplier"
    ]

    mae_multiplier = params[
        "monitoring"
    ][
        "performance"
    ][
        "mae_multiplier"
    ]

    baseline_rmse = baseline["rmse"]
    baseline_mae = baseline["mae"]

    current_rmse = current["rmse"]
    current_mae = current["mae"]

    rmse_limit = (
        baseline_rmse *
        rmse_multiplier
    )

    mae_limit = (
        baseline_mae *
        mae_multiplier
    )

    performance_degraded = (
        current_rmse > rmse_limit
        or
        current_mae > mae_limit
    )

    significant_drift = (
        drift_count >= min_drift_features
    )

    require_performance = params[
        "agent"
    ][
        "require_performance_degradation"
    ]

    if require_performance:

        retraining_required = (
            significant_drift
            and
            performance_degraded
        )

    else:

        retraining_required = (
            significant_drift
            or
            performance_degraded
        )

    reasons = []

    if significant_drift:

        reasons.append(
            f"{drift_count} features show significant drift."
        )

    else:

        reasons.append(
            "Feature drift is below the configured threshold."
        )

    if performance_degraded:

        reasons.append(
            "Model performance has degraded beyond the configured threshold."
        )

    else:

        reasons.append(
            "Model performance remains within the accepted range."
        )

    if retraining_required:

        decision = "RETRAIN_RECOMMENDED"

    else:

        decision = "NO_RETRAINING"

    return {
        "decision": decision,

        "retraining_required": (
            retraining_required
        ),

        "reasoning": reasons,

        "evidence": {

            "baseline_mae": baseline_mae,
            "current_mae": current_mae,
            "mae_threshold": mae_limit,

            "baseline_rmse": baseline_rmse,
            "current_rmse": current_rmse,
            "rmse_threshold": rmse_limit,

            "drifted_feature_count": (
                drift_count
            ),

            "minimum_drift_features": (
                min_drift_features
            ),
        },
    }