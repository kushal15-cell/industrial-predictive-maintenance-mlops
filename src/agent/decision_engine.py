import json
import math
from datetime import datetime, timezone

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

    with (ROOT_DIR / "reports/metrics.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def evaluate_decision(
    monitoring_summary
):

    params = load_params()

    baseline = get_baseline_metrics()
    problems = []
    for metric in ("mae", "rmse"):
        for label, values in (("baseline", baseline), ("current", monitoring_summary.get("performance", {}))):
            value = values.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                problems.append(f"Invalid {label} {metric}.")
    count = monitoring_summary.get("sample_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < params["agent"].get("min_samples", 100):
        problems.append("Insufficient or unknown labeled sample count.")
    drift = monitoring_summary.get("drifted_feature_count")
    total = monitoring_summary.get("total_monitored_features")
    if not isinstance(drift, int) or isinstance(drift, bool) or drift < 0 or not isinstance(total, int) or total < drift:
        problems.append("Invalid drift feature counts.")
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(monitoring_summary["generated_at"])).total_seconds()
        if age < 0 or age > params["agent"].get("max_evidence_age_hours", 24) * 3600:
            problems.append("Monitoring evidence is stale or future-dated.")
    except (KeyError, ValueError, TypeError):
        problems.append("Monitoring timestamp is missing or invalid.")
    if problems:
        return {"decision": "WAIT_FOR_DATA", "retraining_required": False,
                "reasoning": problems, "evidence": monitoring_summary,
                "uncertainty": "Evidence validation failed; obtain fresh labeled monitoring data."}

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

        "uncertainty": "A threshold crossing does not prove retraining will improve the model; candidate evaluation is required.",
        "evidence": {
            "sample_count": count,
            "generated_at": monitoring_summary["generated_at"],
            "window": monitoring_summary.get("window"),
            "baseline_source": "reports/metrics.json",

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