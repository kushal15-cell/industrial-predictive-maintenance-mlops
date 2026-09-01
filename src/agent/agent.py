import json
from datetime import datetime, timezone

from src.agent.decision_engine import (
    evaluate_decision,
    load_monitoring_summary,
)
from src.config import ROOT_DIR, load_params


def generate_agent_report(
    decision
):

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    report = {
        "timestamp": timestamp,
        "agent": "industrial_predictive_maintenance_agent",
        "decision": decision,
    }

    return report


def save_agent_decision(
    report
):

    params = load_params()

    report_dir = (
        ROOT_DIR /
        params["monitoring"]["report_dir"]
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        report_dir /
        "agent_decision.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    return path


def print_decision(
    report
):

    decision = report[
        "decision"
    ]

    print("\n")
    print("=" * 70)
    print("MLOPS AGENT DECISION")
    print("=" * 70)

    print(
        f"\nDecision: "
        f"{decision['decision']}"
    )

    print("\nReasoning:")

    for reason in decision[
        "reasoning"
    ]:

        print(
            f"  - {reason}"
        )

    print("\nEvidence:")

    evidence = decision[
        "evidence"
    ]

    print(
        f"  Baseline MAE: "
        f"{evidence['baseline_mae']:.4f}"
    )

    print(
        f"  Current MAE: "
        f"{evidence['current_mae']:.4f}"
    )

    print(
        f"  MAE threshold: "
        f"{evidence['mae_threshold']:.4f}"
    )

    print(
        f"  Baseline RMSE: "
        f"{evidence['baseline_rmse']:.4f}"
    )

    print(
        f"  Current RMSE: "
        f"{evidence['current_rmse']:.4f}"
    )

    print(
        f"  RMSE threshold: "
        f"{evidence['rmse_threshold']:.4f}"
    )

    print(
        f"  Drifted features: "
        f"{evidence['drifted_feature_count']}"
    )

    print(
        f"  Required drifted features: "
        f"{evidence['minimum_drift_features']}"
    )

    print("\n" + "=" * 70)


def main():

    print(
        "Loading Evidently monitoring results..."
    )

    monitoring_summary = (
        load_monitoring_summary()
    )

    decision = evaluate_decision(
        monitoring_summary
    )

    report = generate_agent_report(
        decision
    )

    path = save_agent_decision(
        report
    )

    print_decision(
        report
    )

    print(
        f"\nAgent decision saved to:"
        f"\n{path}"
    )


if __name__ == "__main__":
    main()