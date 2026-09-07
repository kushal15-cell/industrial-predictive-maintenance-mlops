import json

from src.agent.decision_engine import (
    evaluate_decision,
    load_monitoring_summary,
)
from src.config import ROOT_DIR, load_params
from src.agent.trace import new_report, event, write_json


def generate_agent_report(decision, monitoring_summary=None):
    return new_report(decision, monitoring_summary)


def save_agent_decision(report):
    path = ROOT_DIR / load_params()["monitoring"]["report_dir"] / "agent_decision.json"
    if not report["lifecycle"]:
        event(report, "RECOMMENDED" if report["decision"]["retraining_required"] else "REVIEWED")
    write_json(path, report)
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

    print(json.dumps(evidence, indent=2))


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
        decision, monitoring_summary
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