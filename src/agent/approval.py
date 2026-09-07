import json
import subprocess
import shutil
import getpass
from src.agent.trace import event, trace_dir, digest
from src.agent.decision_engine import evaluate_decision
from src.agent.agent import save_agent_decision

from src.config import ROOT_DIR, load_params


def load_agent_decision():
    params = load_params()

    path = (
        ROOT_DIR
        / params["monitoring"]["report_dir"]
        / "agent_decision.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Agent decision not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)
    if report.get('decision_id'):
        return json.loads((trace_dir() / report['decision_id'] / 'record.json').read_text(encoding='utf-8'))
    return report


def request_human_approval(report):
    if not report.get("decision_id"):
        raise ValueError("Legacy decision: regenerate the agent report before approval.")
    if any(e["status"] in {"APPROVED", "DISPATCH_REQUESTED", "DISPATCHED", "DISPATCH_UNKNOWN"} for e in report.get("lifecycle", [])):
        raise ValueError("This decision was already approved or dispatched. Review its run before retrying.")
    fresh = evaluate_decision(report.get("monitoring_snapshot") or {})
    if not report.get('model_sha256') or report['model_sha256'] != digest(ROOT_DIR / 'models/random_forest_rul_model.pkl'):
        event(report, 'APPROVAL_BLOCKED', reason='Incumbent model missing or changed; regenerate evidence and decision.')
        save_agent_decision(report)
        return False
    if not fresh["retraining_required"]:
        event(report, "APPROVAL_BLOCKED", reason="Evidence no longer supports retraining.")
        save_agent_decision(report)
        return False
    decision = report["decision"]

    if not decision["retraining_required"]:
        print("\nAgent does not recommend retraining.")
        return False

    print("\n")
    print("=" * 70)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 70)

    print(
        "\nThe MLOps agent recommends retraining."
    )

    print("\nReason:")

    for reason in decision["reasoning"]:
        print(f"  - {reason}")

    answer = input(
        "\nType APPROVE to continue: "
    )

    if answer.strip().upper() == "APPROVE":
        event(report, "APPROVED", actor=getpass.getuser())
        save_agent_decision(report)
        print("\nHuman approval received.")
        return True

    event(report, "REJECTED", actor=getpass.getuser())
    save_agent_decision(report)
    print("\nRetraining rejected.")
    return False


def trigger_github_workflow(report):
    params = load_params()

    workflow = params["agent"]["github"]["workflow"]
    branch = params["agent"]["github"]["branch"]

    reasoning = report["decision"]["reasoning"]
    reason_text = " | ".join(reasoning)

    if not report.get("lifecycle") or report["lifecycle"][-1]["status"] != "APPROVED":
        raise ValueError("A recorded human approval is required before dispatch.")
    gh_path = shutil.which("gh") or "gh"

    command = [
        gh_path,
        "workflow",
        "run",
        workflow,
        "--ref",
        branch,
        "-f",
        f"reason={reason_text}",
        "-f",
        "approved=true",
        "-f",
        f"decision_id={report['decision_id']}",
        "-f",
        f"incumbent_sha256={report['model_sha256']}",
    ]

    print("\nTriggering GitHub Actions workflow...")

    event(report, "DISPATCH_REQUESTED")
    save_agent_decision(report)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )

        print("\nGitHub Actions workflow triggered successfully.")

        if result.stdout.strip():
            print(result.stdout)

        event(report, "DISPATCHED", run_lookup=f"GitHub Actions run title: Retrain {report['decision_id']}")
        save_agent_decision(report)
        return True

    except FileNotFoundError:
        event(report, "DISPATCH_FAILED", reason="GitHub CLI executable not found")
        save_agent_decision(report)
        print("\nGitHub CLI executable was not found.")
        return False

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        event(report, "DISPATCH_UNKNOWN", reason="Dispatch failed or timed out; inspect GitHub before retrying.")
        save_agent_decision(report)
        print("\nFailed to trigger GitHub Actions workflow.")

        if error.stdout:
            print(error.stdout)

        if error.stderr:
            print(error.stderr)

        return False

def main():
    report = load_agent_decision()

    approved = request_human_approval(
        report
    )

    if not approved:
        print(
            "\nNo retraining will be triggered."
        )
        return

    triggered = trigger_github_workflow(
        report
    )

    if triggered:
        print(
            "\nCI/CD retraining request submitted."
        )

        print(
            "Check GitHub Actions for the workflow run."
        )


if __name__ == "__main__":
    main()
