import json
import subprocess

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
        return json.load(f)


def request_human_approval(report):
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
        print("\nHuman approval received.")
        return True

    print("\nRetraining rejected.")
    return False


def trigger_github_workflow(report):
    params = load_params()

    workflow = params["agent"]["github"]["workflow"]
    branch = params["agent"]["github"]["branch"]

    reasoning = report["decision"]["reasoning"]
    reason_text = " | ".join(reasoning)

    gh_path = r"C:\Program Files\GitHub CLI\gh.exe"

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
    ]

    print("\nTriggering GitHub Actions workflow...")

    try:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

        print("\nGitHub Actions workflow triggered successfully.")

        if result.stdout.strip():
            print(result.stdout)

        return True

    except FileNotFoundError:
        print("\nGitHub CLI executable was not found.")
        return False

    except subprocess.CalledProcessError as error:
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