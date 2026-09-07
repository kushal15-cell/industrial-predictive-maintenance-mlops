# Agent reliability

This checkout uses a threshold rule agent and Random Forest training. No Anthropic call is present; the trace identifies its engine explicitly. This layer can also retain a future LLM's structured explanation without storing private chain-of-thought.

## Generate and review a decision

```sh
python -m src.monitoring.evidently_monitor
python -m src.agent.agent
streamlit run app/main.py
```

Open **Agent Review** in Streamlit. Each decision has a unique ID, model SHA-256, full monitoring snapshot, policy, baseline source, measured values, thresholds, explanation and uncertainty. History lives under `monitoring/reports/decisions/<id>/record.json`, with separate lifecycle event files. These are local audit records, not a tamper-proof audit service. Batch monitoring records sample counts and generation time; actual observation start/end times are unknown and explicitly null.

Missing/invalid metrics, insufficient samples (default 100), or stale evidence (default 24 hours) yield WAIT_FOR_DATA. Baseline metrics come from reports/metrics.json instead of hard-coded constants. Keep that file paired with the incumbent model. Monitoring generation time describes computation freshness, not sensor observation freshness.

## Approve training and inspect the outcome

```sh
python -m src.agent.approval
```

The command requires typed APPROVE, records the local actor, revalidates evidence and model identity, and submits a workflow with the decision ID. A dispatched decision cannot be submitted again through this flow. An ambiguous dispatch failure requires inspecting GitHub before creating a new decision; it is never retried automatically. The workflow title contains the decision ID. GitHub repository workflow-dispatch permissions control who can manually approve runs; the boolean input is an operator attestation, not a protected-environment review.

CI needs a populated `gdrive_remote` and the `GDRIVE_CREDENTIALS_DATA` repository secret. The existing default remote points to a Windows local directory, so CI explicitly selects Google Drive. No credentials were configured by this change.

CI restores the DVC-versioned incumbent and processed data, verifies the incumbent hash against the approved decision, copies training inputs into a temporary workspace, and trains there. Both models are evaluated on the identical engine holdout from the configured split. Lower RMSE and no MAE regression are required. Ties, invalid metrics and evaluation exceptions fail closed. Only an eligible candidate is uploaded. Neither this workflow nor the runner modifies or deploys the incumbent.

The DVC incumbent must match the deployed artifact. The holdout assumes the current engine split was excluded during incumbent training; there is no historical training manifest to prove that. Do not change the split or training implementation without validating this assumption. This gate demonstrates regression containment, not statistical significance or performance on a new external test population.

Download the `retraining-audit-<run-id>-<attempt>` artifact, then attach it to the local history:

```sh
python -m src.agent.record_outcome path/to/downloaded-artifact
```

The review page then shows the run URL, status, both metric sets and gate result. Training exceptions produce PIPELINE_FAILED and a training log. The workflow's always-run audit step captures failures before training as well. A hard runner loss can prevent artifact upload; inspect GitHub run status in that case. Outcome import is explicit, not a background watcher. Candidates passing evaluation remain NOT_PROMOTED pending a separate deployment review.

## Injected borderline wrong-call case

Run `python -m src.agent.reliability_demo`. It writes `docs/examples/borderline-decision.json`; it does not call GitHub or train/deploy a model.

The synthetic batch has 150 labeled rows, two drifted features (exactly the minimum), and MAE/RMSE at 1.251 times baseline (just above the 1.25 threshold). The rule recommends retraining. A simulated human approves it, but injected candidate metrics worsen from MAE 10 / RMSE 16 to MAE 11 / RMSE 17. Promotion is blocked.

This is an intentionally injected false-positive recommendation, not an observed production incident or proof that drift was spurious. It demonstrates that recommendation and approval do not imply model quality. The record retains the weak evidence and failed candidate comparison for review.

## Verification

```sh
python -m unittest discover -s tests -v
```

Tests cover borderline recommendations, evidence abstention, regression/tie/nonfinite gate rejection, approval enforcement, retained history, and a failing training subprocess that leaves incumbent bytes unchanged. CI runs the same suite before accessing DVC artifacts.

## Local integration check: runtime incompatibility

The real local retraining check on 2026-09-07 found that the incumbent pickle reports scikit-learn 1.9.0 while the local runtime has 1.8.0. This is an observed compatibility failure, separate from the synthetic borderline example. The runner now treats scikit-learn's InconsistentVersionWarning as an error for both incumbent and candidate loads and records Python/package versions in outcome.json. The follow-up check stops with PIPELINE_FAILED before training or candidate publication.

Resolve this by reproducing the incumbent's training environment from its original dependency lock, or by deliberately establishing a new incumbent in a supported, pinned environment. Do not silently reserialize the existing pickle to change its version metadata. Requirements are currently unpinned, so CI compatibility is not established.

GitHub repository secret names were inspected: no repository Actions secrets were configured. Google Drive restore still needs credentials before live CI validation. Neither deployment nor remote retraining was triggered.


Deployment preparation update: the project .venv was verified to use Python 3.11.9 and scikit-learn 1.9.0, matching the incumbent artifact. The earlier mismatch was specific to the system Python 3.14 environment. requirements.txt now pins direct dependencies to the project environment, and all 13 tests pass there. Render credentials and release approval/artifact verification remain prerequisites for remote deployment.
