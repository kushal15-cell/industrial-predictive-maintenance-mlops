# Reviewed deployment and interpretability

The checked-out implementation is Random Forest plus a rule-based agent. The UI labels that engine honestly; ANTHROPIC_API_KEY is reserved for the future LLM integration. No confidence interval or calibrated agent confidence is available. SHAP explains each selected observation in cycles relative to the tree explainer baseline, not causal sensor effects.

## Local services

Install requirements.txt in the model's compatible environment. Run `docker compose up --build` for Streamlit on 8501 and FastAPI on 8000. The existing Streamlit app still performs local inference; the API is a separate serving interface, not a proxy used by the UI. Both containers use the same checked-out model. GET /health returns 503 if the model cannot load compatibly; POST /predict accepts a features dictionary matching the model's recorded columns.

The current saved model reports sklearn 1.9.0 while the local environment uses 1.8.0. Prediction and release readiness intentionally stop on that mismatch. Resolve the original model environment before live deployment; these requirements are not yet a reproducible dependency lock.

## Render setup

Use two Python web services from the same repository and reviewed commit. Build command for both:

```sh
pip install -r requirements.txt 'dvc[gdrive]'
```

API start command: `python -m scripts.render_start api`
Frontend start command: `python -m scripts.render_start frontend`
API health check: `/health`. Frontend health check: `/_stcore/health` (process readiness; API health checks model readiness).

The startup script pulls the DVC model and processed dataset from Google Drive using the selected commit's dvc.lock, validates the release, and then starts the process using Render's PORT. No training happens at startup. Both services need GDRIVE_CREDENTIALS_DATA. Copy variable names from .env.example, and supply actual secrets only in Render/GitHub settings. Disable Render automatic deploys so unreviewed main changes do not bypass the release workflow.

## Release procedure

1. Obtain an eligible candidate and its outcome.json from the retraining workflow. Training approval alone does not authorize deployment.
2. In a release branch, place that candidate at models/random_forest_rul_model.pkl, update its DVC output hash and push its content to gdrive_remote. Retain the candidate's evaluated training dataset and split. Do not run `dvc repro` here: that would train another artifact.
3. Commit the updated dvc.lock with release/approved.json containing `approved: true`, `approved_by: <reviewer identity>`, and `evaluation: <complete outcome.json object>`. No fabricated approval manifest is included in this repository.
4. Require a human review of that release PR. Configure GitHub's production environment with required reviewers and branch protection; JSON approval is an attestation, not authentication.
5. Configure GDRIVE_CREDENTIALS_DATA and RENDER_DEPLOY_HOOK in GitHub. Merging the release to main runs tests, restores the model, rechecks eligibility/hash/runtime, then requests deployment of that exact commit. A hook targets one service; configure a second separately controlled deployment for the other service. The workflow does not claim the service is live after accepting the hook.
6. Verify Render deployment status and API /health model_sha256 against the approved candidate hash.

## Rollback

Render checks readiness before sending traffic to a new instance. A failed startup can leave the old deployment serving; a model that serves successfully but regresses is a different failure. Use Render's rollback action for the last known-good deployment, confirm its model hash through /health, and revert the release manifest and dvc.lock through review before the next deployment. Preserve old DVC blobs so the previous version remains retrievable.

Official references: [health checks](https://render.com/docs/health-checks), [deploy hooks](https://render.com/docs/deploy-hooks), [rollbacks](https://render.com/docs/rollbacks), [SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).


Deployment preparation update: the project .venv was verified to use Python 3.11.9 and scikit-learn 1.9.0, matching the incumbent artifact. The earlier mismatch was specific to the system Python 3.14 environment. requirements.txt now pins direct dependencies to the project environment, and all 13 tests pass there. Render credentials and release approval/artifact verification remain prerequisites for remote deployment.


## Current Render application release

The existing Render service is Docker-based and has no Google Drive credentials. For this application update, Docker restores `release/bundle.json`: a SHA-256-verified GitHub release mirror of the existing DVC model and processed dataset. Both local artifact MD5 values were checked against dvc.lock before packaging. The bundle includes saved NASA monitoring evidence, labeled as batch data. No candidate model is promoted by this path. New candidate promotion still requires the separate approved.json evaluation workflow. The fallback agent view explicitly labels its injected demonstration.

The GitHub release mirror avoids requiring Google Drive secrets for the existing public demo. It is not a configured Google Drive CI integration. To use the original DVC startup procedure instead, configure the credentials and approved release described above. The current Docker CMD serves Streamlit; the FastAPI container is available but is not provisioned as an additional Render service in this release.
