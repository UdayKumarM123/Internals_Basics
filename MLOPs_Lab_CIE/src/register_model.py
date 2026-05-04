"""
Task 2 - Model Versioning
- Reads step1_s1.json to identify the best model and its RMSE
- Queries MLflow to find the run_id for the best model (Ridge)
- Registers the model artifact in the MLflow Model Registry
- Registered model name: "testforge-defect-count-predictor"
- Retrieves the assigned version number
- Saves results to results/step2_s6.json
"""

import json
import pathlib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent.parent   # MLOPs_Lab_CIE/
RESULTS_DIR = ROOT / "results"
STEP1_JSON  = RESULTS_DIR / "step1_s1.json"
OUT_JSON    = RESULTS_DIR / "step2_s6.json"

# ── MLflow setup (same tracking URI as Task 1) ────────────────────────────────
TRACKING_URI  = f"file:///{(ROOT / 'mlruns').as_posix()}"
EXPERIMENT_NAME      = "testforge-defect-count"
REGISTERED_MODEL_NAME = "testforge-defect-count-predictor"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# ── Read Task 1 results ───────────────────────────────────────────────────────
with open(STEP1_JSON) as f:
    step1 = json.load(f)

best_model_name  = step1["best_model"]            # "Ridge"
best_rmse        = step1["best_metric_value"]     # 1.74592
print(f"Best model from Task 1: {best_model_name}  RMSE={best_rmse}")

# ── Locate the MLflow run for the best model ──────────────────────────────────
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{best_model_name}'",
    order_by=["metrics.rmse ASC"],
    max_results=1,
)

if not runs:
    raise RuntimeError(
        f"No MLflow run found with name '{best_model_name}'. "
        "Ensure train.py was executed and the run was logged correctly."
    )

best_run = runs[0]
run_id   = best_run.info.run_id
print(f"Found run_id: {run_id}")

# ── Register the model ────────────────────────────────────────────────────────
# Model artifact path logged in train.py: artifact_path=model_name
model_uri = f"runs:/{run_id}/{best_model_name}"
print(f"Registering model from URI: {model_uri}")

mv = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)

version = int(mv.version)
print(f"Registered as version: {version}")

# ── Build and save output JSON ────────────────────────────────────────────────
output = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "version": version,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": best_rmse,
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[DONE] Results saved -> {OUT_JSON}")
print(json.dumps(output, indent=2))
