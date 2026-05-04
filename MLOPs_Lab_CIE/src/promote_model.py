"""
Task 3 - Model Promotion
- Reads step2_s6.json to get version 1 (champion) info
- Assigns alias "live" to version 1
- Trains a second Ridge variant (random_state=99) on the same data/split
- Logs to MLflow and registers as version 2 (challenger)
- Compares RMSE: if version 2 is better -> moves "live" to v2 (promoted)
                  else                  -> keeps "live" on v1 (kept)
- Saves results to results/step3_s7.json
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "training_data.csv"
RESULTS_DIR = ROOT / "results"
STEP2_JSON  = RESULTS_DIR / "step2_s6.json"
OUT_JSON    = RESULTS_DIR / "step3_s7.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup (same as previous tasks) ─────────────────────────────────────
TRACKING_URI          = f"file:///{(ROOT / 'mlruns').as_posix()}"
EXPERIMENT_NAME       = "testforge-defect-count"
REGISTERED_MODEL_NAME = "testforge-defect-count-predictor"
ALIAS_NAME            = "live"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri=TRACKING_URI)

# ── Read Task 2 results (champion = version 1) ────────────────────────────────
with open(STEP2_JSON) as f:
    step2 = json.load(f)

champion_version    = step2["version"]          # 1
champion_rmse       = step2["source_metric_value"]  # 1.74592
print(f"Champion: version={champion_version}  RMSE={champion_rmse}")

# ── Step 1: Assign "live" alias to version 1 ─────────────────────────────────
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias=ALIAS_NAME,
    version=str(champion_version),
)
print(f"Alias '{ALIAS_NAME}' -> version {champion_version}")

# ── Step 2: Train challenger model (Ridge, random_state=99) ──────────────────
df = pd.read_csv(DATA_PATH)
X = df[["test_cases", "code_churn_lines", "sprint_velocity", "is_regression_suite"]]
y = df["defect_count"]

# Mandatory split parameters per instructions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Challenger: Ridge with solver='sag' so random_state=99 is actually used
challenger_model = Ridge(alpha=1.0, solver="sag", random_state=99, max_iter=10000)

def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {
        "mae":  round(mae, 6),
        "rmse": round(rmse, 6),
        "r2":   round(r2, 6),
        "mape": round(mape, 6),
    }

with mlflow.start_run(run_name="Ridge_challenger"):
    mlflow.set_tag("experiment_type", "baseline_comparison")
    mlflow.log_params({"alpha": 1.0, "solver": "sag", "random_state": 99})

    challenger_model.fit(X_train, y_train)
    y_pred = challenger_model.predict(X_test)

    metrics = compute_metrics(y_test.values, y_pred)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(challenger_model, name="Ridge_challenger")

    challenger_run_id = mlflow.active_run().info.run_id

challenger_rmse = metrics["rmse"]
print(f"Challenger metrics: {metrics}  run_id={challenger_run_id}")

# ── Step 3: Register challenger as version 2 ──────────────────────────────────
challenger_model_uri = f"runs:/{challenger_run_id}/Ridge_challenger"
mv2 = mlflow.register_model(
    model_uri=challenger_model_uri,
    name=REGISTERED_MODEL_NAME,
)
challenger_version = int(mv2.version)
print(f"Challenger registered as version: {challenger_version}")

# ── Step 4: Compare and promote / keep ────────────────────────────────────────
print(f"\nChampion RMSE (v{champion_version}):  {champion_rmse}")
print(f"Challenger RMSE (v{challenger_version}): {challenger_rmse}")

if challenger_rmse < champion_rmse:
    # Challenger is better — move "live" alias to version 2
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS_NAME,
        version=str(challenger_version),
    )
    final_version = challenger_version
    action = "promoted"
    print(f"Challenger is BETTER -> alias '{ALIAS_NAME}' moved to v{challenger_version}")
else:
    # Champion holds — keep "live" on version 1
    final_version = champion_version
    action = "kept"
    print(f"Champion is BETTER   -> alias '{ALIAS_NAME}' stays on v{champion_version}")

# ── Step 5: Save output JSON ──────────────────────────────────────────────────
output = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "alias_name": ALIAS_NAME,
    "champion_version": champion_version,
    "challenger_version": challenger_version,
    "action": action,
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[DONE] Results saved -> {OUT_JSON}")
print(json.dumps(output, indent=2))
