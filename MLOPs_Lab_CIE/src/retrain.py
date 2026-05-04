"""
Task 4 - Retraining Pipeline
- Combines training_data.csv + new_data.csv
- Retrains the same model type that won Task 1 (Ridge)
- Evaluates both champion and retrained on the SAME original test set
- Promotes if retrained MAE < champion MAE (any improvement)
- Registers retrained model as version 3
- Saves results to results/step4_s8.json
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
ROOT          = pathlib.Path(__file__).resolve().parent.parent
TRAIN_CSV     = ROOT / "data" / "training_data.csv"
NEW_CSV       = ROOT / "data" / "new_data.csv"
RESULTS_DIR   = ROOT / "results"
OUT_JSON      = RESULTS_DIR / "step4_s8.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup (same tracking URI as all previous tasks) ────────────────────
TRACKING_URI          = f"file:///{(ROOT / 'mlruns').as_posix()}"
EXPERIMENT_NAME       = "testforge-defect-count"
REGISTERED_MODEL_NAME = "testforge-defect-count-predictor"
ALIAS_NAME            = "live"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri=TRACKING_URI)

FEATURES = ["test_cases", "code_churn_lines", "sprint_velocity", "is_regression_suite"]
TARGET   = "defect_count"

# ── Metric helper ─────────────────────────────────────────────────────────────
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

# ── Step 1: Load datasets ─────────────────────────────────────────────────────
# pandas skip_blank_lines=True (default) — blank row in new_data.csv is skipped
df_train   = pd.read_csv(TRAIN_CSV)
df_new     = pd.read_csv(NEW_CSV)          # blank line skipped automatically

original_data_rows = len(df_train)         # 25
new_data_rows      = len(df_new)           # 20 (blank line auto-skipped)
combined_data_rows = original_data_rows + new_data_rows  # 45

print(f"Original rows : {original_data_rows}")
print(f"New data rows : {new_data_rows}")
print(f"Combined rows : {combined_data_rows}")

# ── Step 2: Fix the SAME test set as Task 1 ───────────────────────────────────
# Split original training data (rs=42, ts=0.2) -> identical test set to Task 1
X_orig = df_train[FEATURES]
y_orig = df_train[TARGET]

X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42
)
print(f"Original train: {len(X_train_orig)} rows | Test set: {len(X_test)} rows (FIXED)")

# ── Step 3: Get champion MAE from MLflow (alias "live" = version 2) ───────────
champion_version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_NAME)
champion_version_num  = int(champion_version_info.version)
champion_run_id       = champion_version_info.run_id
champion_run          = client.get_run(champion_run_id)
champion_mae          = round(champion_run.data.metrics["mae"], 6)
print(f"Champion: version={champion_version_num}  run_id={champion_run_id}  MAE={champion_mae}")

# ── Step 4: Build combined training set ──────────────────────────────────────
# Train on original X_train (20 rows) + ALL of new_data (20 rows) = 40 rows
X_new = df_new[FEATURES]
y_new = df_new[TARGET]

X_retrain = pd.concat([X_train_orig, X_new], ignore_index=True)
y_retrain = pd.concat([y_train_orig, y_new], ignore_index=True)
print(f"Retrain set size: {len(X_retrain)} rows")

# ── Step 5: Train retrained model (same type as Task 1 winner: Ridge) ─────────
retrained_model = Ridge(alpha=1.0)   # same type that won Task 1

with mlflow.start_run(run_name="Ridge_retrained"):
    mlflow.set_tag("experiment_type", "retraining")
    mlflow.set_tag("trigger", "new_data_arrival")
    mlflow.log_params({
        "alpha": 1.0,
        "original_rows": original_data_rows,
        "new_rows": new_data_rows,
        "combined_rows": combined_data_rows,
    })

    retrained_model.fit(X_retrain, y_retrain)
    y_pred = retrained_model.predict(X_test)

    metrics = compute_metrics(y_test.values, y_pred)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(retrained_model, name="Ridge_retrained")

    retrained_run_id = mlflow.active_run().info.run_id

retrained_mae = metrics["mae"]
print(f"Retrained model metrics: {metrics}  run_id={retrained_run_id}")

# ── Step 6: Register retrained model as version 3 ─────────────────────────────
retrained_model_uri = f"runs:/{retrained_run_id}/Ridge_retrained"
mv3 = mlflow.register_model(
    model_uri=retrained_model_uri,
    name=REGISTERED_MODEL_NAME,
)
retrained_version = int(mv3.version)
print(f"Retrained model registered as version: {retrained_version}")

# ── Step 7: Compare MAE and promote/keep ─────────────────────────────────────
improvement = round(champion_mae - retrained_mae, 6)   # positive = retrained is better
min_improvement_threshold = 0

print(f"\nChampion MAE  (v{champion_version_num}): {champion_mae}")
print(f"Retrained MAE (v{retrained_version}): {retrained_mae}")
print(f"Improvement  : {improvement}  (threshold: >{min_improvement_threshold})")

if improvement > min_improvement_threshold:
    # Retrained is better — move "live" alias to retrained version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS_NAME,
        version=str(retrained_version),
    )
    action = "promoted"
    print(f"Retrained is BETTER -> alias '{ALIAS_NAME}' moved to v{retrained_version}")
else:
    # Champion holds — keep "live" on current champion
    action = "kept_champion"
    print(f"Champion is BETTER  -> alias '{ALIAS_NAME}' stays on v{champion_version_num}")

# ── Step 8: Save output JSON ──────────────────────────────────────────────────
output = {
    "original_data_rows":      original_data_rows,
    "new_data_rows":           new_data_rows,
    "combined_data_rows":      combined_data_rows,
    "champion_mae":            champion_mae,
    "retrained_mae":           retrained_mae,
    "improvement":             improvement,
    "min_improvement_threshold": min_improvement_threshold,
    "action":                  action,
    "comparison_metric":       "mae",
}

with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[DONE] Results saved -> {OUT_JSON}")
print(json.dumps(output, indent=2))
