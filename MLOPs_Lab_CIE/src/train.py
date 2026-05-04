"""
Task 1 — Experiment Tracking & Model Comparison
- Trains LinearRegression and Ridge on training_data.csv
- Logs params, metrics (MAE, RMSE, R2, MAPE) and tag to MLflow
- Experiment name: "testforge-defect-count"
- Selects best model by RMSE (lower is better)
- Saves results to results/step1_s1.json
"""

import os
import json
import math
import pathlib
import joblib

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent          # MLOPs_Lab_CIE/
DATA_PATH    = ROOT / "data" / "training_data.csv"
RESULTS_DIR  = ROOT / "results"
MODELS_DIR   = ROOT / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ───────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "testforge-defect-count"
mlflow.set_tracking_uri(f"file:///{(ROOT / 'mlruns').as_posix()}")
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Load & split data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df[["test_cases", "code_churn_lines", "sprint_velocity", "is_regression_suite"]]
y = df["defect_count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── Metric helpers ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE — guard against zero targets
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"mae": round(mae, 6), "rmse": round(rmse, 6),
            "r2": round(r2, 6), "mape": round(mape, 6)}

# ── Models to compare ─────────────────────────────────────────────────────────
models = {
    "LinearRegression": (LinearRegression(), {}),
    "Ridge":            (Ridge(alpha=1.0, random_state=42), {"alpha": 1.0}),
}

results_models = []

for model_name, (model, params) in models.items():
    with mlflow.start_run(run_name=model_name):
        # Tag
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Log hyperparameters
        if params:
            mlflow.log_params(params)
        else:
            mlflow.log_param("fit_intercept", True)   # default LR param

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        metrics = compute_metrics(y_test.values, y_pred)
        mlflow.log_metrics(metrics)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        # Save model file to models/ directory
        model_file = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_file)
        print(f"Model saved -> {model_file}")

        run_id = mlflow.active_run().info.run_id
        print(f"{model_name}: {metrics}  run_id={run_id}")

    results_models.append({
        "name":  model_name,
        "mae":   metrics["mae"],
        "rmse":  metrics["rmse"],
        "r2":    metrics["r2"],
        "mape":  metrics["mape"],
    })

# ── Select best model by RMSE ─────────────────────────────────────────────────
best = min(results_models, key=lambda m: m["rmse"])

output = {
    "experiment_name": EXPERIMENT_NAME,
    "models": results_models,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"],
}

out_path = RESULTS_DIR / "step1_s1.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[DONE] Results saved -> {out_path}")
print(json.dumps(output, indent=2))
