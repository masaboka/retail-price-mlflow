"""Simulate post-deployment monitoring: batch scores + drift-style alerts logged to MLflow."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    ARTIFACT_ROOT,
    DATA_PATH,
    MLFLOW_DIR,
    MONITORING_EXPERIMENT_NAME,
    REGISTRY_MODEL_NAME,
    TRACKING_DB,
    TRACKING_URI,
)
from data_prep import FEATURE_COLS, engineer_features, load_raw, clean_transactions


def _ensure_mlflow_dirs() -> None:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)


def load_monitoring_batch(csv_path: str, sample_frac: float = 0.05, random_state: int = 7) -> pd.DataFrame:
    raw = load_raw(csv_path, sample_frac=sample_frac, random_state=random_state)
    clean = clean_transactions(raw)
    return engineer_features(clean)


def resolve_latest_model_uri() -> str:
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTRY_MODEL_NAME}'")
    if not versions:
        raise SystemExit(
            f"No registered model named {REGISTRY_MODEL_NAME!r}. "
            "Run register_model.py first or pass --model-uri runs:/<run_id>/model"
        )
    best = max(versions, key=lambda v: int(v.version))
    return f"models:/{REGISTRY_MODEL_NAME}/{best.version}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        default=os.environ.get("MLFLOW_MODEL_URI", ""),
        help="models:/name/version or runs:/id/model (default: latest registered version).",
    )
    args = parser.parse_args()

    _ensure_mlflow_dirs()
    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = args.model_uri.strip()
    if not model_uri:
        model_uri = resolve_latest_model_uri()
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(MONITORING_EXPERIMENT_NAME)
    if existing is None:
        experiment_id = client.create_experiment(
            MONITORING_EXPERIMENT_NAME,
            artifact_location=ARTIFACT_ROOT.as_uri(),
        )
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    batch = load_monitoring_batch(str(DATA_PATH), sample_frac=0.08, random_state=7)
    X = batch[FEATURE_COLS]
    y = np.log1p(batch["Price"].values.astype(np.float64))

    model = mlflow.pyfunc.load_model(model_uri)
    pred = model.predict(X)
    pred = np.asarray(pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    residual_mean = float(np.mean(np.abs(y - pred)))

    with mlflow.start_run(run_name=f"monitor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"):
        mlflow.set_tag("monitoring", "simulated_batch")
        mlflow.set_tag("model_uri", model_uri)
        mlflow.log_param("batch_rows", len(batch))
        mlflow.log_metrics(
            {
                "monitoring_rmse": rmse,
                "monitoring_mae": mae,
                "monitoring_mean_abs_residual": residual_mean,
            }
        )
        drift_proxy = float(np.std(pred) / (np.std(y) + 1e-9))
        mlflow.log_metric("pred_std_over_target_std", drift_proxy)

    print(f"Logged monitoring metrics: rmse={rmse:.4f} mae={mae:.4f} (uri={model_uri})")


if __name__ == "__main__":
    main()
