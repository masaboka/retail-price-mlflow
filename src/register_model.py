"""Register the best run from an experiment in the MLflow Model Registry."""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from config import (
    ARTIFACT_ROOT,
    EXPERIMENT_NAME,
    MLFLOW_DIR,
    REGISTRY_MODEL_NAME,
    TRACKING_DB,
    TRACKING_URI,
)


def _ensure_mlflow_dirs() -> None:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)


def pick_best_run(client: MlflowClient, experiment_name: str, metric: str = "rmse") -> str:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"No experiment named {experiment_name!r}. Run train_experiments.py first.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No runs found for that experiment.")
    return runs[0].info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Register best sklearn model from MLflow runs.")
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT", EXPERIMENT_NAME),
        help="Experiment name to search (default: retail_unit_price).",
    )
    parser.add_argument("--metric", default="rmse", help="Metric to minimize (default: rmse).")
    parser.add_argument("--stage", default="None", help="Optional: staging or production after register.")
    args = parser.parse_args()

    _ensure_mlflow_dirs()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    run_id = pick_best_run(client, args.experiment, args.metric)
    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=REGISTRY_MODEL_NAME)
    except MlflowException as e:
        msg = str(e).lower()
        if "readonly" in msg or "read-only" in msg:
            raise SystemExit(
                "MLflow could not write to the SQLite store (readonly database). "
                "On macOS, ensure the project folder is writable, remove stale "
                "mlflow_data/tracking.db-wal and tracking.db-shm if present, or "
                "delete mlflow_data/ and re-run train_experiments.py."
            ) from e
        raise
    print(f"Registered model version {mv.version} from run {run_id} uri {model_uri}")

    if args.stage and args.stage.lower() != "none":
        client.transition_model_version_stage(
            name=REGISTRY_MODEL_NAME,
            version=int(mv.version),
            stage=args.stage.capitalize(),
        )
        print(f"Transitioned version {mv.version} to {args.stage}.")


if __name__ == "__main__":
    main()
