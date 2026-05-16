"""Train and compare baseline regressors with MLflow tracking.

Entry-point alias kept at the project root for convenience.
All source modules live in src/; this file adds src/ to sys.path so imports resolve
whether you run it from the project root or from inside src/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on the path so `config` and `data_prep` are importable.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from config import ARTIFACT_ROOT, DATA_PATH, EXPERIMENT_NAME, MLFLOW_DIR, TRACKING_DB, TRACKING_URI
from data_prep import build_feature_preprocessor, train_validation_split

try:
    from xgboost import XGBRegressor

    _XGB_OK = True
except Exception:  # pragma: no cover - platform / libomp
    XGBRegressor = None  # type: ignore[misc, assignment]
    _XGB_OK = False


def _gradient_boosting_step(random_state: int = 42):
    if _XGB_OK:
        return (
            "xgboost",
            XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    return (
        "sklearn_hist_gbrt",
        HistGradientBoostingRegressor(
            max_depth=10,
            learning_rate=0.08,
            max_iter=200,
            l2_regularization=0.1,
            random_state=random_state,
        ),
    )


def _ensure_mlflow_dirs() -> None:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    prep = build_feature_preprocessor()
    gb_name, gb_estimator = _gradient_boosting_step(random_state)
    return {
        "ridge_linear": Pipeline(
            steps=[
                ("prep", prep),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", build_feature_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=None,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        gb_name: Pipeline(
            steps=[
                ("prep", build_feature_preprocessor()),
                ("model", gb_estimator),
            ]
        ),
    }


def main() -> None:
    _ensure_mlflow_dirs()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=ARTIFACT_ROOT.as_uri(),
        )
    else:
        experiment_id = existing.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    sample_env = os.environ.get("RETAIL_SAMPLE_FRAC", "0.25")
    sample_frac = float(sample_env) if sample_env else None
    if sample_frac >= 1:
        sample_frac = None

    X_train, X_val, y_train, y_val = train_validation_split(
        DATA_PATH, sample_frac=sample_frac, test_size=0.2, random_state=42
    )

    models = build_models()
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("target_transform", "log1p_price")
            mlflow.log_param("sample_frac", sample_frac if sample_frac is not None else 1.0)
            mlflow.log_param("model_name", name)
            mlflow.log_param("xgboost_runtime_available", _XGB_OK)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)
            m = _metrics(y_val, pred)
            mlflow.log_metrics(m)

            signature = mlflow.models.infer_signature(X_train, pipe.predict(X_train))
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                signature=signature,
                registered_model_name=None,
            )

    print("Done. Compare runs in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()
