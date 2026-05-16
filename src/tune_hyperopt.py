"""Hyperparameter search with nested MLflow runs (Hyperopt if installed, else RandomizedSearchCV)."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from config import ARTIFACT_ROOT, DATA_PATH, MLFLOW_DIR, TRACKING_DB, TRACKING_URI, TUNING_EXPERIMENT_NAME
from data_prep import build_feature_preprocessor, train_validation_split

try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

    _HAVE_HYPEROPT = True
except ModuleNotFoundError:  # pragma: no cover - missing hyperopt / setuptools
    _HAVE_HYPEROPT = False

try:
    from xgboost import XGBRegressor

    _XGB_OK = True
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore[misc, assignment]
    _XGB_OK = False


def _ensure_mlflow_dirs() -> None:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)


def _run_hyperopt_parent(
    X_train,
    X_val,
    y_train,
    y_val,
    sample_frac: float | None,
    max_evals: int,
) -> None:
    from xgboost import XGBRegressor as XGB

    space = {
        "max_depth": hp.quniform("max_depth", 3, 14, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "n_estimators": hp.quniform("n_estimators", 80, 400, 20),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
    }

    with mlflow.start_run(run_name="hyperopt_xgboost_parent"):
        mlflow.log_param("search_algorithm", "tpe")
        mlflow.log_param("hyperopt_max_evals", max_evals)
        mlflow.log_param("sample_frac", sample_frac if sample_frac is not None else 1.0)
        mlflow.log_param("tuner_backend", "hyperopt_xgboost")

        def objective(params: dict) -> dict:
            params = params.copy()
            params["max_depth"] = int(params["max_depth"])
            params["n_estimators"] = int(params["n_estimators"])
            params["min_child_weight"] = int(params["min_child_weight"])
            with mlflow.start_run(nested=True):
                model = XGB(
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42,
                    tree_method="hist",
                    **params,
                )
                pipe = Pipeline(
                    steps=[
                        ("prep", build_feature_preprocessor()),
                        ("model", model),
                    ]
                )
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_val)
                rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
                mlflow.log_params({str(k): float(v) for k, v in params.items()})
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(pipe, artifact_path="model")
                return {"loss": rmse, "status": STATUS_OK}

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=Trials(),
        )
        mlflow.log_params({f"best_{k}": float(v) for k, v in best.items()})


def _param_grid_xgb() -> list[dict]:
    rng = np.random.default_rng(42)
    grid = []
    for _ in range(200):
        grid.append(
            {
                "model__max_depth": int(rng.integers(3, 15)),
                "model__n_estimators": int(rng.integers(80, 401)),
                "model__learning_rate": float(10 ** rng.uniform(-2.0, -0.5)),
                "model__subsample": float(rng.uniform(0.6, 1.0)),
                "model__colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "model__min_child_weight": int(rng.integers(1, 11)),
            }
        )
    return grid


def _param_grid_hgb() -> list[dict]:
    rng = np.random.default_rng(43)
    grid = []
    for _ in range(200):
        grid.append(
            {
                "model__max_depth": int(rng.integers(3, 16)),
                "model__max_iter": int(rng.integers(80, 401)),
                "model__learning_rate": float(10 ** rng.uniform(-2.0, -0.5)),
                "model__l2_regularization": float(10 ** rng.uniform(-6.0, 0.0)),
                "model__min_samples_leaf": int(rng.integers(5, 81)),
            }
        )
    return grid


def _run_sklearn_random_search(
    X_train,
    X_val,
    y_train,
    y_val,
    sample_frac: float | None,
    n_iter: int,
) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    if _XGB_OK:
        base = XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        )
        param_grid = _param_grid_xgb()
        label = "randomizedsearch_xgboost"
    else:
        base = HistGradientBoostingRegressor(random_state=42)
        param_grid = _param_grid_hgb()
        label = "randomizedsearch_hist_gbrt"

    pipe = Pipeline(
        steps=[
            ("prep", build_feature_preprocessor()),
            ("model", base),
        ]
    )

    with mlflow.start_run(run_name=f"{label}_parent"):
        mlflow.log_param("search_algorithm", "parameter_sampler_loops")
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("sample_frac", sample_frac if sample_frac is not None else 1.0)
        mlflow.log_param("tuner_backend", label)

        rng = np.random.default_rng(42)
        replace = n_iter > len(param_grid)
        idx = rng.choice(len(param_grid), size=n_iter, replace=replace)
        sampled = [param_grid[i] for i in idx]
        for params in sampled:
            with mlflow.start_run(nested=True):
                pipe.set_params(**params)
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_val)
                rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
                flat = {k.split("__", 1)[1]: v for k, v in params.items()}
                mlflow.log_params({str(k): float(v) for k, v in flat.items()})
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(pipe, artifact_path="model")


def main() -> None:
    _ensure_mlflow_dirs()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(TUNING_EXPERIMENT_NAME)
    if existing is None:
        experiment_id = client.create_experiment(
            TUNING_EXPERIMENT_NAME,
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

    max_evals = int(os.environ.get("HYPEROPT_MAX_EVALS", "18"))

    if _HAVE_HYPEROPT and _XGB_OK:
        _run_hyperopt_parent(X_train, X_val, y_train, y_val, sample_frac, max_evals)
    else:
        _run_sklearn_random_search(X_train, X_val, y_train, y_val, sample_frac, max_evals)

    print("Tuning finished. Inspect nested runs under the parent run in MLflow UI.")


if __name__ == "__main__":
    main()
