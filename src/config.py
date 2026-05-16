"""Paths and MLflow settings for the retail price project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "online_retail_II.csv"
MLFLOW_DIR = PROJECT_ROOT / "mlflow_data"
TRACKING_DB = MLFLOW_DIR / "tracking.db"
ARTIFACT_ROOT = MLFLOW_DIR / "artifacts"

EXPERIMENT_NAME = "retail_unit_price"
TUNING_EXPERIMENT_NAME = "retail_unit_price_hyperopt"
MONITORING_EXPERIMENT_NAME = "retail_price_monitoring"
REGISTRY_MODEL_NAME = "retail_price_regressor"

# SQLite backend enables Model Registry without a separate server.
TRACKING_URI = f"sqlite:///{TRACKING_DB}"
