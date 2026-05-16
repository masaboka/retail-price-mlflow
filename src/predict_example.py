"""Load a logged or registered sklearn pipeline and print a few predictions (log1p space)."""

from __future__ import annotations

import argparse
import os

import mlflow
import pandas as pd

from config import DATA_PATH, TRACKING_URI
from data_prep import FEATURE_COLS, engineer_features, load_raw, clean_transactions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-uri",
        default=os.environ.get("MLFLOW_MODEL_URI", ""),
        help="runs:/<id>/model or models:/<name>/<version>",
    )
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    if not args.model_uri.strip():
        raise SystemExit("Set --model-uri or MLFLOW_MODEL_URI, e.g. runs:/abc123/model")

    mlflow.set_tracking_uri(TRACKING_URI)
    raw = load_raw(DATA_PATH, sample_frac=0.002)
    batch = engineer_features(clean_transactions(raw))
    X = batch[FEATURE_COLS].head(args.rows)

    model = mlflow.pyfunc.load_model(args.model_uri.strip())
    pred = model.predict(X)
    out = X.copy()
    out["pred_log1p_price"] = pred
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
