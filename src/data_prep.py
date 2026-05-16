"""Load and clean Online Retail II data; feature columns for price modeling."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw(
    csv_path: str | os.PathLike[str],
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load CSV. Optionally subsample rows for faster iteration (sample_frac in (0,1])."""
    df = pd.read_csv(csv_path, encoding="latin1", parse_dates=["InvoiceDate"], low_memory=False)
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid sales lines; drop cancellations (Invoice starting with C)."""
    out = df.copy()
    out = out.rename(columns={"Customer ID": "CustomerID"})
    inv = out["Invoice"].astype(str)
    out = out[~inv.str.startswith("C")]
    out = out.dropna(subset=["Description", "StockCode", "Country"])
    out = out[(out["Quantity"] > 0) & (out["Price"] > 0)]
    out["Description"] = out["Description"].astype(str).str.strip()
    out["StockCode"] = out["StockCode"].astype(str).str.strip()
    out["Country"] = out["Country"].astype(str)
    out = out[out["Description"].str.len() > 2]
    return out.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple product / context features used as model inputs."""
    x = df.copy()
    x["desc_len"] = x["Description"].str.len()
    x["desc_word_count"] = x["Description"].str.split().str.len().clip(upper=50)
    x["stock_prefix"] = x["StockCode"].str[:3]
    x["month"] = x["InvoiceDate"].dt.month.astype(np.int16)
    x["dow"] = x["InvoiceDate"].dt.dayofweek.astype(np.int16)
    x["hour"] = x["InvoiceDate"].dt.hour.astype(np.int16)
    return x


FEATURE_COLS = [
    "Quantity",
    "desc_len",
    "desc_word_count",
    "month",
    "dow",
    "hour",
    "Country",
    "stock_prefix",
]


def build_feature_preprocessor() -> ColumnTransformer:
    """Numeric scaling + categorical one-hot."""
    numeric = ["Quantity", "desc_len", "desc_word_count", "month", "dow", "hour"]
    categorical = ["Country", "stock_prefix"]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )


def train_validation_split(
    csv_path: str | os.PathLike[str],
    sample_frac: float | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return raw feature frames and log1p(price) targets for train/validation."""
    raw = load_raw(csv_path, sample_frac=sample_frac, random_state=random_state)
    clean = clean_transactions(raw)
    eng = engineer_features(clean)

    y = np.log1p(eng["Price"].values.astype(np.float64))
    X = eng[FEATURE_COLS]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
