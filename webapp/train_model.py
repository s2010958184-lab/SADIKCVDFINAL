"""Train the CVD risk model from the Kaggle sulianova cardiovascular dataset
and save it to models/cvd_model.pkl. Mirrors the pipeline in §8.5 of the notebook.

Usage:
    python train_model.py
The CSV must be at data/cardio_train.csv (semicolon-separated, downloaded from
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
TEST_SIZE = 0.20
NUM_COLS = ["age_years", "bmi", "ap_hi", "ap_lo", "pulse_pressure"]
CAT_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
TARGET = "cardio"

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "cardio_train.csv"
MODEL_PATH = HERE / "models" / "cvd_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, sep=";")
    df = raw.drop(columns=["id"], errors="ignore").copy()
    df["age_years"] = (df["age"] / 365.25).round(1)
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df = df.drop(columns=["age", "height", "weight"])

    df = df[(df["ap_hi"] >= 60) & (df["ap_hi"] <= 250)]
    df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 200)]
    df = df[df["ap_hi"] > df["ap_lo"]]
    df = df[df["pulse_pressure"] > 0]
    df = df[(df["age_years"] >= 18) & (df["age_years"] <= 90)]
    df = df[(df["bmi"] >= 10) & (df["bmi"] <= 80)]
    return df.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, solver="lbfgs", random_state=RANDOM_SEED,
        )),
    ])


def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        print("Download cardio_train.csv from:")
        print("  https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
        print(f"Then place it at {DATA_PATH}")
        sys.exit(1)

    print(f"Loading dataset: {DATA_PATH}")
    df = load_and_clean(DATA_PATH)
    print(f"Cleaned rows: {len(df):,}")

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    pipe = build_pipeline()
    print("Fitting Logistic Regression pipeline...")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, proba)
    print(f"Validation ROC-AUC: {auc:.4f}")

    joblib.dump({
        "pipeline": pipe,
        "auc": float(auc),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "trained_on": str(DATA_PATH.name),
    }, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
