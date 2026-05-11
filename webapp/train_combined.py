"""Phase 2 — multi-source CVD model training.

Trains the production web-app model on the *combined* dataset built across
Kaggle (sulianova), Framingham, BRFSS 2022, and NHANES 2017–2018. Mirrors the
§11–§14 pipeline in the main notebook so the web app can re-use the exact
same data preparation logic.

Design goals
------------
1. **Best effort.** Every loader is wrapped in a try/except so the script can
   train even if some sources are missing — useful for the user without
   Kaggle credentials or the 400 MB BRFSS SAS file.
2. **Schema harmonisation.** Every source is mapped to the eight features the
   web app submits via the questionnaire:
   age_years, gender, height_cm, weight_kg, ap_hi, ap_lo,
   cholesterol (1/2/3), gluc (1/2/3), smoke (0/1), alco (0/1), active (0/1).
3. **Best-of-N.** Trains LR, RandomForest, GradientBoosting (and XGBoost when
   installed) and keeps the highest validation-AUC pipeline.

Usage
-----
    python train_combined.py [--limit-brfss 50000]

Data files (any subset is fine):
    data/cardio_train.csv              # Kaggle Sulianova
    data/framingham.csv                # Framingham (UCI-style CSV)
    data/LLCP2022.XPT                  # BRFSS 2022 SAS XPT (optional)
    data/nhanes_2017_2018.csv          # NHANES harmonised CSV (optional)

The script writes models/cvd_model.pkl in the *same* schema as train_model.py
so model.py picks it up automatically — no other code changes required.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_PATH = HERE / "models" / "cvd_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NUM_COLS = ["age_years", "bmi", "ap_hi", "ap_lo", "pulse_pressure"]
CAT_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
TARGET = "cardio"
SCHEMA = NUM_COLS + CAT_COLS + [TARGET, "source"]


# ===========================================================================
# Loaders — each returns a DataFrame in the unified schema or None.
# ===========================================================================
def load_kaggle() -> pd.DataFrame | None:
    p = DATA_DIR / "cardio_train.csv"
    if not p.exists():
        print(f"[skip] kaggle: {p.name} not found")
        return None
    df = pd.read_csv(p, sep=";").drop(columns=["id"], errors="ignore")
    df["age_years"] = (df["age"] / 365.25).round(1)
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    df["height_cm"] = df["height"]
    df["weight_kg"] = df["weight"]
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df = df[[*NUM_COLS, *CAT_COLS, TARGET]].copy()
    df["source"] = "kaggle"
    return _drop_outliers(df)


def load_framingham() -> pd.DataFrame | None:
    p = DATA_DIR / "framingham.csv"
    if not p.exists():
        print(f"[skip] framingham: {p.name} not found")
        return None
    raw = pd.read_csv(p)
    raw.columns = [c.strip().lower() for c in raw.columns]
    target_col = "tenyearchd" if "tenyearchd" in raw.columns else (
        "ten_year_chd" if "ten_year_chd" in raw.columns else None
    )
    if target_col is None:
        print("[skip] framingham: no target column found")
        return None

    df = pd.DataFrame()
    df["age_years"] = raw["age"]
    df["gender"]    = raw["male"].map({0: 1, 1: 2})  # 1 female, 2 male — same as Kaggle
    df["height_cm"] = np.nan
    df["weight_kg"] = np.nan
    df["bmi"]       = raw.get("bmi", np.nan)
    df["ap_hi"]     = raw["sysbp"]
    df["ap_lo"]     = raw["diabp"]
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["cholesterol"] = pd.cut(
        raw["totchol"],
        bins=[-np.inf, 200, 240, np.inf],
        labels=[1, 2, 3],
    ).astype("Int64").fillna(1).astype(int)
    df["gluc"] = pd.cut(
        raw["glucose"],
        bins=[-np.inf, 100, 125, np.inf],
        labels=[1, 2, 3],
    ).astype("Int64").fillna(1).astype(int)
    df["smoke"]  = raw["currentsmoker"].fillna(0).astype(int)
    df["alco"]   = 0  # not measured
    df["active"] = 0  # not measured
    df[TARGET]   = raw[target_col].astype(int)
    df["source"] = "framingham"
    return _drop_outliers(df)


def load_brfss(limit: int | None) -> pd.DataFrame | None:
    p = DATA_DIR / "LLCP2022.XPT"
    if not p.exists():
        print(f"[skip] brfss: {p.name} not found")
        return None
    try:
        raw = pd.read_sas(p, format="xport", encoding="latin-1")
    except Exception as e:
        print(f"[skip] brfss: read_sas failed ({e})")
        return None
    if limit and len(raw) > limit:
        raw = raw.sample(limit, random_state=RANDOM_SEED).reset_index(drop=True)
    raw.columns = [c.strip().upper() for c in raw.columns]

    def code_chol(v):
        return 3 if v == 1 else (1 if v == 2 else np.nan)
    def code_smoke(v):
        return 1 if v in (1, 2) else (0 if v == 3 else np.nan)

    df = pd.DataFrame()
    df["age_years"] = raw["_AGEG5YR"].map({
        1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57,
        9: 62, 10: 67, 11: 72, 12: 77, 13: 82
    })
    df["gender"]    = raw["SEX"].map({1: 2, 2: 1})  # 1 female, 2 male
    df["height_cm"] = raw.get("HTM4", np.nan)       # HTM4 is height in cm * 100? actually it's cm
    df["weight_kg"] = raw.get("WTKG3", np.nan) / 100  # WTKG3 is kg * 100
    df["bmi"]       = raw.get("_BMI5", np.nan) / 100
    df["ap_hi"]     = np.where(raw.get("BPHIGH6", 0) == 1, 145, 125)
    df["ap_lo"]     = np.where(raw.get("BPHIGH6", 0) == 1, 92, 80)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["cholesterol"]    = raw.get("TOLDHI3", np.nan).map(code_chol).fillna(1).astype(int)
    df["gluc"]           = (raw.get("DIABETE4", np.nan).isin([1])).astype(int).replace({0: 1, 1: 3})
    df["smoke"]          = raw.get("SMOKE100", np.nan).map(code_smoke).fillna(0).astype(int)
    df["alco"]           = (raw.get("DRNKANY5", np.nan) == 1).astype(int)
    df["active"]         = (raw.get("EXERANY2", np.nan) == 1).astype(int)
    df[TARGET]           = (raw.get("CVDINFR4", np.nan) == 1).astype(int)
    df["source"]         = "brfss22"
    return _drop_outliers(df)


def load_nhanes() -> pd.DataFrame | None:
    p = DATA_DIR / "nhanes_2017_2018.csv"
    if not p.exists():
        print(f"[skip] nhanes: {p.name} not found")
        return None
    raw = pd.read_csv(p)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if not {"age", "gender", "bmi", "sbp", "dbp"}.issubset(raw.columns):
        print("[skip] nhanes: required columns missing")
        return None
    df = pd.DataFrame()
    df["age_years"] = raw["age"]
    df["gender"]    = raw["gender"]   # already 1/2
    df["bmi"]       = raw["bmi"]
    df["height_cm"] = raw.get("height_cm", np.nan)
    df["weight_kg"] = raw.get("weight_kg", np.nan)
    df["ap_hi"]     = raw["sbp"]
    df["ap_lo"]     = raw["dbp"]
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["cholesterol"]    = raw.get("cholesterol", 1).fillna(1).astype(int)
    df["gluc"]           = raw.get("gluc", 1).fillna(1).astype(int)
    df["smoke"]          = raw.get("smoke", 0).fillna(0).astype(int)
    df["alco"]           = raw.get("alco", 0).fillna(0).astype(int)
    df["active"]         = raw.get("active", 0).fillna(0).astype(int)
    df[TARGET]           = raw.get("cardio", 0).fillna(0).astype(int)
    df["source"]         = "nhanes17"
    return _drop_outliers(df)


def _drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["age_years", "ap_hi", "ap_lo", "bmi", TARGET])
    df = df[(df["age_years"].between(18, 95)) &
            (df["ap_hi"].between(60, 260)) &
            (df["ap_lo"].between(30, 200)) &
            (df["ap_hi"] > df["ap_lo"]) &
            (df["bmi"].between(10, 80))]
    for c in ("cholesterol", "gluc"):
        df[c] = df[c].clip(1, 3).astype(int)
    for c in ("smoke", "alco", "active", "gender"):
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True)


LOADERS: list[tuple[str, Callable[[], pd.DataFrame | None]]] = []


def build_combined(brfss_limit: int | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, loader in [
        ("kaggle",     load_kaggle),
        ("framingham", load_framingham),
        ("brfss",      lambda: load_brfss(brfss_limit)),
        ("nhanes",     load_nhanes),
    ]:
        try:
            f = loader()
        except Exception as e:
            print(f"[skip] {name}: loader raised {e}")
            continue
        if f is None or f.empty:
            continue
        print(f"[ok]   {name}: {len(f):,} rows, prevalence={f[TARGET].mean():.2%}")
        frames.append(f)

    if not frames:
        print("\n[ERROR] No dataset loaded. Place at least one of:")
        for f in ("cardio_train.csv", "framingham.csv", "LLCP2022.XPT", "nhanes_2017_2018.csv"):
            print(f"   - data/{f}")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[NUM_COLS + CAT_COLS + [TARGET, "source"]]
    return combined


# ===========================================================================
# Modelling
# ===========================================================================
def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])


def candidate_models() -> dict[str, object]:
    models: dict[str, object] = {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, solver="lbfgs", random_state=RANDOM_SEED,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_SEED,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            eval_metric="logloss", random_state=RANDOM_SEED,
        )
    except ImportError:
        print("[info] xgboost not installed — skipping XGBoost candidate")
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HeartGuard model on combined datasets")
    ap.add_argument("--limit-brfss", type=int, default=80_000,
                    help="Sample limit for BRFSS to keep training tractable.")
    args = ap.parse_args()

    print("=" * 60)
    print(" HeartGuard — combined-dataset training")
    print("=" * 60)

    df = build_combined(args.limit_brfss)
    print(f"\nCombined dataset: {len(df):,} rows")
    print(df["source"].value_counts().to_string())
    print(f"Overall CVD prevalence: {df[TARGET].mean():.2%}\n")

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED,
    )

    pre = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    best_name = None
    best_auc  = -np.inf
    best_pipe = None
    results: list[dict] = []

    for name, clf in candidate_models().items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        print(f"--- Training {name} ---")
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        val_auc = roc_auc_score(y_va, proba)
        print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   Val ROC-AUC: {val_auc:.4f}")
        results.append({
            "model": name,
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std":  float(cv_scores.std()),
            "val_auc":     float(val_auc),
        })
        if val_auc > best_auc:
            best_auc, best_name, best_pipe = val_auc, name, pipe

    assert best_pipe is not None
    print("\n" + "=" * 60)
    print(f" Best model: {best_name}  (Val AUC = {best_auc:.4f})")
    print("=" * 60)

    joblib.dump({
        "pipeline": best_pipe,
        "auc": float(best_auc),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "trained_on": "combined: " + ",".join(sorted(df["source"].unique())),
        "results": results,
        "rows": int(len(df)),
        "best_model": best_name,
    }, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
