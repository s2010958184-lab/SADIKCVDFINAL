"""Risk prediction layer.

Loads the trained Logistic Regression pipeline from models/cvd_model.pkl
when available. If not available, falls back to a rule-based risk score
that mirrors the prevention_advisor() flag logic from §8.9 of the notebook.

This module exposes:
    predict_risk(patient_dict) -> dict with probability, tier, flags
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "cvd_model.pkl"

NUM_COLS = ["age_years", "bmi", "ap_hi", "ap_lo", "pulse_pressure"]
CAT_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


@dataclass
class RiskAssessment:
    probability: float
    tier: str
    tier_color: str
    tier_emoji: str
    tier_summary: str
    flags: list[dict[str, str]] = field(default_factory=list)
    engine: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 4),
            "probability_pct": round(self.probability * 100, 1),
            "tier": self.tier,
            "tier_color": self.tier_color,
            "tier_emoji": self.tier_emoji,
            "tier_summary": self.tier_summary,
            "flags": self.flags,
            "engine": self.engine,
        }


_MODEL_CACHE: dict[str, Any] | None = None


def _load_model() -> dict[str, Any] | None:
    """Load the saved pipeline once; return None if no .pkl is present."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        return None
    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        return _MODEL_CACHE
    except Exception as e:
        print(f"[model] Failed to load {MODEL_PATH}: {e}")
        return None


def model_available() -> bool:
    return _load_model() is not None


def model_metadata() -> dict[str, Any]:
    m = _load_model()
    if m is None:
        return {"available": False}
    return {
        "available": True,
        "auc": m.get("auc"),
        "trained_on": m.get("trained_on"),
    }


def get_tier(prob: float) -> tuple[str, str, str, str]:
    """Map probability → (tier, hex color, emoji, one-line summary)."""
    if prob <= 0.30:
        return (
            "Low Risk",
            "#10b981",
            "🟢",
            "Low CVD risk. Focus on maintaining and building cardiovascular fitness.",
        )
    if prob <= 0.55:
        return (
            "Moderate Risk",
            "#f59e0b",
            "🟡",
            "Moderate CVD risk. Structured aerobic exercise is the single most evidence-backed intervention.",
        )
    if prob <= 0.75:
        return (
            "High Risk",
            "#f97316",
            "🟠",
            "High CVD risk. Exercise is still strongly recommended, but must be structured and physician-cleared.",
        )
    return (
        "Very High Risk",
        "#ef4444",
        "🔴",
        "Very high CVD risk. A formal Cardiac Rehabilitation programme is required before independent exercise.",
    )


def _flags_from_patient(p: dict[str, Any]) -> list[dict[str, str]]:
    """Reproduces the alert logic from §8.9 prevention_advisor()."""
    flags = []
    ap_hi = float(p["ap_hi"])
    ap_lo = float(p["ap_lo"])
    bmi = float(p["bmi"])
    chol = int(p["cholesterol"])
    gluc = int(p["gluc"])
    smoke = int(p["smoke"])
    alco = int(p["alco"])
    active = int(p["active"])
    pp = ap_hi - ap_lo
    age = float(p["age_years"])

    if ap_hi >= 160:
        flags.append({"level": "URGENT", "msg": f"Systolic BP {ap_hi:.0f} mmHg — Stage 2 hypertension. Immediate medical review."})
    elif ap_hi >= 140:
        flags.append({"level": "ALERT", "msg": f"Systolic BP {ap_hi:.0f} mmHg — Stage 1 hypertension. Lifestyle intervention indicated."})
    elif ap_hi >= 130:
        flags.append({"level": "WARN", "msg": f"Systolic BP {ap_hi:.0f} mmHg — elevated. Reverse with lifestyle changes."})

    if bmi >= 35:
        flags.append({"level": "ALERT", "msg": f"BMI {bmi:.1f} — Obese Class II+. Weight management is top priority."})
    elif bmi >= 30:
        flags.append({"level": "WARN", "msg": f"BMI {bmi:.1f} — Obese. Target 5% weight loss over 3 months."})
    elif bmi >= 25:
        flags.append({"level": "INFO", "msg": f"BMI {bmi:.1f} — Overweight. Modest weight reduction will lower BP and cholesterol."})

    if chol == 3:
        flags.append({"level": "ALERT", "msg": "Cholesterol well above normal. Lipid panel + statin discussion needed."})
    elif chol == 2:
        flags.append({"level": "WARN", "msg": "Cholesterol above normal. Reduce saturated fat, increase soluble fibre."})

    if gluc == 3:
        flags.append({"level": "ALERT", "msg": "Glucose well above normal — likely diabetes. CVD risk doubled."})
    elif gluc == 2:
        flags.append({"level": "WARN", "msg": "Glucose above normal — pre-diabetes range. Reversible with exercise + diet."})

    if smoke:
        flags.append({"level": "ALERT", "msg": "Smoker — doubles CVD risk. Quitting is the single highest-impact intervention."})
    if alco:
        flags.append({"level": "WARN", "msg": "Alcohol intake — raises BP and triglycerides. Minimise."})
    if not active:
        flags.append({"level": "ALERT", "msg": "Physically inactive. Start 150 min/week aerobic exercise immediately."})

    if pp > 60:
        flags.append({"level": "INFO", "msg": f"Pulse pressure {pp:.0f} mmHg suggests arterial stiffness."})
    if age >= 60:
        flags.append({"level": "INFO", "msg": f"Age {age:.0f} — CVD rate rises steeply in the 60+ group."})

    return flags


def _rule_based_probability(p: dict[str, Any]) -> float:
    """Fallback probability when no trained model is available.

    Coefficients are loosely calibrated to the §8.6 logistic-regression
    feature importance plot. The intercept is chosen so that an 'average'
    Kaggle patient (age 53, BMI 27, BP 128/82, chol=1, gluc=1, no smoke,
    no alcohol, active, female) lands around the dataset's 50% baseline.
    """
    score = -0.30
    score += 0.045 * (float(p["age_years"]) - 50)
    score += 0.035 * (float(p["ap_hi"]) - 130)
    score += 0.025 * (float(p["ap_lo"]) - 85)
    score += 0.060 * (float(p["bmi"]) - 25)
    score += 0.55 * (int(p["cholesterol"]) - 1)
    score += 0.35 * (int(p["gluc"]) - 1)
    score += 0.45 * int(p["smoke"])
    score += 0.20 * int(p["alco"])
    score -= 0.35 * int(p["active"])
    if int(p["gender"]) == 2:
        score += 0.15
    return float(1.0 / (1.0 + np.exp(-score)))


def predict_risk(patient: dict[str, Any]) -> RiskAssessment:
    """Predict CVD risk for one patient.

    patient keys:
        age_years, gender (1=F, 2=M), height_cm, weight_kg, ap_hi, ap_lo,
        cholesterol (1/2/3), gluc (1/2/3), smoke (0/1), alco (0/1), active (0/1)
    """
    # Derive BMI + pulse pressure if not provided
    p = dict(patient)
    if "bmi" not in p and {"height_cm", "weight_kg"} <= p.keys():
        h_m = float(p["height_cm"]) / 100.0
        p["bmi"] = round(float(p["weight_kg"]) / (h_m * h_m), 1)
    p.setdefault("pulse_pressure", float(p["ap_hi"]) - float(p["ap_lo"]))

    flags = _flags_from_patient(p)

    m = _load_model()
    if m is not None:
        row = pd.DataFrame([{c: p[c] for c in NUM_COLS + CAT_COLS}])
        proba = float(m["pipeline"].predict_proba(row)[0, 1])
        engine = "model"
    else:
        proba = _rule_based_probability(p)
        engine = "rule"

    tier, color, emoji, summary = get_tier(proba)
    return RiskAssessment(
        probability=proba,
        tier=tier,
        tier_color=color,
        tier_emoji=emoji,
        tier_summary=summary,
        flags=flags,
        engine=engine,
    )
