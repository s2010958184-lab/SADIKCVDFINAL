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
    """Reproduces the alert logic from §8.9 prevention_advisor() + enrichment flags."""
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
    e = p.get("_enrichment") or {}
    gender = int(p.get("gender", 1))

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

    # ── Enrichment-based flags (questionnaire steps 5–10) ─────────────────
    wc = e.get("waist_cm")
    if wc is not None:
        wc_f = float(wc)
        if gender == 2 and wc_f >= 102:
            flags.append({"level": "ALERT", "msg": f"Waist {wc_f:.0f} cm — central obesity threshold (men). Strong CVD/metabolic signal."})
        elif gender == 1 and wc_f >= 88:
            flags.append({"level": "ALERT", "msg": f"Waist {wc_f:.0f} cm — central obesity threshold (women). Strong CVD/metabolic signal."})
        elif (gender == 2 and wc_f >= 94) or (gender == 1 and wc_f >= 80):
            flags.append({"level": "WARN", "msg": f"Waist {wc_f:.0f} cm — elevated; track with BP and lipids."})

    hr = e.get("resting_hr")
    if hr is not None and float(hr) >= 90:
        flags.append({"level": "WARN", "msg": f"Resting heart rate {float(hr):.0f} bpm — on the high side; discuss with your clinician."})

    if e.get("hdl") == "low":
        flags.append({"level": "WARN", "msg": "Low HDL — discuss lifestyle and lipid targets with your doctor."})

    if e.get("prior_event") == 1:
        flags.append({"level": "URGENT", "msg": "Prior heart attack, stroke, or TIA — you are in secondary prevention. Follow cardiology care closely."})

    if e.get("chest_pain") == 1:
        flags.append({"level": "URGENT", "msg": "Chest discomfort or breathlessness on exertion — needs prompt medical evaluation (possible angina)."})
    if e.get("sleep_apnea") == 1:
        flags.append({"level": "ALERT", "msg": "Sleep apnea symptoms — strongly linked to hypertension and arrhythmia; consider formal testing."})
    if e.get("family_history") == 1:
        flags.append({"level": "INFO", "msg": "Family history of early heart disease — earlier screening and risk-factor control matter."})

    conds = set(e.get("conditions") or [])
    if "afib" in conds:
        flags.append({"level": "ALERT", "msg": "Atrial fibrillation — stroke prevention and rate control are essential; stay in cardiology follow-up."})
    if "diabetes" in conds:
        if not any("Glucose" in f["msg"] for f in flags):
            flags.append({"level": "ALERT", "msg": "Diabetes — tight BP, lipid, and glucose control dramatically lowers CVD risk."})
    if "kidney" in conds:
        flags.append({"level": "ALERT", "msg": "Kidney disease — cardiovascular risk is markedly elevated; coordinate care with your nephrologist/PCP."})

    return flags


def _enrichment_log_odds_delta(e: dict[str, Any], p: dict[str, Any]) -> float:
    """Extra log-odds from the expanded questionnaire (used only in rule-based mode).

    These terms make the fallback track user-reported risk factors that the core
    11-feature ML pipeline does not see. They are conservative but monotonic.
    """
    delta = 0.0
    gender = int(p.get("gender", 1))

    # Central adiposity (ATP III style thresholds)
    wc = e.get("waist_cm")
    if wc is not None:
        wc_f = float(wc)
        if gender == 2:  # Male in our schema: 2
            if wc_f >= 102:
                delta += 0.55
            elif wc_f >= 94:
                delta += 0.28
        else:  # Female
            if wc_f >= 88:
                delta += 0.55
            elif wc_f >= 80:
                delta += 0.28

    hr = e.get("resting_hr")
    if hr is not None:
        hr_f = float(hr)
        delta += 0.014 * max(0.0, hr_f - 72)  # ~0.14 per 10 bpm above 72
        delta -= 0.06 * max(0.0, 60 - hr_f)   # athletic bradycardia slight negative

    hdl = e.get("hdl")
    if hdl == "low":
        delta += 0.48
    elif hdl == "high":
        delta -= 0.22

    if e.get("bp_med") == 1:
        delta += 0.22
    if e.get("statin") == 1:
        delta += 0.12

    if e.get("prior_event") == 1:
        delta += 1.85
    if e.get("chest_pain") == 1:
        delta += 0.95
    if e.get("sleep_apnea") == 1:
        delta += 0.38
    if e.get("family_history") == 1:
        delta += 0.32

    conds = set(e.get("conditions") or [])
    if "diabetes" in conds:
        delta += 0.55
    if "hypertension" in conds:
        delta += 0.18
    if "kidney" in conds:
        delta += 0.45
    if "afib" in conds:
        delta += 0.85

    fv = e.get("fruit_veg")
    if fv == "0":
        delta += 0.18
    elif fv == "1-2":
        delta += 0.06
    elif fv in ("3-4",):
        delta -= 0.04
    elif fv == "5+":
        delta -= 0.14

    rm = e.get("red_meat")
    if rm == "daily":
        delta += 0.18
    elif rm == "weekly":
        delta += 0.05
    elif rm == "rarely":
        delta -= 0.05

    if e.get("salt_intake") == "high":
        delta += 0.14
    elif e.get("salt_intake") == "low":
        delta -= 0.05

    sd = e.get("sugary_drinks")
    if sd == "3+":
        delta += 0.14
    elif sd == "none":
        delta -= 0.05

    dp = e.get("diet_pattern")
    if dp == "processed":
        delta += 0.22
    elif dp == "mediterranean":
        delta -= 0.16
    elif dp in ("vegetarian", "vegan"):
        delta -= 0.08

    sit = e.get("sitting_hours")
    if sit == "gt8":
        delta += 0.22
    elif sit == "4-8":
        delta += 0.08
    elif sit == "lt4":
        delta -= 0.05

    sh = e.get("sleep_hours")
    if sh is not None:
        sh_f = float(sh)
        if sh_f < 6:
            delta += 0.12
        elif sh_f > 9:
            delta += 0.04

    if e.get("stress_level") == "high":
        delta += 0.14
    elif e.get("stress_level") == "low":
        delta -= 0.04

    if e.get("mental_health") == 1:
        delta += 0.14

    # Many imputed core fields → score is less person-specific; nudge toward neutral
    n_imputed = len(p.get("_imputed") or [])
    if n_imputed >= 6:
        delta += 0.08
    if n_imputed >= 9:
        delta += 0.12

    return float(delta)


def _rule_based_probability(p: dict[str, Any]) -> float:
    """Fallback probability when no trained model is available.

    Base linear score is calibrated so a reference adult lands near moderate
    risk; `_enrichment_log_odds_delta` adds terms for all optional questionnaire
    answers (waist, HR, HDL, meds, prior events, diet, etc.).
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

    e = p.get("_enrichment") or {}
    score += _enrichment_log_odds_delta(e, p)
    return float(1.0 / (1.0 + np.exp(-np.clip(score, -8.0, 8.0))))


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
