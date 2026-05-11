"""CVD Risk web app — Flask entry point.

Routes
------
GET  /                landing page
GET  /assess          questionnaire form
POST /assess          submit + render results
POST /chat            JSON endpoint for the Ask-AI box (RAG over knowledge.py)
GET  /health          JSON status (model + ollama)

Student : Ahmed Al Sadik (B00983817)
Project : DSA502 — CVD Risk Prediction & Prevention
"""
from __future__ import annotations

import datetime as _dt
import os
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from db import (
    delete_assessment,
    ensure_session_id,
    get_assessment,
    init_db,
    list_assessments,
    save_assessment,
)
from knowledge import icon_name, plan_for_tier
from model import model_available, model_metadata, predict_risk
from ollama_client import ask_ollama, is_available
from pdf_report import build_pdf_report
from retriever import build_context, retrieve

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = _dt.timedelta(days=365)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

init_db()

# Expose icon_name() inside Jinja templates so {% if icon_name(emoji) %} works.
app.jinja_env.globals["icon_name"] = icon_name


@app.template_filter("dt")
def _dt_filter(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(float(epoch)).strftime("%d %b %Y, %H:%M")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MIN_ANSWERED = 10
TOTAL_QUESTIONS = 30

# Population-median defaults used when a core model field was skipped.
# Chosen to land an "average adult" near the dataset's 50% baseline.
_DEFAULTS: dict[str, Any] = {
    "age_years": 50.0,
    "gender": 1,        # female (slight majority in Kaggle cohort)
    "height_cm": 165.0,
    "weight_kg": 75.0,
    "ap_hi": 120.0,
    "ap_lo": 80.0,
    "cholesterol": 1,   # normal
    "gluc": 1,          # normal
    "smoke": 0,
    "alco": 0,
    "active": 1,
}

_CORE_FIELDS: tuple[str, ...] = tuple(_DEFAULTS.keys())


def _opt_float(form: dict, name: str, lo: float, hi: float) -> tuple[float | None, bool]:
    """Return (value, was_invalid). Empty value → (None, False)."""
    raw = (form.get(name) or "").strip()
    if not raw:
        return None, False
    try:
        v = float(raw)
    except ValueError:
        return None, True
    if not (lo <= v <= hi):
        return None, True
    return v, False


def _opt_int_in(form: dict, name: str, allowed: set[int]) -> tuple[int | None, bool]:
    raw = (form.get(name) or "").strip()
    if not raw:
        return None, False
    try:
        v = int(raw)
    except ValueError:
        return None, True
    if v not in allowed:
        return None, True
    return v, False


def _opt_bool(form: dict, name: str) -> int | None:
    """Yes/No radio chips: '1' → 1, '0' → 0, missing → None (skipped)."""
    raw = (form.get(name) or "").strip()
    if raw == "1":
        return 1
    if raw == "0":
        return 0
    return None


def _parse_questionnaire(form) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the 18-question form. Every field is optional; we just need ≥7.

    Returns ``(patient_dict, None)`` on success, or ``(None, error_msg)`` on failure.
    Missing core model fields are imputed with population medians and recorded
    in ``patient['_imputed']`` so the UI can flag them.
    """
    raw: dict[str, Any] = {}
    invalid: list[str] = []

    def _take(name, value, was_invalid, label):
        if was_invalid:
            invalid.append(label)
        raw[name] = value

    _take("age_years",   *_opt_float(form, "age_years", 12, 110), "Age")
    _take("gender",      *_opt_int_in(form, "gender", {1, 2}),    "Sex")
    _take("height_cm",   *_opt_float(form, "height_cm", 80, 230), "Height")
    _take("weight_kg",   *_opt_float(form, "weight_kg", 25, 300), "Weight")
    _take("ap_hi",       *_opt_float(form, "ap_hi", 60, 260),     "Systolic BP")
    _take("ap_lo",       *_opt_float(form, "ap_lo", 30, 200),     "Diastolic BP")
    _take("cholesterol", *_opt_int_in(form, "cholesterol", {1, 2, 3}), "Cholesterol")
    _take("gluc",        *_opt_int_in(form, "gluc",        {1, 2, 3}), "Glucose")
    raw["smoke"]  = _opt_bool(form, "smoke")
    raw["alco"]   = _opt_bool(form, "alco")
    raw["active"] = _opt_bool(form, "active")

    if invalid:
        return None, (
            "These values look out of range — please correct them or skip the field: "
            + ", ".join(invalid) + "."
        )

    # BP sanity (only if both supplied)
    if raw["ap_hi"] is not None and raw["ap_lo"] is not None and raw["ap_hi"] <= raw["ap_lo"]:
        return None, "Systolic BP must be higher than diastolic BP."

    # ── Optional enrichment fields (not used by the trained model) ─────────────
    sleep_hours, sleep_bad = _opt_float(form, "sleep_hours", 3, 14)
    if sleep_bad:
        return None, "Sleep hours look out of range — please use a value between 3 and 14."

    waist_cm, waist_bad = _opt_float(form, "waist_cm", 40, 200)
    if waist_bad:
        return None, "Waist circumference looks out of range — please use a value between 40 and 200 cm."

    resting_hr, hr_bad = _opt_float(form, "resting_hr", 30, 220)
    if hr_bad:
        return None, "Resting heart rate looks out of range — please use a value between 30 and 220 bpm."

    def _opt_choice(name, allowed):
        v = (form.get(name) or "").strip().lower()
        return v if v in allowed else None

    enrichment = {
        # — diet & lifestyle —
        "diet_pattern":   _opt_choice("diet_pattern",
                                      {"mediterranean", "balanced", "processed", "vegetarian", "vegan"}),
        "salt_intake":    _opt_choice("salt_intake", {"low", "medium", "high"}),
        "sugary_drinks":  _opt_choice("sugary_drinks", {"none", "1-2", "3+"}),
        "fruit_veg":      _opt_choice("fruit_veg", {"0", "1-2", "3-4", "5+"}),
        "red_meat":       _opt_choice("red_meat", {"rarely", "weekly", "daily"}),
        # — sleep & sitting —
        "sleep_hours":    sleep_hours,
        "sleep_apnea":    _opt_bool(form, "sleep_apnea"),
        "sitting_hours":  _opt_choice("sitting_hours", {"lt4", "4-8", "gt8"}),
        # — body composition + autonomic —
        "waist_cm":       waist_cm,
        "resting_hr":     resting_hr,
        # — labs (extra) —
        "hdl":            _opt_choice("hdl", {"low", "normal", "high"}),
        # — medications —
        "bp_med":         _opt_bool(form, "bp_med"),
        "statin":         _opt_bool(form, "statin"),
        # — history & symptoms —
        "family_history": _opt_bool(form, "family_history"),
        "prior_event":    _opt_bool(form, "prior_event"),
        "chest_pain":     _opt_bool(form, "chest_pain"),
        # — mental health & wellbeing —
        "stress_level":   _opt_choice("stress_level", {"low", "medium", "high"}),
        "mental_health":  _opt_bool(form, "mental_health"),
        # — multi-select conditions —
        "conditions":     [c for c in form.getlist("condition")
                           if c in {"diabetes", "hypertension", "kidney", "afib", "none"}],
    }

    # ── Count answered questions ───────────────────────────────────────────────
    answered = 0
    for f in _CORE_FIELDS:
        if raw.get(f) is not None:
            answered += 1
    for k, v in enrichment.items():
        if k == "conditions":
            if v:
                answered += 1
        elif v is not None:
            answered += 1

    if answered < MIN_ANSWERED:
        return None, (
            f"Please answer at least {MIN_ANSWERED} of the {TOTAL_QUESTIONS} questions to see your risk. "
            f"You answered {answered}."
        )

    # ── Impute missing core fields ─────────────────────────────────────────────
    imputed: list[str] = []
    patient: dict[str, Any] = {}
    for f in _CORE_FIELDS:
        if raw[f] is None:
            patient[f] = _DEFAULTS[f]
            imputed.append(f)
        else:
            patient[f] = raw[f]

    patient["_imputed"] = imputed
    patient["_provided_count"] = answered
    patient["_enrichment"] = enrichment
    return patient, None


# Pretty labels for imputation banner / patient summary
_IMPUTED_LABEL = {
    "age_years": "age",
    "gender": "sex",
    "height_cm": "height",
    "weight_kg": "weight",
    "ap_hi": "systolic BP",
    "ap_lo": "diastolic BP",
    "cholesterol": "cholesterol",
    "gluc": "glucose",
    "smoke": "smoking status",
    "alco": "alcohol use",
    "active": "physical activity",
}


def _patient_summary(patient: dict[str, Any], assessment) -> str:
    chol_labels = {1: "normal", 2: "above normal", 3: "well above normal"}
    gluc_labels = {1: "normal", 2: "above normal", 3: "well above normal"}
    sex_labels  = {1: "female", 2: "male"}

    imputed = set(patient.get("_imputed") or [])
    def _mark(name: str, text: str) -> str:
        return f"{text} (estimated)" if name in imputed else text

    bmi = patient.get("bmi") or round(
        patient["weight_kg"] / ((patient["height_cm"] / 100) ** 2), 1
    )
    bmi_estimated = "height_cm" in imputed or "weight_kg" in imputed

    parts = [
        _mark("age_years", f"Age {patient['age_years']:.0f}"),
        _mark("gender",    f"sex {sex_labels[patient['gender']]}"),
        ("BMI " + (f"{bmi:.1f} (estimated)" if bmi_estimated else f"{bmi:.1f}")),
        _mark("ap_hi",     f"blood pressure {patient['ap_hi']:.0f}/{patient['ap_lo']:.0f} mmHg"),
        _mark("cholesterol", f"cholesterol {chol_labels[patient['cholesterol']]}"),
        _mark("gluc",        f"glucose {gluc_labels[patient['gluc']]}"),
        _mark("smoke",  "smoker" if patient["smoke"]  else "non-smoker"),
        _mark("alco",   "drinks alcohol" if patient["alco"] else "no alcohol"),
        _mark("active", "physically active" if patient["active"] else "physically inactive"),
    ]
    summary = "; ".join(parts) + "."

    # Append enrichment signals if the user gave them
    e = patient.get("_enrichment") or {}
    extras: list[str] = []

    # body composition + autonomic
    if e.get("waist_cm") is not None:
        extras.append(f"waist circumference {e['waist_cm']:.0f} cm")
    if e.get("resting_hr") is not None:
        extras.append(f"resting HR {e['resting_hr']:.0f} bpm")

    # labs (extra)
    if e.get("hdl"):
        extras.append(f"HDL cholesterol: {e['hdl']}")

    # medications (important context for any BP / lipid reading)
    if e.get("bp_med") is not None:
        extras.append("on BP medication" if e["bp_med"] else "not on BP medication")
    if e.get("statin") is not None:
        extras.append("on statin / lipid-lowering medication" if e["statin"] else "not on lipid-lowering medication")

    # diet
    if e.get("diet_pattern"):    extras.append(f"diet pattern: {e['diet_pattern']}")
    if e.get("salt_intake"):     extras.append(f"salt intake: {e['salt_intake']}")
    if e.get("sugary_drinks"):   extras.append(f"sugary drinks/day: {e['sugary_drinks']}")
    if e.get("fruit_veg"):       extras.append(f"fruit+veg servings/day: {e['fruit_veg']}")
    if e.get("red_meat"):        extras.append(f"red/processed meat: {e['red_meat']}")

    # sleep & sitting
    if e.get("sleep_hours") is not None:
        extras.append(f"sleeps ~{e['sleep_hours']:.1f} h/night")
    if e.get("sleep_apnea") is not None:
        extras.append("sleep apnea / heavy snoring + daytime fatigue" if e["sleep_apnea"]
                      else "no sleep-apnea symptoms")
    if e.get("sitting_hours"):
        _sit_label = {"lt4": "< 4 h", "4-8": "4–8 h", "gt8": "> 8 h"}.get(e["sitting_hours"], e["sitting_hours"])
        extras.append(f"daily sitting: {_sit_label}")

    # history & symptoms
    if e.get("family_history") is not None:
        extras.append("family history of early CVD" if e["family_history"] else "no family history of early CVD")
    if e.get("prior_event") is not None:
        extras.append("PRIOR HEART ATTACK / STROKE / TIA" if e["prior_event"]
                      else "no prior heart attack or stroke")
    if e.get("chest_pain") is not None:
        extras.append("chest discomfort / dyspnea on exertion" if e["chest_pain"]
                      else "no chest discomfort on exertion")

    # mental health
    if e.get("stress_level"):
        extras.append(f"stress: {e['stress_level']}")
    if e.get("mental_health") is not None:
        extras.append("history of anxiety/depression" if e["mental_health"]
                      else "no anxiety/depression history")

    # conditions
    conds = [c for c in (e.get("conditions") or []) if c != "none"]
    if conds:
        extras.append("existing diagnoses: " + ", ".join(conds))
    elif "none" in (e.get("conditions") or []):
        extras.append("no known prior diagnoses")

    if extras:
        summary += "  Additional context: " + "; ".join(extras) + "."

    summary += (
        f"  Predicted CVD probability {assessment.probability*100:.1f}% — tier {assessment.tier}."
    )
    return summary


def _spark_points(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Convert history rows (newest-first) into chart-ready (x,y) values.

    Output: oldest-first list of dicts with x in [0..1] and y = probability.
    """
    if not rows:
        return []
    ordered = list(reversed(rows))
    n = len(ordered)
    return [
        {"x": (i / max(n - 1, 1)), "y": float(r["probability"]), "tier": r["tier"]}
        for i, r in enumerate(ordered)
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    meta = model_metadata()
    return render_template("index.html", model_meta=meta, ollama_up=is_available())


@app.route("/assess", methods=["GET", "POST"])
def assess():
    if request.method == "GET":
        return render_template("questionnaire.html")

    patient, error = _parse_questionnaire(request.form)
    if patient is None:
        return render_template(
            "questionnaire.html",
            error=error or "Please check your inputs.",
            form=request.form,
        )

    assessment = predict_risk(patient)
    bmi = round(patient["weight_kg"] / ((patient["height_cm"] / 100) ** 2), 1)
    plan = plan_for_tier(assessment.tier)
    summary = _patient_summary(patient, assessment)

    sid = ensure_session_id(session)
    assessment_dict = assessment.to_dict()
    assessment_id = save_assessment(sid, patient, assessment_dict, bmi)

    imputed_labels = [_IMPUTED_LABEL.get(k, k) for k in (patient.get("_imputed") or [])]

    return render_template(
        "result.html",
        patient=patient,
        bmi=bmi,
        assessment=assessment_dict,
        plan=plan,
        summary=summary,
        ollama_up=is_available(),
        assessment_id=assessment_id,
        imputed_labels=imputed_labels,
        provided_count=patient.get("_provided_count", 0),
        total_questions=TOTAL_QUESTIONS,
        enrichment=patient.get("_enrichment") or {},
    )


# ---------------------------------------------------------------------------
# History page
# ---------------------------------------------------------------------------
@app.route("/history")
def history():
    sid = ensure_session_id(session)
    rows = list_assessments(sid, limit=25)
    spark_points = _spark_points(rows)
    return render_template("history.html", rows=rows, spark_points=spark_points)


@app.route("/history/<int:assessment_id>/delete", methods=["POST"])
def history_delete(assessment_id: int):
    sid = ensure_session_id(session)
    delete_assessment(assessment_id, sid)
    return redirect(url_for("history"))


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------
@app.route("/download/<int:assessment_id>.pdf")
def download_pdf(assessment_id: int):
    sid = ensure_session_id(session)
    row = get_assessment(assessment_id, sid)
    if row is None:
        abort(404)
    plan = plan_for_tier(row["result"]["tier"])
    pdf_bytes = build_pdf_report(row["patient"], row["result"], plan, row["bmi"], row["created_at"])
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="HeartGuard_Assessment_{assessment_id}.pdf"',
        },
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    patient = data.get("patient") or {}
    tier = (data.get("tier") or "").strip() or None

    if not question:
        return jsonify(ok=False, error="Empty question."), 400

    retrieved = retrieve(question, top_k=5, tier_filter=tier)
    summary = data.get("summary") or ""
    context = build_context(retrieved, patient_summary=summary)

    ok, answer = ask_ollama(question, context)
    return jsonify(
        ok=ok,
        answer=answer,
        retrieved=[
            {
                "kind": r["kind"],
                "tier": r["tier"],
                "title": r["title"],
                "emoji": r["emoji"],
                "score": round(r["score"], 4),
                "text": r["text"],
            }
            for r in retrieved
        ],
    )


@app.route("/health")
def health():
    return jsonify(
        ok=True,
        model=model_metadata(),
        model_available=model_available(),
        ollama_available=is_available(),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=True)
