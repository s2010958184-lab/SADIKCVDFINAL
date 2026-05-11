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
def _parse_questionnaire(form: dict[str, str]) -> dict[str, Any] | None:
    """Validate and coerce the 11 form fields into the patient dict."""
    try:
        patient = {
            "age_years": float(form["age_years"]),
            "gender": int(form["gender"]),
            "height_cm": float(form["height_cm"]),
            "weight_kg": float(form["weight_kg"]),
            "ap_hi": float(form["ap_hi"]),
            "ap_lo": float(form["ap_lo"]),
            "cholesterol": int(form["cholesterol"]),
            "gluc": int(form["gluc"]),
            "smoke": int(form.get("smoke", 0)),
            "alco": int(form.get("alco", 0)),
            "active": int(form.get("active", 0)),
        }
    except (KeyError, ValueError):
        return None

    if not (12 <= patient["age_years"] <= 110):
        return None
    if patient["gender"] not in (1, 2):
        return None
    if not (80 <= patient["height_cm"] <= 230):
        return None
    if not (25 <= patient["weight_kg"] <= 300):
        return None
    if not (60 <= patient["ap_hi"] <= 260):
        return None
    if not (30 <= patient["ap_lo"] <= 200):
        return None
    if patient["ap_hi"] <= patient["ap_lo"]:
        return None
    if patient["cholesterol"] not in (1, 2, 3):
        return None
    if patient["gluc"] not in (1, 2, 3):
        return None
    return patient


def _patient_summary(patient: dict[str, Any], assessment) -> str:
    chol_labels = {1: "normal", 2: "above normal", 3: "well above normal"}
    gluc_labels = {1: "normal", 2: "above normal", 3: "well above normal"}
    sex_labels = {1: "female", 2: "male"}
    bmi = patient.get("bmi") or round(
        patient["weight_kg"] / ((patient["height_cm"] / 100) ** 2), 1
    )
    return (
        f"Age {patient['age_years']:.0f}; sex {sex_labels[patient['gender']]}; "
        f"BMI {bmi:.1f}; blood pressure {patient['ap_hi']:.0f}/{patient['ap_lo']:.0f} mmHg; "
        f"cholesterol {chol_labels[patient['cholesterol']]}; "
        f"glucose {gluc_labels[patient['gluc']]}; "
        f"{'smoker' if patient['smoke'] else 'non-smoker'}; "
        f"{'drinks alcohol' if patient['alco'] else 'no alcohol'}; "
        f"{'physically active' if patient['active'] else 'physically inactive'}.  "
        f"Predicted CVD probability {assessment.probability*100:.1f}% — tier {assessment.tier}."
    )


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

    patient = _parse_questionnaire(request.form)
    if patient is None:
        return render_template(
            "questionnaire.html",
            error="Please check your inputs — some values look out of range.",
            form=request.form,
        )

    assessment = predict_risk(patient)
    bmi = round(patient["weight_kg"] / ((patient["height_cm"] / 100) ** 2), 1)
    plan = plan_for_tier(assessment.tier)
    summary = _patient_summary(patient, assessment)

    sid = ensure_session_id(session)
    assessment_dict = assessment.to_dict()
    assessment_id = save_assessment(sid, patient, assessment_dict, bmi)

    return render_template(
        "result.html",
        patient=patient,
        bmi=bmi,
        assessment=assessment_dict,
        plan=plan,
        summary=summary,
        ollama_up=is_available(),
        assessment_id=assessment_id,
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
