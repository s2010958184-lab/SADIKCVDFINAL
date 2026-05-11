"""PDF report builder for HeartGuard assessments.

Generates a clean, single-document PDF a patient can hand to a clinician.
Layout:
    1. Header with logo emoji, app name, generated-at timestamp.
    2. Risk gauge equivalent: probability + tier banner.
    3. Patient inputs table.
    4. Clinical flags (if any).
    5. Exercise plan summary + top exercise cards.
    6. Dietary plan summary + meal plan (if available).
    7. Daily tips (top three).
    8. Disclaimer footer.

Uses ReportLab (Platypus flowables) so layout is text-flow based and prints
cleanly on both A4 and Letter.
"""
from __future__ import annotations

import datetime as _dt
import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=22, textColor=colors.HexColor("#0f172a"),
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=10,
                                   textColor=colors.HexColor("#64748b"), spaceAfter=14),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=14,
                             textColor=colors.HexColor("#0f172a"),
                             spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=12,
                             textColor=colors.HexColor("#334155"),
                             spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=10.5,
                               leading=15, textColor=colors.HexColor("#1f2937"),
                               alignment=TA_LEFT, spaceAfter=4),
        "muted": ParagraphStyle("muted", parent=base["BodyText"], fontSize=9,
                                textColor=colors.HexColor("#64748b"), leading=12),
        "tier-banner": ParagraphStyle("tier", parent=base["BodyText"], fontSize=14,
                                      textColor=colors.white, alignment=1,
                                      fontName="Helvetica-Bold"),
        "footer": ParagraphStyle("footer", parent=base["BodyText"], fontSize=8.5,
                                 textColor=colors.HexColor("#94a3b8"), alignment=1,
                                 leading=11),
    }
    return custom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CHOL = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
_GLUC = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
_SEX  = {1: "Female", 2: "Male"}


def _patient_table(patient: dict[str, Any], bmi: float) -> Table:
    data = [
        ["Age",           f"{int(patient['age_years'])} years"],
        ["Sex",           _SEX.get(int(patient["gender"]), "—")],
        ["Height / Weight", f"{patient['height_cm']:.0f} cm · {patient['weight_kg']:.0f} kg"],
        ["BMI",           f"{bmi:.1f}"],
        ["Blood pressure", f"{int(patient['ap_hi'])}/{int(patient['ap_lo'])} mmHg"],
        ["Cholesterol",   _CHOL.get(int(patient["cholesterol"]), "—")],
        ["Glucose",       _GLUC.get(int(patient["gluc"]), "—")],
        ["Smoker",        "Yes" if patient.get("smoke") else "No"],
        ["Alcohol",       "Yes" if patient.get("alco")  else "No"],
        ["Physically active", "Yes" if patient.get("active") else "No"],
    ]
    t = Table(data, colWidths=[55*mm, 100*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0f172a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
    ]))
    return t


def _tier_banner(assessment: dict[str, Any]) -> Table:
    prob_pct = assessment["probability"] * 100
    text = (
        f"<b>{prob_pct:.1f}%</b> &nbsp;&nbsp;CVD probability &nbsp;·&nbsp; "
        f"{assessment['tier']}"
    )
    p = Paragraph(text, _styles()["tier-banner"])
    t = Table([[p]], colWidths=[165*mm], rowHeights=[18*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(assessment["tier_color"])),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(assessment["tier_color"])),
    ]))
    return t


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_pdf_report(
    patient: dict[str, Any],
    assessment: dict[str, Any],
    plan: dict[str, Any],
    bmi: float,
    created_at: float | None = None,
) -> bytes:
    """Render the assessment as a PDF and return the bytes."""
    s = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=16*mm,
        title="HeartGuard CVD assessment",
        author="HeartGuard",
    )

    story: list[Any] = []
    now = _dt.datetime.fromtimestamp(created_at) if created_at else _dt.datetime.now()

    # --- Header
    story.append(Paragraph("HeartGuard CVD assessment", s["title"]))
    story.append(Paragraph(
        f"Generated {now.strftime('%d %B %Y, %H:%M')} &nbsp;·&nbsp; "
        f"Engine: {'logistic regression' if assessment.get('engine') == 'model' else 'rule-based fallback'}",
        s["subtitle"],
    ))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 8))

    # --- Tier banner
    story.append(_tier_banner(assessment))
    story.append(Spacer(1, 4))
    story.append(Paragraph(assessment.get("tier_summary", ""), s["muted"]))
    story.append(Spacer(1, 8))

    # --- Patient inputs
    story.append(Paragraph("Your inputs", s["h2"]))
    story.append(_patient_table(patient, bmi))

    # --- Flags
    flags = assessment.get("flags") or []
    if flags:
        story.append(Paragraph("Clinical flags", s["h2"]))
        for f in flags:
            story.append(Paragraph(f"<b>[{f['level']}]</b> {f['msg']}", s["body"]))

    # --- Exercise plan
    ex = plan.get("exercise", {})
    if ex:
        story.append(Paragraph("Exercise plan", s["h2"]))
        story.append(Paragraph(
            f"<b>Frequency:</b> {ex.get('frequency', '')} &nbsp; · &nbsp;"
            f"<b>Duration:</b> {ex.get('duration', '')} &nbsp; · &nbsp;"
            f"<b>Intensity:</b> {ex.get('intensity', '')}",
            s["muted"],
        ))
        story.append(Paragraph(ex.get("summary", ""), s["body"]))

        story.append(Paragraph("Recommended exercises", s["h3"]))
        for item in ex.get("exercises", [])[:6]:
            story.append(Paragraph(
                f"<b>{item['emoji']} {item['name']}</b> — {item['detail']}", s["body"],
            ))

        story.append(Paragraph("7-day schedule", s["h3"]))
        plan_data = [["Day", "Activity"]]
        plan_data += [[d, a] for d, a in ex.get("weekly_plan", [])]
        t = Table(plan_data, colWidths=[35*mm, 130*mm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)

        if ex.get("precautions"):
            story.append(Paragraph("Safety precautions", s["h3"]))
            for p in ex["precautions"]:
                story.append(Paragraph(f"• {p}", s["body"]))

    # --- Dietary plan
    diet = plan.get("diet", {})
    if diet:
        story.append(PageBreak())
        story.append(Paragraph("Dietary plan", s["h2"]))
        story.append(Paragraph(
            f"<b>Framework:</b> {diet.get('framework', '')}", s["muted"],
        ))
        story.append(Paragraph(diet.get("summary", ""), s["body"]))

        story.append(Paragraph("Recommended foods", s["h3"]))
        for item in diet.get("foods", [])[:8]:
            story.append(Paragraph(
                f"<b>{item['emoji']} {item['name']}</b> — {item['detail']}", s["body"],
            ))

        if diet.get("meals"):
            story.append(Paragraph("Sample 1-day meal plan", s["h3"]))
            for m in diet["meals"]:
                story.append(Paragraph(
                    f"<b>{m['meal']} — {m['emoji']} {m['name']}</b><br/>"
                    f"<font color='#475569'>{m['detail']}</font>",
                    s["body"],
                ))
                story.append(Spacer(1, 2))

        if diet.get("avoid"):
            story.append(Paragraph("Foods to avoid", s["h3"]))
            for a in diet["avoid"]:
                story.append(Paragraph(f"• {a}", s["body"]))

    # --- Daily tips (top 3)
    tips = plan.get("tips") or []
    if tips:
        story.append(Paragraph("Top daily tips", s["h2"]))
        for t in tips[:3]:
            story.append(Paragraph(
                f"<b>{t['tip']}</b><br/>"
                f"<font color='#475569'>Why: {t['why']}</font><br/>"
                f"<font color='#475569'>Action: {t['action']}</font><br/>"
                f"<font color='#94a3b8' size='8'>Source: {t['source']}</font>",
                s["body"],
            ))
            story.append(Spacer(1, 4))

    # --- Disclaimer footer
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        "Educational tool only. Not a medical device. "
        "Always consult a qualified clinician before making changes to your treatment or medication.",
        s["footer"],
    ))
    story.append(Paragraph(
        "Ahmed Al Sadik · B00983817 · DSA502 — Data Science with AI",
        s["footer"],
    ))

    doc.build(story)
    return buf.getvalue()
