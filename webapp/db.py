"""Lightweight SQLite layer for persistent assessment history.

The web app keeps a single SQLite file at ``data/history.db`` (created on
first use). Each Flask session gets its own UUID stored in the cookie, and
every completed assessment becomes one row in the ``assessments`` table.

Only stdlib ``sqlite3`` is used — no extra dependency, no ORM. Rows store the
patient inputs and the model output as JSON blobs so the schema is forwards
compatible with future model changes.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "history.db")


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL,
                created_at   REAL    NOT NULL,
                probability  REAL    NOT NULL,
                tier         TEXT    NOT NULL,
                tier_color   TEXT    NOT NULL,
                engine       TEXT    NOT NULL,
                bmi          REAL    NOT NULL,
                patient_json TEXT    NOT NULL,
                result_json  TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assessments_session "
            "ON assessments(session_id, created_at DESC)"
        )
        conn.commit()


def ensure_session_id(session: dict[str, Any]) -> str:
    """Return the Flask-session UUID, creating one on first use."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]


def save_assessment(
    session_id: str,
    patient: dict[str, Any],
    assessment_dict: dict[str, Any],
    bmi: float,
) -> int:
    """Persist one assessment and return its rowid."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO assessments
                (session_id, created_at, probability, tier, tier_color, engine,
                 bmi, patient_json, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                time.time(),
                float(assessment_dict["probability"]),
                assessment_dict["tier"],
                assessment_dict["tier_color"],
                assessment_dict.get("engine", "unknown"),
                float(bmi),
                json.dumps(patient, ensure_ascii=False),
                json.dumps(assessment_dict, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_assessments(session_id: str, limit: int = 25) -> list[dict[str, Any]]:
    """Most-recent assessments first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM assessments WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_assessment(assessment_id: int, session_id: str | None = None) -> dict[str, Any] | None:
    """Fetch one assessment. If ``session_id`` is given, returns only rows
    that belong to that session (prevents cross-session access)."""
    with _connect() as conn:
        if session_id:
            row = conn.execute(
                "SELECT * FROM assessments WHERE id = ? AND session_id = ?",
                (assessment_id, session_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM assessments WHERE id = ?",
                (assessment_id,),
            ).fetchone()
    return _row_to_dict(row) if row else None


def delete_assessment(assessment_id: int, session_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM assessments WHERE id = ? AND session_id = ?",
            (assessment_id, session_id),
        )
        conn.commit()
        return cur.rowcount > 0


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["patient"] = json.loads(d.pop("patient_json"))
    d["result"]  = json.loads(d.pop("result_json"))
    return d
