"""Ollama client for the minimax-m2.1:cloud model.

Mirrors the ask_ollama() function in the movie-RAG starter.
Returns a tuple (success: bool, text: str) so the UI can render errors cleanly.
"""
from __future__ import annotations

import os
from typing import Any

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "minimax-m2.1:cloud")

SYSTEM_PROMPT = """You are a careful CVD-prevention assistant.

You ALWAYS:
  • Ground every recommendation in the PATIENT PROFILE and the RELEVANT GUIDANCE block given to you.
  • Stay concise — 4 to 8 short sentences unless the user explicitly asks for more.
  • Use plain language; avoid medical jargon unless the patient has used it themselves.
  • Add this disclaimer once at the very end: "This is educational information, not a medical diagnosis. Speak to your doctor before making major changes."
  • Refuse to provide drug dosing or to diagnose specific conditions; redirect those questions to the patient's physician.
  • Never invent statistics; if the guidance block lacks a number, do not make one up.
"""


def is_available(timeout: float = 1.5) -> bool:
    """Quick check used by the UI to enable/disable the chat box."""
    try:
        r = requests.get(OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def ask_ollama(question: str, context: str, timeout: float = 60.0) -> tuple[bool, str]:
    """POST a grounded prompt to /api/generate and return (success, text)."""
    if not question or not question.strip():
        return False, "Please type a question first."

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{question.strip()}\n\n"
        f"ANSWER:"
    )

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "top_p": 0.9},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return False, ("Could not reach Ollama at "
                       f"{OLLAMA_URL}. Start it with `ollama serve` and try again.")
    except requests.exceptions.Timeout:
        return False, "Ollama request timed out. The cloud model may be slow — try again."
    except Exception as e:
        return False, f"Unexpected error talking to Ollama: {type(e).__name__}: {e}"

    if r.status_code != 200:
        return False, f"Ollama returned HTTP {r.status_code}: {r.text[:200]}"
    try:
        return True, (r.json().get("response") or "").strip()
    except Exception:
        return False, f"Could not parse Ollama response: {r.text[:200]}"
