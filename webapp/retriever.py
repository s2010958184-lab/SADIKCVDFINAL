"""TF-IDF retrieval over the curated knowledge corpus.

Mirrors the structure of the movie-RAG example, but the corpus here is
the exercise + diet + tip entries defined in knowledge.py.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge import all_knowledge_documents

_DOCS: list[dict[str, Any]] | None = None
_VECT: TfidfVectorizer | None = None
_MATRIX = None


def _build_index() -> None:
    global _DOCS, _VECT, _MATRIX
    _DOCS = all_knowledge_documents()
    _VECT = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000,
    )
    _MATRIX = _VECT.fit_transform([d["text"] for d in _DOCS])


def _ensure_built() -> None:
    if _DOCS is None or _VECT is None or _MATRIX is None:
        _build_index()


def retrieve(question: str, top_k: int = 5,
             tier_filter: str | None = None,
             kind_filter: str | None = None) -> list[dict[str, Any]]:
    """Return top_k knowledge entries most similar to the question.

    Optional filters narrow the corpus before scoring:
        tier_filter — keep only entries tagged for this risk tier
        kind_filter — 'exercise' | 'diet' | 'tip'
    """
    if not question or not question.strip():
        return []
    _ensure_built()
    assert _VECT is not None and _MATRIX is not None and _DOCS is not None

    q_vec = _VECT.transform([question])
    sims = cosine_similarity(q_vec, _MATRIX).ravel()

    idx = np.arange(len(_DOCS))
    if tier_filter:
        mask = np.array([tier_filter in d["tier"] for d in _DOCS])
        idx = idx[mask]
    if kind_filter:
        mask = np.array([_DOCS[i]["kind"] == kind_filter for i in idx])
        idx = idx[mask]

    if len(idx) == 0:
        return []

    ranked = idx[np.argsort(-sims[idx])][:top_k]
    out = []
    for i in ranked:
        d = dict(_DOCS[i])
        d["score"] = float(sims[i])
        out.append(d)
    return out


def build_context(rows: list[dict[str, Any]], patient_summary: str = "") -> str:
    """Format retrieved entries + patient summary into a compact context block."""
    pieces = []
    if patient_summary:
        pieces.append(f"PATIENT PROFILE:\n{patient_summary}")
    if rows:
        pieces.append("RELEVANT GUIDANCE (retrieved from the clinical knowledge base):")
        for i, r in enumerate(rows, 1):
            pieces.append(
                f"[{i}] kind={r['kind']} · tier={r['tier']} · score={r['score']:.3f}\n"
                f"    {r['emoji']} {r['title']}\n"
                f"    {r['text']}"
            )
    return "\n\n".join(pieces)
