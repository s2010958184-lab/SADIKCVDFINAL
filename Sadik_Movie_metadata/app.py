"""Mini-Project 2 — DSA 502 S26
Flask + Ollama RAG for Movie Metadata.

Pipeline
--------
    Question → Retrieve (TF-IDF top-k) → Build Context → Ollama LLM → Grounded Answer

Built by Ahmed Al Sadik (B00983817).
Runs at http://127.0.0.1:5005.
"""
from __future__ import annotations

import ast
import logging
import os
import time

import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_URL      = os.environ.get(
    "MOVIE_CSV_URL",
    "https://hiperc.buffalostate.edu/courses/movies_metadata.csv",
)
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "minimax-m2.1:cloud")
KEEP_COLS    = ["title", "overview", "genres", "release_date", "vote_average"]
TOP_K        = 5
PORT         = 5005

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("movie-rag")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


# ---------------------------------------------------------------------------
# Globals — built once at startup
# ---------------------------------------------------------------------------
DF: pd.DataFrame | None = None
VECTORIZER: TfidfVectorizer | None = None
MATRIX = None
BOOTSTRAP_ERROR: str | None = None


# ---------------------------------------------------------------------------
# 1) load_movies  ─  download + clean the CSV
# ---------------------------------------------------------------------------
def _parse_genres(raw: object) -> str:
    """Convert the stringified-list-of-dicts `genres` field into a clean
    comma-separated string. Returns "" on any parse failure."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ""
    if not isinstance(parsed, list):
        return ""
    names: list[str] = []
    for item in parsed:
        if isinstance(item, dict) and "name" in item:
            names.append(str(item["name"]).strip())
    return ", ".join(n for n in names if n)


def load_movies(csv_url: str = CSV_URL) -> pd.DataFrame:
    """Download `movies_metadata.csv`, keep the five required columns,
    parse `genres`, drop rows missing `title` or `overview`."""
    log.info("Downloading dataset: %s", csv_url)
    df = pd.read_csv(csv_url, low_memory=False)
    log.info("Raw rows: %s", f"{len(df):,}")

    keep = [c for c in KEEP_COLS if c in df.columns]
    missing = set(KEEP_COLS) - set(keep)
    if missing:
        log.warning("Missing expected columns (proceeding without them): %s", missing)
    df = df[keep].copy()

    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(_parse_genres)

    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

    df = df.dropna(subset=["title", "overview"])
    df = df[(df["title"].astype(str).str.strip() != "")
            & (df["overview"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    log.info("Cleaned rows: %s", f"{len(df):,}")
    return df


def build_index(df: pd.DataFrame) -> tuple[TfidfVectorizer, np.ndarray]:
    """TF-IDF over the concatenation of title + overview + genres."""
    text = (
        df["title"].fillna("") + " "
        + df["overview"].fillna("") + " "
        + df.get("genres", pd.Series([""] * len(df))).fillna("")
    )
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vec.fit_transform(text)
    log.info("TF-IDF index: %s docs × %s features", matrix.shape[0], matrix.shape[1])
    return vec, matrix


# ---------------------------------------------------------------------------
# 2) retrieve_movies  ─  TF-IDF top-k retrieval
# ---------------------------------------------------------------------------
def retrieve_movies(
    question: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Return the top_k rows most similar to `question` with a `score` column."""
    if not question or not question.strip():
        return df.head(0).assign(score=[]).reset_index(drop=True)

    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, matrix).ravel()

    k = min(top_k, len(sims))
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    out = df.iloc[top_idx].copy()
    out["score"] = sims[top_idx]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) build_context  ─  compact, numbered context block
# ---------------------------------------------------------------------------
def build_context(rows: pd.DataFrame) -> str:
    """Compact, numbered context block. Trims overviews so the prompt stays
    short enough for any local LLM."""
    if rows is None or rows.empty:
        return "(no movies retrieved)"
    blocks: list[str] = []
    for i, row in rows.iterrows():
        title = str(row.get("title") or "Unknown").strip()
        year  = str(row.get("release_date") or "")[:4] or "n/a"
        gen   = str(row.get("genres") or "").strip() or "n/a"
        rat   = row.get("vote_average")
        rat_s = f"{float(rat):.1f}/10" if pd.notna(rat) else "n/a"
        ov    = str(row.get("overview") or "").replace("\n", " ").strip()
        if len(ov) > 380:
            ov = ov[:377] + "…"
        blocks.append(
            f"[{i+1}] {title} ({year}) — genres: {gen} | rating: {rat_s}\n"
            f"    {ov}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 4) ask_ollama  ─  grounded prompt → POST to /api/generate
# ---------------------------------------------------------------------------
INSUFFICIENT_PHRASE = (
    "The retrieved context does not contain enough information to answer that."
)

def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a movie expert assistant. Answer the user's question using "
        "ONLY the retrieved movie context below. Do NOT invent movies, ratings, "
        "or facts that are not in the context. Keep your answer concise (3–6 "
        "sentences) and cite each movie you mention by its bracketed number "
        "(e.g. [2]).\n"
        f"If the context is insufficient, reply with exactly: "
        f"\"{INSUFFICIENT_PHRASE}\"\n\n"
        f"=== RETRIEVED CONTEXT ===\n{context}\n\n"
        f"=== USER QUESTION ===\n{question}\n\n"
        "=== GROUNDED ANSWER ==="
    )


def ask_ollama(
    question: str,
    context: str,
    *,
    ollama_url: str = OLLAMA_URL,
    model: str = OLLAMA_MODEL,
    timeout: int = 120,
) -> tuple[bool, str]:
    """Send the grounded prompt to Ollama. Returns (ok, answer_or_error)."""
    payload = {
        "model": model,
        "prompt": _build_prompt(question, context),
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        r = requests.post(ollama_url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return False, (
            "Ollama is not running. Start it in another terminal:\n"
            "    ollama serve\n"
            f"and make sure the model is available:\n"
            f"    ollama pull {model}"
        )
    except requests.exceptions.Timeout:
        return False, f"Ollama took longer than {timeout}s to respond."
    except requests.exceptions.RequestException as e:
        return False, f"Unexpected error talking to Ollama: {e}"

    if r.status_code != 200:
        return False, f"Ollama returned HTTP {r.status_code}: {r.text[:300]}"

    try:
        data = r.json()
    except ValueError:
        return False, f"Ollama returned non-JSON: {r.text[:300]}"

    answer = (data.get("response") or "").strip()
    if not answer:
        return False, "Ollama returned an empty answer."
    return True, answer


# ---------------------------------------------------------------------------
# Bootstrap (eager, defensive)
# ---------------------------------------------------------------------------
def bootstrap() -> None:
    """Load the CSV + build the TF-IDF index. Sets BOOTSTRAP_ERROR on failure
    so the UI can show a friendly message instead of crashing."""
    global DF, VECTORIZER, MATRIX, BOOTSTRAP_ERROR
    try:
        DF = load_movies()
        VECTORIZER, MATRIX = build_index(DF)
        BOOTSTRAP_ERROR = None
        log.info("RAG index ready.")
    except Exception as exc:        # noqa: BLE001
        log.exception("Bootstrap failed")
        DF, VECTORIZER, MATRIX = None, None, None
        BOOTSTRAP_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if DF is None and BOOTSTRAP_ERROR is None:
        bootstrap()

    question  = ""
    error     = None
    rows      = None
    answer    = None
    context   = None
    retrieval_ms = 0
    llm_ms       = 0

    if BOOTSTRAP_ERROR:
        error = (
            f"Could not load the movie dataset: {BOOTSTRAP_ERROR}\n"
            "Check your internet connection or set MOVIE_CSV_URL to a local path."
        )

    if request.method == "POST" and not BOOTSTRAP_ERROR:
        question = (request.form.get("question") or "").strip()
        if not question:
            error = "Please enter a question."
        else:
            try:
                t0 = time.perf_counter()
                rows = retrieve_movies(question, DF, VECTORIZER, MATRIX, top_k=TOP_K)
                retrieval_ms = int((time.perf_counter() - t0) * 1000)

                context = build_context(rows)

                t1 = time.perf_counter()
                ok, llm_out = ask_ollama(question, context)
                llm_ms = int((time.perf_counter() - t1) * 1000)
                if ok:
                    answer = llm_out
                else:
                    error = llm_out
            except Exception as exc:        # noqa: BLE001
                log.exception("Retrieval/LLM error")
                error = f"Unexpected error: {exc}"

    return render_template(
        "index.html",
        question=question,
        error=error,
        rows=(rows.to_dict(orient="records") if rows is not None else None),
        answer=answer,
        context=context,
        n_corpus=(len(DF) if DF is not None else 0),
        model_name=OLLAMA_MODEL,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        top_k=TOP_K,
    )


@app.route("/health")
def health():
    """Quick status endpoint."""
    return {
        "ok": DF is not None,
        "rows": (len(DF) if DF is not None else 0),
        "model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
        "bootstrap_error": BOOTSTRAP_ERROR,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bootstrap()
    app.run(host="127.0.0.1", port=PORT, debug=False)
