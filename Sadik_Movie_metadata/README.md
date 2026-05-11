# Sadik_Movie_metadata

**Mini-Project 2 — DSA 502 S26**
**Flask + Ollama RAG for Movie Metadata**

By **Ahmed Al Sadik** · `B00983817`

A small Flask web application that answers movie-related questions by:

1. **Retrieving** the top-5 most relevant movies from `movies_metadata.csv` using **TF-IDF + cosine similarity** (the Retrieval stage of RAG).
2. **Building** a compact context block from those rows.
3. **Sending** the question + context to a locally-hosted LLM (`minimax-m2.1:cloud`) via the **Ollama HTTP API**.
4. **Returning** a grounded answer that uses only the retrieved context.

```text
Question → Retrieve (TF-IDF top 5) → Build Context → Ollama LLM → Grounded Answer
```

---

## Project structure

```text
Sadik_Movie_metadata/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

---

## Dataset

- **URL:** <https://hiperc.buffalostate.edu/courses/movies_metadata.csv>
- **Columns kept:** `title`, `overview`, `genres`, `release_date`, `vote_average`
- The `genres` column (a stringified list of dicts in the source file) is parsed into a clean comma-separated list of genre names.
- Rows missing a `title` or `overview` are dropped.

You can override the source URL at runtime:

```bash
# Windows PowerShell
$env:MOVIE_CSV_URL = "C:\path\to\movies_metadata.csv"
python app.py
```

---

## Setup

### 1. Install Ollama and pull the model

```bash
ollama --version
ollama pull minimax-m2.1:cloud
ollama serve         # leave running in another terminal
```

The app talks to `http://localhost:11434/api/generate`.
Override with the `OLLAMA_URL` / `OLLAMA_MODEL` env vars if needed.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Then open **<http://127.0.0.1:5005>** in a browser.

> The first request takes a moment because the CSV is downloaded and the TF-IDF index is built **once at startup**.

---

## How it works

`app.py` exposes exactly the four pure functions required by the spec:

| Function | Purpose |
|---|---|
| `load_movies()` | Downloads + cleans the CSV. Returns a `pandas.DataFrame`. |
| `retrieve_movies(question, df, vectorizer, matrix, top_k=5)` | Returns the top-k movies and their cosine-similarity scores. |
| `build_context(rows)` | Builds a compact, numbered context block sent to the LLM. |
| `ask_ollama(question, context)` | POSTs a **grounded** prompt to `http://localhost:11434/api/generate` with `model: "minimax-m2.1:cloud"` and `stream: false`. |

The LLM prompt explicitly instructs the model to **answer only from the retrieved context** and to say

> "The retrieved context does not contain enough information to answer that."

if the context is insufficient — this is the grounding requirement from the rubric.

---

## Sample question

> **Q:** Find movies about space travel and exploration.

The UI shows:

- A **table of the top 5 retrieved movies** with cosine-similarity scores, year, rating, genre tags, and a trimmed overview.
- The **grounded answer** produced by `minimax-m2.1:cloud`.
- An **expandable section** showing the exact context block that was sent to the LLM.
- **Retrieval time** and **LLM time** in milliseconds in the header.

A screenshot of the running app is included as `screenshot.png` (captured on `http://127.0.0.1:5005`).

### Other example questions

- "What are some high-rated drama movies in this dataset?"
- "Which retrieved movies are related to war themes?"
- "Find movies with romance and comedy elements."
- "Suggest movies about time travel or alternate realities."

---

## Error handling

| Failure mode | Behavior |
|---|---|
| Empty input | Rejected with a friendly message. |
| Dataset download fails at startup | UI shows the underlying error and the page stays usable. |
| Ollama not running | "Ollama is not running. Start it in another terminal: `ollama serve` …" |
| HTTP non-200 from Ollama | The status code + first 300 chars of the body are surfaced. |
| Non-JSON response from Ollama | Raw text is surfaced instead of crashing. |
| LLM timeout (>120 s) | Friendly timeout message. |

The app never crashes the request — every error becomes a readable red banner.

---

## Routes

| Method | Path | What it does |
|---|---|---|
| `GET` / `POST` | `/` | Main form, retrieval table, grounded answer, context block. |
| `GET` | `/health` | JSON status: `{ok, rows, model, ollama_url, bootstrap_error}` |

---

## Configuration via env vars

| Variable | Default | Purpose |
|---|---|---|
| `MOVIE_CSV_URL` | `https://hiperc.buffalostate.edu/courses/movies_metadata.csv` | Override the dataset source (e.g. a local path). |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint. |
| `OLLAMA_MODEL` | `minimax-m2.1:cloud` | Model name passed to Ollama. |

---

## Submission

- **GitHub repo:** [`Sadik_Movie_metadata`](https://github.com/s2010958184-lab/SADIKCVDFINAL/tree/main/Sadik_Movie_metadata) (lives inside the main CVD repo)
- All files (`app.py`, `templates/index.html`, `requirements.txt`, `README.md`, `screenshot.png`) committed and pushed.
- Repo URL submitted via the class Google Form / Dropbox.

---

## License

Educational use only — DSA 502, Spring 2026, Buffalo State University.
