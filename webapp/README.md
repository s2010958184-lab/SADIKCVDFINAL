# HeartGuard — CVD Risk Web App

> Built by **Ahmed Al Sadik** · B00983817 — productisation layer for the DSA502 final project.

Flask web app that asks the user 11 health questions, predicts their cardiovascular disease (CVD) risk, then shows a personalised exercise plan, dietary plan, daily tips, and clinical flags — with an optional Ollama-powered chatbot, history tracking, PDF download, and animated SVG iconography for every food and exercise.

---

## What it does

| Feature | Where |
|---|---|
| Multi-step questionnaire (11 inputs) | `/assess` |
| Animated risk gauge + tier-coloured banner | `/assess` (POST) |
| Personalised exercise & diet plans with **sample meal plan** | `result.html` |
| **Animated SVG icons** for running, cycling, swimming, lifting, yoga, walking, heart, apple, fish, salad, avocado, berries, milk, etc. | `templates/_icons.html` + `static/css/icons.css` |
| **Assessment history** with sparkline of CVD probability over time | `/history` |
| **PDF download** of any past assessment | `/download/<id>.pdf` |
| Ollama-grounded "Ask the AI" chat (TF-IDF retrieval) | `/chat` |

---

## Quick start

```bash
cd webapp
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Option A — train on the single Kaggle CSV (place it at data/cardio_train.csv)
python train_model.py

# Option B — train on the COMBINED dataset (best accuracy)
#   place any of:
#     data/cardio_train.csv          (Kaggle Sulianova)
#     data/framingham.csv            (Framingham)
#     data/LLCP2022.XPT              (BRFSS 2022 SAS XPT)
#     data/nhanes_2017_2018.csv      (NHANES 2017–18 harmonised CSV)
#   then run:
python train_combined.py

# Option C — skip training; the app falls back to a calibrated rule-based engine

python app.py
# Open http://127.0.0.1:5005
```

## Optional — Ollama chatbot

Install Ollama (https://ollama.com), then in a second terminal:

```bash
ollama pull minimax-m2.1:cloud
ollama serve
```

The chat box on the results page becomes active once Ollama is reachable.

---

## Folder layout

```
webapp/
├── app.py                 # Flask routes (incl. /history, /download/<id>.pdf)
├── model.py               # Load model + risk-tier mapping + rule-based fallback
├── train_model.py         # LR on cardio_train.csv → models/cvd_model.pkl
├── train_combined.py      # Multi-source training (Kaggle+Framingham+BRFSS+NHANES)
├── knowledge.py           # Exercise + diet + tip database + emoji→icon map
├── retriever.py           # TF-IDF retrieval over knowledge entries
├── ollama_client.py       # POST to http://localhost:11434/api/generate
├── db.py                  # SQLite history (data/history.db)
├── pdf_report.py          # ReportLab PDF builder
├── data/                  # Datasets + history.db
├── models/                # cvd_model.pkl generated here
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── questionnaire.html
│   ├── result.html
│   ├── history.html
│   └── _icons.html        # Inline animated-SVG macro
└── static/
    ├── css/style.css
    ├── css/icons.css      # SVG keyframe animations
    └── js/app.js
```

---

## How it mirrors the movie-RAG pattern

| Movie RAG | This app |
|---|---|
| `load_movies()` loads CSV | `train_model.py` / `train_combined.py` builds the supervised model; `knowledge.py` is the curated corpus |
| `retrieve_movies()` TF-IDF top-k | `retrieve()` TF-IDF top-k over exercise/diet/tip entries |
| `build_context()` builds prompt | `build_context()` formats retrieved entries + the user's risk profile |
| `ask_ollama()` POST `/api/generate` | `ask_ollama()` same endpoint, model `minimax-m2.1:cloud` |
| Form input → retrieved rows → answer | Form input → predicted risk + retrieved plan items + AI explanation |

---

## Smoke test

```bash
python -c "from app import app; c = app.test_client(); print(c.get('/').status_code)"
```

A full 6-route smoke test (landing, questionnaire, results, history, PDF, /health) is verified to pass before each commit.
