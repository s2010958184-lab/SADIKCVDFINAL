# DSA502 — Cardiovascular Disease Risk Prediction & Prevention

**Student:** Ahmed Al Sadik
**Banner ID:** B00983817
**Course:** Data Science with AI — DSA502
**Phase:** Phase 1 + Phase 2 (Multi-source training, full responsible-AI suite, productised web app)

This project predicts cardiovascular disease (CVD) risk and prescribes personalised, risk-tiered exercise plans, daily wellness tips, and a flag-based prevention advisor — driven by a model trained on a 70,000-patient Kaggle clinical dataset (Phase 1) and on **~480,000 rows pooled from 5 hand-picked public CVD cohorts** (Phase 2).

---

## Project structure

```
.
├── CVDPHASE1_AhmedAlSadik_B00983817.ipynb   # main notebook (Phase 1 + Phase 2 §11–§22)
├── README.md
├── requirements.txt
├── webapp/                                   # productised Flask + Ollama RAG demo
│   ├── app.py, model.py, train_model.py
│   ├── train_combined.py                     # multi-source training (Kaggle+Framingham+BRFSS+NHANES)
│   ├── db.py, pdf_report.py
│   ├── knowledge.py, retriever.py, ollama_client.py
│   ├── templates/    (incl. animated SVG macro)
│   └── static/       (incl. icons.css for keyframe-animated icons)
└── data_extra/                               # auto-created on first run; caches downloaded datasets
```

---

## Datasets used

### Primary training (Phase 1)
| Dataset | n | Source | License |
|---|---|---|---|
| Kaggle Cardiovascular Disease (sulianova) | 70,000 | https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset | CC BY-SA 4.0 |

### Additional training cohorts (Phase 2 §11–§13)
| Dataset | n | Why it was selected | License |
|---|---|---|---|
| **Framingham Heart Study** | ~4,240 | Near-perfect feature overlap, continuous chol & glucose, industry gold standard | Public |
| **CDC BRFSS 2022 — Indicators of Heart Disease** | ~400,000 | Massive row volume (5× current dataset); lifestyle features | Public domain |
| **NHANES 2017–2018** | ~9,000 | Lab-grade continuous BP, cholesterol, glucose, BMI — highest feature quality | Public domain |

### External validation cohorts (Phase 2 §15)
| Dataset | n | Why it was selected | License |
|---|---|---|---|
| **UCI Heart Disease** (Cleveland + Hungarian + Switzerland + VA) | 920 | 4-country external test set — never used for training | CC BY 4.0 |
| **Z-Alizadeh Sani CAD (Iran)** | 303 | Different ethnic cohort — tests geographic bias raised in §10.4 | CC BY 4.0 |

All Phase 2 loaders have **graceful fallback**: if Kaggle authentication is unavailable or a mirror is down, the relevant cell prints `[skip]` and execution continues without crashing.

---

## How to run

### Option A — Google Colab (recommended)

1. Open the notebook in Colab (badge at the top of the notebook).
2. Optional but recommended: upload your `kaggle.json` API token to enable the Kaggle datasets (Kaggle → Settings → API → Create New Token).
3. `Runtime → Run all`.

### Option B — Local

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
jupyter notebook CVDPHASE1_AhmedAlSadik_B00983817.ipynb
```

For Kaggle datasets locally, place your `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (macOS/Linux) before running the §11 cells.

NHANES, UCI Heart Disease, and Z-Alizadeh download from public no-auth URLs (NCHS, UCI ML Repository).

---

## Notebook section map

| Section | Content |
|---|---|
| §0 | Quick-start checklist |
| §6.1 – §6.3 | Project overview, AI usage log, data card |
| §7 | Reproducibility / environment |
| §8.1 – §8.6 | Setup, loading, cleaning, EDA (7 plots + 7 insights), baseline LR model |
| §8.7 | Personalised exercise recommendation engine (4 risk tiers) |
| §8.8 | Daily tip + clinical insight engine |
| §8.9 | Flag-based prevention advisor |
| §9 | Phase 1 written narrative |
| §10.4 | Responsible AI |
| **§11** | Phase 2 — Multi-source data acquisition (5 datasets) |
| **§12** | Schema harmonisation onto common 11-feature schema |
| **§13** | Combined training set assembly |
| **§14** | Retrain on combined data: LR + RF + Gradient Boosting + XGBoost |
| **§15** | External validation on held-out cohorts (UCI + Z-Alizadeh) |
| **§16** | Updated Phase 2 narrative + roadmap |
| **§17** | **Hyperparameter tuning — `RandomizedSearchCV` on best model** |
| **§18** | **Probability calibration (reliability + Brier) + clinical threshold tuning** |
| **§19** | **Permutation importance + learning curves** |
| **§20** | **Subgroup fairness analysis (gender, age, source)** |
| **§21** | **Model Card (Mitchell et al. template)** |
| **§22** | **Conclusion, limitations, future work + executive summary** |

---

## Productised demo — `webapp/`

A Flask web application (`webapp/`) packages the trained model as a full self-service tool:
multi-step questionnaire, animated risk gauge, animated SVG plans for exercises and foods,
sample meal plans by risk tier, persistent assessment history (SQLite) with a sparkline,
PDF report download (ReportLab), and an Ollama-grounded RAG chat (`minimax-m2.1:cloud`).

See `webapp/README.md` for setup + run instructions.

---

## AI usage

See §6.2 of the notebook for the running AI usage log. AI tools were used for brainstorming and code-structure suggestions only; all clinical reasoning, feature engineering choices, and final code are the student's own work and verified against authoritative clinical guidelines (WHO, ACC/AHA, ACSM, JNC 8, NCEP ATP III, ADA).

---

## Reproducibility

- All cells use `RANDOM_SEED = 42`.
- Environment pinned in `requirements.txt`.
- Stratified 80/20 splits.
- 5-fold `StratifiedKFold` cross-validation in §8.6, §14, §17.
- Each Phase 2 dataset loader caches downloaded files to `data_extra/` so reruns are fast.

## Rubric coverage (Phase 1 + Phase 2)

| Rubric area | Where in this project |
|---|---|
| GitHub repo + commit history | `https://github.com/s2010958184-lab/SADIKCVDFINAL.git` |
| Problem framing | §6.1, §9.1 |
| Data card | §6.3 |
| EDA (≥ 3 plots, ≥ 2 insights, leakage check) | §8.4 (7 plots, 7 insights, leakage audit) |
| Baseline + error analysis | §8.5–§8.6 |
| Phase 2 plan | §9.5, §16 |
| Reproducibility | §7, `requirements.txt`, `webapp/requirements.txt` |
| AI usage log | §6.2 (6 entries) |
| Responsible AI | §10.4 |
| **Better validation strategy** | 5-fold CV in §8.6, §14, §17 |
| **Feature engineering** | `age_years`, `bmi`, `pulse_pressure` (§8.5); per-source feature (§14) |
| **Model comparison** | LR + RF + Gradient Boosting + XGBoost (§14) |
| **Hyperparameter tuning** | `RandomizedSearchCV` over the best model (§17) |
| **Calibration** | Reliability diagram + Brier score (§18) |
| **Threshold tuning** | Three principled choices, clinical default adopted (§18) |
| **Class imbalance** | `class_weight="balanced"` + stratified splits |
| **Interpretability** | Permutation importance + learning curve (§19) |
| **Fairness / subgroup metrics** | Gender, age, source bucket AUC + Recall + PPV (§20) |
| **Robustness — external validation** | UCI Heart + Z-Alizadeh zero-shot (§15) |
| **Model card** | §21 (Mitchell et al. format) |
| **Conclusions + next steps** | §22 |
| **Productisation** | `webapp/` Flask + Ollama RAG demo |

## License & attribution

Code: MIT — see notebook header.
Datasets: see individual licences listed above. Attribute the original authors when reusing.
