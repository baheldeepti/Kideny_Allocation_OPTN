# OPTN Kidney Waitlist — Risk & Time-to-Transplant

Predictive modeling and analysis of the U.S. kidney transplant **waiting list**:
who is at risk of dying/being removed before transplant, how long candidates
wait, and whether the models are fair and explainable. Built on the OPTN
`KIDPAN` STAR file (June 2026 release), kidney-alone candidates listed 2015+.

> ⚠️ **Data governance.** The registry data are governed by an OPTN Data Use
> Agreement. Row-level extracts live under `data/` and must **never** be
> committed to a public repo or shared outside the team. Only code, aggregate
> summaries, figures, and trained models are shareable.

## Structure

```
optn/
├── extraction/                  Cohort construction from the raw STAR file
│   └── build_analytic_extract.py
│
├── data/                        DUA-governed extracts (LOCAL ONLY, gitignored)
│   ├── extract_2015_kidney_only/    ← primary cohort (494,862 rows)
│   ├── extract_2015_ki_kp/          kidney + kidney-pancreas, 2015+
│   ├── extract_2010_ki_kp/          kidney + kidney-pancreas, 2010+
│   └── README.md
│
├── analysis/                    Modeling, evaluation, and the paper
│   ├── run_analysis.py              full pipeline → REPORT.md + figures/
│   ├── build_artifact.py            PAPER.md → paper.html (shareable)
│   ├── REPORT.md                    technical results log
│   ├── PAPER.md                     conference paper (IEEE-style)
│   ├── paper.html                   rendered paper (self-contained)
│   └── figures/                     8 result figures (PNG)
│
└── app/                         Streamlit app (local)
    ├── app.py                       5-tab UI (Predict / Batch / Cohort / Fairness / About)
    ├── train_and_save.py            trains + persists models/ from the extract
    ├── models/                      persisted artifacts (no raw data)
    ├── run.sh                       one-command launcher
    ├── Dockerfile                   optional private hosting
    ├── requirements.txt
    └── README.md
```

The source `KIDPAN_DATA.DAT` (~1.4 GB) and its `.htm` dictionary stay in
`~/Downloads/Delimited Text File 202606/...` — they are not copied here.

## Environment

All code runs in the dedicated venv at `~/ckd-ml` (Python 3.12; pandas,
scikit-learn, XGBoost, lifelines, SHAP, matplotlib, streamlit). Recreate with
`pip install -r app/requirements.txt`.

## How to run, end to end

```bash
PY=~/ckd-ml/bin/python

# 1) build the cohort from the raw STAR file (already done → data/)
$PY extraction/build_analytic_extract.py \
    --input "$HOME/Downloads/Delimited Text File 202606/Kidney_ Pancreas_ Kidney-Pancreas/KIDPAN_DATA.DAT" \
    --outdir data/extract_2015_kidney_only --min-year 2015   # add nothing to keep KI+KP; WL_ORG==KI is set in the script

# 2) run the analysis → REPORT.md + figures/
$PY analysis/run_analysis.py

# 3) (re)build the shareable paper page
$PY analysis/build_artifact.py

# 4) train + persist the app's models, then launch the app
$PY app/train_and_save.py
./app/run.sh                      # → http://localhost:8501
```

## Key results (held-out test)

| task | metric | value |
|---|---|---|
| Risk classification | ROC-AUC / PR-AUC | 0.726 / 0.284 |
| Risk classification | Brier (calibrated) | 0.111 (from 0.212) |
| Risk classification | top-decile lift | 2.51× |
| Time-to-transplant | Cox C-index | 0.629 |
| Fairness | max subgroup-AUC gap | ≤ 0.05 (all dimensions) |

See `analysis/PAPER.md` for the full write-up, assumptions, and references.
