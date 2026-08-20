# Kidney Waitlist Risk & Wait-Time Explorer (Streamlit)

Companion app to the conference paper. Serves the calibrated XGBoost risk model,
the Cox time-to-transplant model, per-candidate SHAP explanations, and the
fairness/cohort views.

## What it does
- **Predict** — enter a candidate's listing-time details → calibrated adverse-outcome
  risk %, risk decile, per-candidate SHAP drivers, and an estimated time-to-transplant
  curve.
- **Cohort** — outcome distribution, median wait by OPTN region, KM curves by policy era.
- **Fairness** — subgroup AUCs (sex, race/ethnicity, blood type, region, age) vs. the
  0.05 tolerance.
- **About** — methods, metrics, and limitations.

## Data governance
Ships **only** model artifacts (`models/`) and aggregate metadata — never row-level
records — consistent with the OPTN Data Use Agreement. Research/education only; **not a
clinical decision tool**.

## Run it
```bash
# from a venv with the stack installed (see requirements.txt)
python train_and_save.py      # one-time: builds models/ from the 2015+ extract
streamlit run app.py          # launches the app at http://localhost:8501
```

`train_and_save.py` reads the extract at
`../extract_2015_kidney_only/kidney_waitlist_analytic.csv`. Rerun it whenever the
extract changes; `app.py` loads whatever is in `models/`.

## Files
| file | purpose |
|---|---|
| `train_and_save.py` | trains + persists preprocessor, XGBoost, isotonic calibrator, Cox model, and `meta.json` |
| `app.py` | the Streamlit UI (loads `models/`) |
| `models/` | persisted artifacts (regenerable; safe to share — no raw data) |
| `requirements.txt` | pinned dependencies |
