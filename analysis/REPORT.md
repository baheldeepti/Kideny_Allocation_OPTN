# Kidney Waitlist Risk & Wait-Time — Analysis Report

_Source: `kidney_waitlist_analytic.csv` (2015+ kidney-alone extract). Aggregates and figures only; no row-level data._

## 1. Data understanding

- **Cohort:** kidney-alone waitlist registrations, listed 2015-01-01 onward
- **Rows:** 494,862
- **Observation window derived fields:** `outcome`, `event_adverse`, `event_transplant`, `days_to_event`

### Outcome base rates

| outcome | n | share |
|---|---:|---:|
| transplanted | 234,429 | 47.4% |
| still_waiting | 103,300 | 20.9% |
| removed_administrative | 45,332 | 9.2% |
| transplanted_elsewhere | 37,243 | 7.5% |
| removed_too_sick | 35,219 | 7.1% |
| died | 34,524 | 7.0% |
| unknown | 4,815 | 1.0% |

- **Adverse (died / removed-too-sick):** 69,743 (14.1%) — the positive class for classification
- **Transplanted (event for survival):** 234,429 (47.4%)
- **Still waiting at window end (censored):** 103,300 (20.9%)

### Missingness of modeling features

| feature | % missing |
|---|---:|
| INIT_AGE | 0.0% |
| INIT_CPRA | 28.8% |
| BMI_TCR | 0.3% |
| GENDER | 0.0% |
| ETHCAT | 0.0% |
| ABO | 0.0% |
| ON_DIALYSIS | 0.0% |
| FUNC_STAT_TCR | 0.0% |
| INIT_STAT | 0.0% |
| REGION | 0.0% |
| policy_era | 0.0% |

_Figures: `outcome_distribution.png`, `adverse_rate_breakdown.png`_

## 2. Classification — who is at risk?

**Target:** adverse waitlist outcome (died or removed-too-sick) vs. all other outcomes. **Features:** listing-time only (no transplant-record fields, to avoid leakage). Stratified 75/25 split.

| model | ROC-AUC | PR-AUC | Brier | Lift@top-decile |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.716 | 0.275 | 0.218 | 2.35x |
| XGBoost (isotonic-calibrated) | 0.726 | 0.284 | 0.111 | 2.51x |

_Positive-class prevalence in test set: 14.1% (PR-AUC baseline = prevalence)._

**Post-hoc calibration:** isotonic calibration cut the XGBoost Brier score from **0.212** (raw — inflated because `scale_pos_weight` upweights the positive class) to **0.111**, now within the ≤ 0.18 target. Ranking metrics (ROC-AUC, PR-AUC, lift) are unchanged: isotonic regression is a monotonic transform.

**vs. success targets (best model — XGBoost):**

| metric | target | achieved | met? |
|---|---:|---:|:--:|
| ROC-AUC | >= 0.78 | 0.726 | ❌ |
| PR-AUC | >= 0.65 | 0.284 | ❌ |
| Brier | <= 0.18 | 0.111 | ✅ |
| Lift@decile | >= 2.5 | 2.509x | ✅ |

_Figures: `roc_pr_curves.png`, `calibration.png`, `decile_lift.png`_

## 3. Survival — how long until transplant?

**Event:** transplant. Death, removal, and still-waiting are treated as **censored** (right-censoring). Cox proportional hazards; evaluated with Harrell's C-index. Fit on a 150k random sub-sample for tractability.

- **Cox C-index (time-to-transplant):** 0.629  (target >= 0.72 → ❌ below target)

- **Median observed days-to-transplant** (transplanted only) ranges 211–400 days across OPTN regions — strong geographic variation.

_Figure: `km_survival.png`_

## 4. Fairness audit

Subgroup ROC-AUC for the XGBoost risk model. Target: subgroup AUC within 0.05 of the overall AUC.

**Overall test AUC: 0.726**


**Sex**

| subgroup | n | AUC | Δ vs overall |
|---|---:|---:|---:|
| F | 47,073 | 0.722 | -0.004 |
| M | 76,643 | 0.728 | +0.001 |

_Max gap: 0.004 — ✅ within 0.05_

**Race/ethnicity**

| subgroup | n | AUC | Δ vs overall |
|---|---:|---:|---:|
| Black | 36,237 | 0.703 | -0.024 |
| Multiracial | 1,090 | 0.712 | -0.014 |
| Amer Indian | 945 | 0.725 | -0.001 |
| White | 50,797 | 0.726 | -0.000 |
| Pacific Isl | 619 | 0.727 | +0.001 |
| Asian | 9,102 | 0.741 | +0.014 |
| Hispanic | 24,498 | 0.749 | +0.022 |

_Max gap: 0.024 — ✅ within 0.05_

**Blood type**

| subgroup | n | AUC | Δ vs overall |
|---|---:|---:|---:|
| A1 | 566 | 0.683 | -0.044 |
| AB | 4,648 | 0.713 | -0.013 |
| A | 38,998 | 0.717 | -0.010 |
| B | 18,499 | 0.720 | -0.006 |
| O | 60,838 | 0.731 | +0.005 |

_Max gap: 0.044 — ✅ within 0.05_

**OPTN region**

| subgroup | n | AUC | Δ vs overall |
|---|---:|---:|---:|
| 3 | 17,296 | 0.703 | -0.023 |
| 4 | 14,184 | 0.709 | -0.017 |
| 11 | 14,932 | 0.712 | -0.014 |
| 10 | 8,040 | 0.713 | -0.014 |
| 7 | 10,278 | 0.719 | -0.007 |
| 8 | 6,513 | 0.723 | -0.004 |
| 2 | 14,845 | 0.725 | -0.001 |
| 9 | 8,611 | 0.730 | +0.004 |
| 5 | 20,466 | 0.749 | +0.023 |
| 1 | 5,061 | 0.755 | +0.028 |
| 6 | 3,490 | 0.761 | +0.034 |

_Max gap: 0.034 — ✅ within 0.05_

**Age band**

| subgroup | n | AUC | Δ vs overall |
|---|---:|---:|---:|
| 65+ | 22,428 | 0.684 | -0.042 |
| 35-49 | 32,361 | 0.685 | -0.041 |
| 50-64 | 51,170 | 0.691 | -0.036 |
| <35 | 17,729 | 0.719 | -0.008 |

_Max gap: 0.042 — ✅ within 0.05_

_Figure: `fairness_region_auc.png`_

## 5. Explainability — model drivers (SHAP)

SHAP computed via XGBoost's native `pred_contribs` on a 4,000-row test sample. Bars are mean |SHAP| — average impact on the log-odds of an adverse waitlist outcome.

| rank | feature | mean\|SHAP\| |
|---:|---|---:|
| 1 | cat__FUNC_STAT_TCR_MISSING | 1.0879 |
| 2 | cat__policy_era_AcuityCircles_2021+ | 0.5020 |
| 3 | cat__REGION_4 | 0.4629 |
| 4 | num__BMI_TCR | 0.4617 |
| 5 | cat__FUNC_STAT_TCR_4040.0 | 0.3827 |
| 6 | cat__ETHCAT_998 | 0.3482 |
| 7 | cat__FUNC_STAT_TCR_2010.0 | 0.3161 |
| 8 | cat__FUNC_STAT_TCR_infrequent_sklearn | 0.3077 |
| 9 | num__INIT_AGE | 0.2994 |
| 10 | cat__FUNC_STAT_TCR_4050.0 | 0.2903 |
| 11 | cat__ABO_A1B | 0.2834 |
| 12 | cat__FUNC_STAT_TCR_2040.0 | 0.2779 |
| 13 | cat__FUNC_STAT_TCR_2030.0 | 0.2602 |
| 14 | cat__ETHCAT_1 | 0.2431 |
| 15 | cat__ETHCAT_6 | 0.2393 |

_Figure: `shap_importance.png`_

## Notes & caveats

- **Censoring:** the classifier labels still-waiting candidates as non-adverse; some will later have an adverse event. The survival model handles this properly — read the two together.
- **Competing risks:** transplant, death, and removal compete. The Cox model treats non-transplant as censored; a Fine–Gray / cause-specific model is the recommended next step.
- **Leakage guard:** transplant-record fields (DIAG_KI, PREV_TX, ORGAN, DON_TY, PSTATUS, TX_DATE, …) were excluded — their ~53% missingness equals the non-transplant rate, i.e. they are only known post-outcome.
- **Distribution shift:** policy era (KAS vs. 2021 Acuity Circles) is encoded as a feature; subgroup/era stability deserves deeper checks.
- FUNC_STAT_TCR / INIT_STAT kept as categorical status codes.