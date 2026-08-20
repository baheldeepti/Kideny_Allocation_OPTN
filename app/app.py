#!/usr/bin/env python3
"""
Kidney Waitlist Risk & Wait-Time Explorer — Streamlit app.

Serves the calibrated XGBoost risk model, the Cox time-to-transplant model,
per-candidate SHAP explanations, and the fairness/cohort views from persisted
artifacts. Loads only models + aggregate metadata; no row-level data.

    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb

MODELS = Path(__file__).parent / "models"

st.set_page_config(page_title="Kidney Waitlist Risk Explorer",
                   page_icon="🫘", layout="wide")

ACCENT = "#0E7C86"

# --------------------------------------------------------------------------
# load artifacts (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    pre = joblib.load(MODELS / "preprocessor.joblib")
    clf = joblib.load(MODELS / "xgb_classifier.joblib")
    iso = joblib.load(MODELS / "isotonic.joblib")
    cph = joblib.load(MODELS / "cox_model.joblib")
    meta = json.loads((MODELS / "meta.json").read_text())
    return pre, clf, iso, cph, meta

pre, clf, iso, cph, meta = load_artifacts()
ETH = meta["ethcat_map"]
NUMF = meta["features"]["num"]
CATF = meta["features"]["cat"]

FUNC_HELP = ("Functional status at listing (Karnofsky-style code; higher = "
             "better function). Strongest single driver in the model.")

def eth_label(code):
    return ETH.get(str(code), f"code {code}")

def pretty_feature(name):
    """Turn 'cat__ETHCAT_2' / 'num__INIT_AGE' into a readable label."""
    n = name.replace("num__", "").replace("cat__", "")
    for raw, lab in [("ETHCAT_", "Ethnicity="), ("ABO_", "Blood type="),
                     ("REGION_", "OPTN region="), ("GENDER_", "Sex="),
                     ("ON_DIALYSIS_", "On dialysis="),
                     ("FUNC_STAT_TCR_", "Func status="),
                     ("INIT_STAT_", "Init status="),
                     ("policy_era_", "Policy era=")]:
        if n.startswith(raw):
            val = n[len(raw):]
            if raw == "ETHCAT_":
                val = eth_label(val)
            return lab + val
    return {"INIT_AGE": "Age at listing", "INIT_CPRA": "CPRA at listing",
            "BMI_TCR": "BMI"}.get(n, n)

def build_row(inp):
    return pd.DataFrame([{k: inp[k] for k in meta["features"]["num"] + meta["features"]["cat"]}])

def batch_predict(df_in):
    """Score a whole CSV of candidates. Returns the input frame plus
    calibrated risk, risk decile, and cause-specific transplant probabilities."""
    df = df_in.copy()
    # derive policy era if not supplied
    if "policy_era" not in df.columns or df["policy_era"].isna().all():
        if "INIT_DATE" in df.columns:
            dt = pd.to_datetime(df["INIT_DATE"], errors="coerce")
            df["policy_era"] = np.where(dt >= pd.Timestamp("2021-03-15"),
                                        "AcuityCircles_2021+", "KAS_2015_2021")
        else:
            df["policy_era"] = "KAS_2015_2021"
    # ensure every model feature exists + matches training dtype handling
    for c in NUMF:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in CATF:
        col = df[c] if c in df.columns else pd.Series([np.nan] * len(df))
        df[c] = col.astype("string").fillna("MISSING").astype(object)

    X = df[NUMF + CATF]
    Xt = pre.transform(X)
    Xd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    p_raw = clf.predict_proba(Xd)[:, 1]
    p_cal = iso.transform(p_raw)
    edges = np.array(meta["decile_edges"])
    b = np.clip(np.searchsorted(edges[1:-1], p_cal, side="right") + 1, 1, 10)
    risk_decile = 11 - b
    # cause-specific transplant probability at 1 and 2 years (vectorized)
    surv = cph.predict_survival_function(
        pd.DataFrame(Xd, columns=meta["cox_cols"]), times=[365, 730])
    p_tx_1yr = (1 - surv.loc[365].values)
    p_tx_2yr = (1 - surv.loc[730].values)

    out = df_in.copy()
    out["adverse_risk"] = np.round(p_cal, 4)
    out["risk_decile"] = risk_decile.astype(int)
    out["p_transplant_1yr"] = np.round(p_tx_1yr, 3)
    out["p_transplant_2yr"] = np.round(p_tx_2yr, 3)
    return out


def predict(inp):
    row = build_row(inp)
    Xt = pre.transform(row)
    Xd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    p_raw = float(clf.predict_proba(Xd)[:, 1][0])
    p_cal = float(iso.transform([p_raw])[0])
    # risk decile (1 = highest risk)
    edges = np.array(meta["decile_edges"])
    b = int(np.clip(np.searchsorted(edges[1:-1], p_cal, side="right") + 1, 1, 10))
    risk_decile = 11 - b
    # SHAP contributions (native xgboost)
    contribs = clf.get_booster().predict(xgb.DMatrix(Xd), pred_contribs=True)[0]
    feat = meta["feat_names"]
    shap = sorted(zip(feat, contribs[:-1]), key=lambda kv: abs(kv[1]), reverse=True)
    # Cox survival curve (cause-specific: censors death/removal)
    Xcox = pd.DataFrame(Xd, columns=meta["cox_cols"])
    sf = cph.predict_survival_function(Xcox)
    t = sf.index.values
    s = sf.iloc[:, 0].values
    return p_raw, p_cal, risk_decile, shap, (t, s)

# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
st.title("🫘 Kidney Waitlist Risk & Wait-Time Explorer")
st.caption(
    f"Calibrated XGBoost + Cox models on {meta['n_cohort']:,} kidney-alone "
    "waitlist registrations (OPTN STAR, 2015+). Research/education only — "
    "not for clinical use. Aggregates and models only; no row-level data.")

tab_pred, tab_batch, tab_cohort, tab_fair, tab_about = st.tabs(
    ["🔮 Predict", "📦 Batch score", "📊 Cohort", "⚖️ Fairness", "ℹ️ About"])

# --------------------------------------------------------------------------
# PREDICT
# --------------------------------------------------------------------------
with tab_pred:
    left, right = st.columns([1, 1.4], gap="large")
    o = meta["options"]; d = meta["defaults"]
    with left:
        st.subheader("Candidate at listing")
        with st.form("candidate"):
            age = st.slider("Age at listing", 18, 90, d["INIT_AGE"])
            sex = st.selectbox("Sex", o["GENDER"])
            eth = st.selectbox("Race / ethnicity", o["ETHCAT"],
                               format_func=eth_label)
            abo = st.selectbox("Blood type (ABO)", o["ABO"])
            dial = st.selectbox("On dialysis at listing?", o["ON_DIALYSIS"])
            cpra = st.slider("CPRA at listing (sensitization %)", 0.0, 100.0,
                             float(d["INIT_CPRA"]))
            bmi = st.slider("BMI", 15.0, 55.0, float(d["BMI_TCR"]))
            func = st.selectbox("Functional status code", o["FUNC_STAT_TCR"],
                                help=FUNC_HELP)
            stat = st.selectbox("Initial status code", o["INIT_STAT"])
            region = st.selectbox("OPTN region", o["REGION"])
            listing_date = st.date_input("Listing date",
                                         value=pd.Timestamp("2022-01-01"))
            go = st.form_submit_button("Predict", use_container_width=True,
                                       type="primary")
    with right:
        if go:
            era = ("AcuityCircles_2021+"
                   if pd.Timestamp(listing_date) >= pd.Timestamp("2021-03-15")
                   else "KAS_2015_2021")
            inp = {"INIT_AGE": age, "INIT_CPRA": cpra, "BMI_TCR": bmi,
                   "GENDER": sex, "ETHCAT": eth, "ABO": abo,
                   "ON_DIALYSIS": dial, "FUNC_STAT_TCR": func,
                   "INIT_STAT": stat, "REGION": region, "policy_era": era}
            p_raw, p_cal, decile, shap, (t, s) = predict(inp)
            base = meta["adverse_rate"]

            st.subheader("Predicted adverse-outcome risk")
            st.caption("Probability of death or removal-as-too-sick on the "
                       "waitlist (before transplant).")
            c1, c2, c3 = st.columns(3)
            c1.metric("Calibrated risk", f"{p_cal*100:.1f}%",
                      f"{(p_cal-base)*100:+.1f} pts vs base",
                      delta_color="inverse")
            c2.metric("Risk decile", f"{decile} / 10",
                      "1 = highest risk", delta_color="off")
            c3.metric("Cohort base rate", f"{base*100:.1f}%", delta_color="off")

            st.progress(min(p_cal / max(base*3, 0.3), 1.0))

            # SHAP drivers
            st.markdown("**Why — top drivers for this candidate**")
            top = shap[:8][::-1]
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            vals = [v for _, v in top]
            labs = [pretty_feature(f) for f, _ in top]
            colors = ["#C0392B" if v > 0 else ACCENT for v in vals]
            ax.barh(labs, vals, color=colors)
            ax.axvline(0, color="#888", lw=.8)
            ax.set_xlabel("SHAP (→ raises risk, ← lowers risk; log-odds)")
            ax.tick_params(labelsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption("Red pushes risk up, teal pulls it down. SHAP values are "
                       "additive contributions to the log-odds.")

            # wait-time curve
            st.markdown("**Estimated time to transplant**")
            p1 = 1 - np.interp(365, t, s)
            p2 = 1 - np.interp(730, t, s)
            below = t[s <= 0.5]
            med = int(below[0]) if len(below) else None
            w1, w2, w3 = st.columns(3)
            w1.metric("P(transplant ≤ 1 yr)", f"{p1*100:.0f}%")
            w2.metric("P(transplant ≤ 2 yr)", f"{p2*100:.0f}%")
            w3.metric("Est. median wait",
                      f"{med} d" if med else "> obs. window")
            fig2, ax2 = plt.subplots(figsize=(6.5, 3))
            ax2.plot(t, s, color=ACCENT, lw=2)
            ax2.fill_between(t, s, color=ACCENT, alpha=.08)
            ax2.set_xlim(0, 2000); ax2.set_ylim(0, 1)
            ax2.set_xlabel("days since listing")
            ax2.set_ylabel("P(not yet transplanted)")
            fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
            st.caption("Cause-specific estimate: death and removal are treated "
                       "as censoring, so this is the transplant rate **among "
                       "those still waiting**, not an absolute probability. "
                       "A competing-risks model is the recommended refinement.")
        else:
            st.info("Enter a candidate's listing details and press **Predict**.")

# --------------------------------------------------------------------------
# BATCH SCORE
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a CSV of candidates")
    st.caption("Upload a CSV with one row per candidate. Each row is scored for "
               "adverse-outcome risk, risk decile, and cause-specific transplant "
               "probability at 1 and 2 years. Runs locally; nothing leaves your machine.")

    req = NUMF + [c for c in CATF if c != "policy_era"]
    st.markdown("**Expected columns** (missing ones are imputed; extra columns are kept):")
    st.code(", ".join(req) + "   [+ optional INIT_DATE or policy_era]", language="text")

    # downloadable template with valid example values
    o, d = meta["options"], meta["defaults"]
    ex = {"INIT_AGE": [55, 68], "INIT_CPRA": [0.0, 80.0], "BMI_TCR": [27.0, 31.0],
          "GENDER": ["M", "F"], "ETHCAT": [o["ETHCAT"][0], o["ETHCAT"][1]],
          "ABO": [o["ABO"][0], o["ABO"][1]], "ON_DIALYSIS": ["N", "Y"],
          "FUNC_STAT_TCR": [o["FUNC_STAT_TCR"][0], o["FUNC_STAT_TCR"][0]],
          "INIT_STAT": [o["INIT_STAT"][0], o["INIT_STAT"][0]],
          "REGION": [o["REGION"][0], o["REGION"][1]],
          "INIT_DATE": ["2019-06-01", "2022-03-01"]}
    template_csv = pd.DataFrame(ex).to_csv(index=False)
    st.download_button("⬇ Download CSV template", template_csv,
                       "candidates_template.csv", "text/csv")

    up = st.file_uploader("Upload candidates CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            st.write(f"Loaded **{len(raw):,}** rows.")
            missing = [c for c in req if c not in raw.columns]
            if missing:
                st.warning(f"Missing columns (will be imputed): {', '.join(missing)}")
            scored = batch_predict(raw)

            base = meta["adverse_rate"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean risk", f"{scored['adverse_risk'].mean()*100:.1f}%",
                      f"{(scored['adverse_risk'].mean()-base)*100:+.1f} pts vs base",
                      delta_color="inverse")
            hi = int((scored["risk_decile"] <= 2).sum())
            m2.metric("High-risk (decile 1–2)", f"{hi:,}",
                      f"{hi/len(scored)*100:.0f}% of batch", delta_color="off")
            m3.metric("Rows scored", f"{len(scored):,}", delta_color="off")

            st.markdown("**Risk distribution**")
            fig, ax = plt.subplots(figsize=(8, 2.6))
            ax.hist(scored["adverse_risk"] * 100, bins=30, color=ACCENT)
            ax.axvline(base * 100, color="#C0392B", ls="--", lw=1,
                       label=f"base rate {base*100:.1f}%")
            ax.set_xlabel("predicted adverse risk (%)"); ax.set_ylabel("candidates")
            ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            st.markdown("**Scored candidates** (sorted by risk)")
            show = scored.sort_values("adverse_risk", ascending=False)
            st.dataframe(show, hide_index=True, use_container_width=True, height=340)
            st.download_button("⬇ Download scored CSV",
                               scored.to_csv(index=False),
                               "candidates_scored.csv", "text/csv", type="primary")
        except Exception as e:
            st.error(f"Could not score this file: {e}. Check that columns match "
                     "the template and try again.")
    else:
        st.info("Download the template, fill in your candidates, and upload it here. "
                "Tip: a slice of the extract CSV works directly.")

# --------------------------------------------------------------------------
# COHORT
# --------------------------------------------------------------------------
with tab_cohort:
    st.subheader("Cohort overview")
    a, b = st.columns(2)
    a.metric("Registrations", f"{meta['n_cohort']:,}")
    a.metric("Adverse rate", f"{meta['adverse_rate']*100:.1f}%")
    b.metric("Transplanted", f"{meta['transplant_rate']*100:.1f}%")
    b.metric("OPTN regions", len(meta["med_wait_region"]))

    st.markdown("**Waitlist outcome distribution**")
    oc = pd.Series(meta["outcome_counts"]).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(oc.index, oc.values, color=ACCENT)
    ax.set_xlabel("registrations")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**Median observed days-to-transplant by OPTN region**")
    mw = pd.Series({k: v for k, v in meta["med_wait_region"].items()})
    mw.index = mw.index.astype(str)
    mw = mw.sort_values()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(mw.index, mw.values, color=ACCENT)
    ax.set_ylabel("median days"); ax.set_xlabel("OPTN region")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption(f"Range {mw.min()}–{mw.max()} days — a nearly two-fold "
               "geographic spread in access.")

    st.markdown("**Time to transplant by policy era (Kaplan–Meier)**")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for era, cur in meta["km"].items():
        ax.plot(cur["t"], cur["s"], lw=2, label=era)
    ax.set_xlim(0, 2000); ax.set_ylim(0, 1)
    ax.set_xlabel("days since listing"); ax.set_ylabel("P(not yet transplanted)")
    ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# --------------------------------------------------------------------------
# FAIRNESS
# --------------------------------------------------------------------------
with tab_fair:
    f = meta["fairness"]
    st.subheader("Fairness audit")
    st.metric("Overall test AUC", f"{f['overall_auc']:.3f}")
    st.caption("Target: each subgroup's AUC within 0.05 of overall. Note that "
               "AUC *ranking* parity does not guarantee equal *access* — see "
               "the two-fold regional wait-time gap in the Cohort tab.")
    for dim in ["Sex", "Race/ethnicity", "Blood type", "OPTN region", "Age band"]:
        rows = f[dim]
        if not rows:
            continue
        df_dim = pd.DataFrame(rows)
        maxgap = df_dim["delta"].abs().max()
        flag = "✅ within 0.05" if maxgap <= 0.05 else "⚠️ exceeds 0.05"
        st.markdown(f"**{dim}** — max gap {maxgap:.3f} · {flag}")
        show = df_dim.rename(columns={"group": "subgroup", "n": "n",
                                      "auc": "AUC", "delta": "Δ vs overall"})
        st.dataframe(show, hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------
# ABOUT
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(f"""
### About this tool

Predicts two linked outcomes for a kidney-alone waitlist candidate from
**listing-time features only**:

1. **Adverse-outcome risk** — probability of death or removal-as-too-sick
   before transplant (calibrated XGBoost; isotonic post-hoc calibration).
2. **Time to transplant** — cause-specific Cox proportional-hazards survival
   curve, with death/removal treated as censoring.

**Cohort.** {meta['n_cohort']:,} kidney-alone registrations listed 2015+, from
the OPTN `KIDPAN` STAR file (June 2026 release). Cohort defined on the
*waitlisted* organ (`WL_ORG == KI`), not the transplanted organ.

**Leakage guard.** Only variables known at listing are used; transplant-record
fields (diagnosis, donor type, transplant date, …) are excluded.

**Model quality (held-out test).** ROC-AUC 0.726 · PR-AUC 0.284 · Brier 0.111
(calibrated, down from 0.212) · top-decile lift 2.51× · Cox C-index 0.629.

**Limitations.** Per-registration (not per-patient); static listing features;
`REM_CD` outcome mapping is a working hypothesis; the survival estimate ignores
competing risks. See the accompanying paper for the full treatment.

> ⚠️ **Not a clinical decision tool.** Research and educational use only. The
> data are governed by an OPTN Data Use Agreement; this app ships models and
> aggregate statistics only, never row-level records.
""")

st.divider()
st.caption("Kidney Waitlist Risk & Wait-Time Explorer · companion to the "
           "conference paper · models trained with train_and_save.py")
