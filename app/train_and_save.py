#!/usr/bin/env python3
"""
Train and persist the deployable model artifacts for the Streamlit app.

Reuses the paper's pipeline: leakage-guarded listing-time features, a single
shared preprocessor, a class-imbalance-weighted XGBoost classifier with
post-hoc isotonic calibration, and a Cox proportional-hazards model for
time-to-transplant on the SAME feature space so one preprocessor serves both.

Persists ONLY model artifacts and aggregate metadata (no row-level records),
consistent with the OPTN Data Use Agreement.

    python train_and_save.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from lifelines import CoxPHFitter, KaplanMeierFitter

SRC = (Path(__file__).resolve().parents[1] / "data"
       / "extract_2015_kidney_only" / "kidney_waitlist_analytic.csv")
OUT = Path(__file__).resolve().parent / "models"
OUT.mkdir(parents=True, exist_ok=True)
RS = 42

NUM = ["INIT_AGE", "INIT_CPRA", "BMI_TCR"]
CAT = ["GENDER", "ETHCAT", "ABO", "ON_DIALYSIS", "FUNC_STAT_TCR",
       "INIT_STAT", "REGION", "policy_era"]
FEATURES = NUM + CAT

ETHCAT_MAP = {"1": "White", "2": "Black", "4": "Hispanic", "5": "Asian",
              "6": "Amer Indian", "7": "Pacific Islander", "9": "Multiracial",
              "998": "Unknown", "6.0": "Amer Indian"}


def load():
    cols = list(set(NUM + ["GENDER", "ETHCAT", "ABO", "ON_DIALYSIS",
              "FUNC_STAT_TCR", "INIT_STAT", "REGION", "INIT_DATE",
              "outcome", "event_adverse", "event_transplant", "days_to_event"]))
    df = pd.read_csv(SRC, usecols=cols, low_memory=False)
    df["INIT_DATE"] = pd.to_datetime(df["INIT_DATE"], errors="coerce")
    df["policy_era"] = np.where(df["INIT_DATE"] >= pd.Timestamp("2021-03-15"),
                                "AcuityCircles_2021+", "KAS_2015_2021")
    for c in ["GENDER", "ETHCAT", "ABO", "ON_DIALYSIS", "FUNC_STAT_TCR",
              "INIT_STAT", "REGION"]:
        df[c] = df[c].astype("string").fillna("MISSING").astype(object)
    return df


def preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def main():
    print("loading extract ...")
    df = load()
    y = df["event_adverse"].astype(int).values

    # split: train -> (fit, calib), plus test for honest metrics
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(
        df[FEATURES], y, df, test_size=0.25, stratify=y, random_state=RS)
    Xfit, Xcal, yfit, ycal = train_test_split(
        Xtr, ytr, test_size=0.20, stratify=ytr, random_state=RS)

    print("fitting preprocessor + XGBoost ...")
    pre = preprocessor()
    Xfit_t = pre.fit_transform(Xfit)
    Xcal_t = pre.transform(Xcal)
    Xte_t = pre.transform(Xte)
    spw = (yfit == 0).sum() / (yfit == 1).sum()
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="aucpr", scale_pos_weight=spw,
        random_state=RS, n_jobs=-1)
    clf.fit(Xfit_t, yfit)

    print("fitting isotonic calibrator ...")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clf.predict_proba(Xcal_t)[:, 1], ycal)

    # calibrated test probabilities -> decile thresholds + overall AUC
    p_raw_te = clf.predict_proba(Xte_t)[:, 1]
    p_cal_te = iso.transform(p_raw_te)
    overall_auc = roc_auc_score(yte, p_cal_te)
    decile_edges = np.quantile(p_cal_te, np.linspace(0, 1, 11)).tolist()

    print("fitting Cox time-to-transplant model ...")
    s = dtr[dtr["days_to_event"].notna() & (dtr["days_to_event"] >= 0)].copy()
    s = s.sample(n=min(120_000, len(s)), random_state=RS)
    Xs = pre.transform(s[FEATURES])
    Xs = pd.DataFrame(Xs.toarray() if hasattr(Xs, "toarray") else Xs)
    fcols = [f"f{i}" for i in range(Xs.shape[1])]
    Xs.columns = fcols
    Xs["_dur"] = s["days_to_event"].clip(lower=1).values
    Xs["_evt"] = s["event_transplant"].astype(int).values
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(Xs, duration_col="_dur", event_col="_evt")

    # ---- aggregate metadata (DUA-safe) -------------------------------------
    feat_names = list(pre.get_feature_names_out())

    # subgroup AUCs on test (calibrated)
    te = Xte.copy(); te["y"] = yte; te["p"] = p_cal_te
    te["ETH_LABEL"] = te["ETHCAT"].map(lambda v: ETHCAT_MAP.get(str(v), "Other/Unknown"))
    te["age_band"] = pd.cut(te["INIT_AGE"], [0, 35, 50, 65, 120],
                            labels=["<35", "35-49", "50-64", "65+"])
    def subgroups(col):
        out = []
        for val, g in te.groupby(col, observed=True):
            if g["y"].nunique() < 2 or len(g) < 500:
                continue
            out.append({"group": str(val), "n": int(len(g)),
                        "auc": round(float(roc_auc_score(g["y"], g["p"])), 3),
                        "delta": round(float(roc_auc_score(g["y"], g["p"]) - overall_auc), 3)})
        return sorted(out, key=lambda r: r["auc"])
    fairness = {"overall_auc": round(float(overall_auc), 3),
                "Sex": subgroups("GENDER"),
                "Race/ethnicity": subgroups("ETH_LABEL"),
                "Blood type": subgroups("ABO"),
                "OPTN region": subgroups("REGION"),
                "Age band": subgroups("age_band")}

    # cohort-level aggregates
    outcome_counts = df["outcome"].value_counts().to_dict()
    med_wait_region = (df[df["event_transplant"] == 1]
                       .groupby("REGION")["days_to_event"].median()
                       .dropna().round(0).astype(int).to_dict())
    med_wait_region = {str(k): int(v) for k, v in med_wait_region.items()}

    # KM curves by policy era (aggregate arrays for plotting)
    km = {}
    for era, g in s.groupby("policy_era"):
        kmf = KaplanMeierFitter()
        kmf.fit(g["days_to_event"].clip(lower=1), g["event_transplant"].astype(int))
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["t", "s"]
        sf = sf[sf["t"] <= 2000]
        km[era] = {"t": sf["t"].round(1).tolist(), "s": sf["s"].round(4).tolist()}

    # input options + defaults for the UI
    def opts(col, top=None):
        vc = df[col].value_counts()
        vals = [str(x) for x in vc.index if str(x) != "MISSING"]
        return vals[:top] if top else vals
    options = {
        "GENDER": ["M", "F"],
        "ETHCAT": [c for c in ["1", "2", "4", "5", "6", "7", "9"] if c in set(df["ETHCAT"])],
        "ABO": [c for c in ["O", "A", "B", "AB", "A1", "A2", "A1B", "A2B"] if c in set(df["ABO"])],
        "ON_DIALYSIS": ["Y", "N"],
        "FUNC_STAT_TCR": sorted(opts("FUNC_STAT_TCR"),
                                key=lambda x: (len(x), x)),
        "INIT_STAT": opts("INIT_STAT", top=12),
        "REGION": [str(r) for r in sorted(
            df["REGION"].dropna().unique(),
            key=lambda x: int(float(x)) if str(x).replace('.','').isdigit() else 99)
            if str(r) != "MISSING"],
    }
    defaults = {"INIT_AGE": int(df["INIT_AGE"].median()),
                "INIT_CPRA": 0.0,
                "BMI_TCR": round(float(df["BMI_TCR"].median()), 1)}
    meta = {
        "n_cohort": int(len(df)),
        "adverse_rate": round(float(df["event_adverse"].mean()), 4),
        "transplant_rate": round(float(df["event_transplant"].mean()), 4),
        "outcome_counts": {k: int(v) for k, v in outcome_counts.items()},
        "decile_edges": decile_edges,
        "fairness": fairness,
        "med_wait_region": med_wait_region,
        "km": km,
        "options": options,
        "defaults": defaults,
        "ethcat_map": ETHCAT_MAP,
        "features": {"num": NUM, "cat": CAT},
        "feat_names": feat_names,
        "cox_cols": fcols,
    }

    print("persisting artifacts ...")
    joblib.dump(pre, OUT / "preprocessor.joblib")
    joblib.dump(clf, OUT / "xgb_classifier.joblib")
    joblib.dump(iso, OUT / "isotonic.joblib")
    joblib.dump(cph, OUT / "cox_model.joblib")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done. overall calibrated AUC={overall_auc:.3f}. artifacts -> {OUT}")


if __name__ == "__main__":
    main()
