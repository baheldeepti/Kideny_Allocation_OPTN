#!/usr/bin/env python3
"""
Build a Colab-sized analytic extract from the OPTN KIDPAN_DATA STAR file.

Run this locally, on the machine holding the STAR file. It never needs
network access and it never writes the full dataset anywhere new.

    python build_analytic_extract.py \
        --input  "/path/to/KIDPAN_DATA.DAT" \
        --outdir "./extract" \
        --min-year 2010

Output:
    extract/kidney_waitlist_analytic.csv    <- share with the team
    extract/extract_summary.txt             <- record counts, QA log
    extract/column_manifest.csv             <- which requested cols existed

IMPORTANT
---------
The STAR file is governed by a Data Use Agreement. Sharing this extract
with anyone still counts as releasing the data. Confirm written OPTN
approval before distributing, and never commit the output to a public repo.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Columns to retain. Verified present in the KIDPAN_DATA data dictionary.
# --------------------------------------------------------------------------

ID_COLS = [
    "PT_CODE",            # encrypted recipient identifier (tracks across listings)
    "WL_ID_CODE",         # encrypted registration identifier
    "TRR_ID_CODE",        # encrypted transplant identifier
    "DONOR_ID",
]

COHORT_COLS = [
    "WL_ORG",             # organ the candidate was WAITLISTED for  <- cohort key
    "ORGAN",              # organ actually transplanted (null until transplant)
    "WLKI",               # listed-for-kidney flag (98% null here; not a filter)
    "INIT_DATE",          # date placed on waiting list
    "END_DATE",           # earliest of removal / transplant / death / cutoff
    "REM_CD",             # reason for removal  <- primary outcome source
]

OUTCOME_COLS = [
    "DAYSWAIT_CHRON",     # total days on waiting list incl. inactive time
    "DAYSWAIT_ALLOC",     # time used for allocation priority
    "PTIME",              # patient survival days (composite death date)
    "PSTATUS",            # 1 = dead, 0 = alive
    "COMPOSITE_DEATH_DATE",
    "TX_DATE",
    "DON_TY",             # deceased vs living donor
]

FEATURE_COLS = [
    "INIT_AGE",           # age at listing
    "GENDER",
    "ETHCAT",             # ethnicity category
    "ABO",                # blood group at registration
    "INIT_CPRA",          # calculated PRA at listing
    "END_CPRA",
    "ON_DIALYSIS",
    "DIALYSIS_DATE",
    "BMI_TCR",
    "FUNC_STAT_TCR",      # functional status at registration
    "PREV_TX",
    "INIT_STAT",
    "END_STAT",
    "DIAG_KI",            # primary kidney diagnosis
    "REGION",             # UNOS region
    "LISTING_CTR_CODE",
    "MULTIORG",
    "A2A2B_ELIGIBILITY",
]

KEEP = ID_COLS + COHORT_COLS + OUTCOME_COLS + FEATURE_COLS

# --------------------------------------------------------------------------
# Outcome mapping from REM_CD.
#
# NOTE FOR THE TEAM: treat this as a starting hypothesis, not settled truth.
# Validate every bucket against the REMCD format tab in the data dictionary
# before you model on it. The multi-listing codes in particular deserve
# scrutiny -- a candidate transplanted at another center is not a failure
# of this registration, but they are also not an event you observed.
# --------------------------------------------------------------------------

REM_CD_TRANSPLANTED = {2, 4, 15, 18, 19, 41, 42, 43, 44, 45}
REM_CD_DIED         = {8, 21, 23}
REM_CD_TOO_SICK     = {5, 13}
REM_CD_TX_ELSEWHERE = {3, 14, 22}
REM_CD_ADMIN        = {7, 9, 10, 11, 16, 17, 20, 24, 40}
# REM_CD null / missing  ->  still waiting (right-censored)


def classify_outcome(rem_cd):
    """Map REM_CD to a coarse outcome bucket."""
    if pd.isna(rem_cd):
        return "still_waiting"
    try:
        code = int(float(rem_cd))
    except (ValueError, TypeError):
        return "unknown"
    if code in REM_CD_TRANSPLANTED:
        return "transplanted"
    if code in REM_CD_DIED:
        return "died"
    if code in REM_CD_TOO_SICK:
        return "removed_too_sick"
    if code in REM_CD_TX_ELSEWHERE:
        return "transplanted_elsewhere"
    if code in REM_CD_ADMIN:
        return "removed_administrative"
    return "unknown"


def column_names_from_htm(htm_path):
    """OPTN STAR .DAT files ship WITHOUT a header row. The ordered column
    layout lives in the sibling SAS-output .htm dictionary, one table row per
    variable: Obs | LABEL | FORMAT | LENGTH | TYPE | START | END. Parse the
    LABEL cell of every numbered row so names line up 1:1 with the data fields.
    """
    html = Path(htm_path).read_text(encoding="latin-1")
    names = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = []
        for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row,
                               flags=re.S | re.I):
            cell = re.sub(r"<[^>]+>", " ", cell)
            cell = cell.replace("&nbsp;", " ")
            cells.append(re.sub(r"\s+", " ", cell).strip())
        # a data row is "<obs number> <LABEL> ..."
        if len(cells) >= 2 and cells[0].isdigit():
            names.append((int(cells[0]), cells[1]))
    names.sort()
    seqs = [n for n, _ in names]
    if seqs != list(range(1, len(seqs) + 1)):
        raise RuntimeError(
            f"Column layout in {Path(htm_path).name} is not contiguous "
            f"1..N -- parser is misaligned, refusing to guess.")
    return [label for _, label in names]


def resolve_columns(dat_path):
    """Find the sibling .htm dictionary for a .DAT file and return its
    ordered column names plus the encoding to read the .DAT under."""
    htm = dat_path.with_suffix(".htm")
    if not htm.exists():
        raise RuntimeError(
            f"Column dictionary not found next to the data file: expected "
            f"{htm.name}. STAR .DAT files carry no header, so it is required.")
    names = column_names_from_htm(htm)
    print(f"Column dictionary: {htm.name}  ({len(names)} columns)")
    return names, "latin-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to KIDPAN_DATA.DAT")
    ap.add_argument("--outdir", default="./extract")
    ap.add_argument("--min-year", type=int, default=2010,
                    help="Keep registrations listed on or after Jan 1 of this year")
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Input not found: {src}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {src}  ({src.stat().st_size / 1e9:.2f} GB)")

    # ---- load the ordered column layout from the sibling .htm dictionary --
    # (STAR .DAT files have NO header row; names must come from the dictionary)
    all_names, encoding = resolve_columns(src)
    available = all_names
    print(f"Encoding: {encoding}")
    print(f"Columns in file: {len(available)}")

    present = [c for c in KEEP if c in available]
    missing = [c for c in KEEP if c not in available]

    if missing:
        print(f"\n  {len(missing)} requested columns not in this file "
              f"(quarterly versions differ):")
        for c in missing:
            print(f"    - {c}")

    pd.DataFrame({
        "column": KEEP,
        "present_in_file": [c in available for c in KEEP],
    }).to_csv(outdir / "column_manifest.csv", index=False)

    if "INIT_DATE" not in present:
        sys.exit("INIT_DATE is required for the cohort filter but is absent.")

    print(f"\nRetaining {len(present)} columns. Streaming in chunks...\n")

    # ---- stream and filter ----------------------------------------------
    kept_frames = []
    rows_read = 0
    rows_kept = 0
    cutoff = pd.Timestamp(year=args.min_year, month=1, day=1)

    # header=None + names: the file has no header, so supply the full ordered
    # layout and select the subset we keep. na_values="." maps the STAR missing
    # sentinel to NaN so the WL_ID_CODE filter and REM_CD outcome logic behave.
    reader = pd.read_csv(
        src, sep="\t", header=None, names=all_names, usecols=present,
        chunksize=args.chunksize, encoding=encoding, low_memory=False,
        na_values=["."], keep_default_na=True,
    )

    for i, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)

        # waitlist registrations only -- transplant-only rows have no WL_ID_CODE
        if "WL_ID_CODE" in chunk.columns:
            chunk = chunk[chunk["WL_ID_CODE"].notna()]

        # kidney-alone registrations only. Filter on WL_ORG -- the WAITLISTED
        # organ, populated for every registration -- NOT ORGAN, which is the
        # transplanted organ and is null until a transplant occurs (filtering
        # on ORGAN silently collapses the cohort to transplants only).
        # WL_ORG == "KI" keeps kidney-alone and DROPS kidney-pancreas ("KP"),
        # pancreas-alone ("PA") and islet ("PI").
        if "WL_ORG" in chunk.columns:
            wl_org = chunk["WL_ORG"].astype(str).str.upper().str.strip()
            chunk = chunk[wl_org == "KI"]

        # listing-date window
        chunk["INIT_DATE"] = pd.to_datetime(chunk["INIT_DATE"], errors="coerce")
        chunk = chunk[chunk["INIT_DATE"] >= cutoff]

        if len(chunk):
            kept_frames.append(chunk)
            rows_kept += len(chunk)

        if i % 10 == 0:
            print(f"  chunk {i:>4}: read {rows_read:>10,}  kept {rows_kept:>9,}")

    if not kept_frames:
        sys.exit("No rows survived the filters. Check --min-year and ORGAN values.")

    df = pd.concat(kept_frames, ignore_index=True)
    del kept_frames

    # ---- derive outcome + time-to-event ---------------------------------
    if "REM_CD" in df.columns:
        df["outcome"] = df["REM_CD"].apply(classify_outcome)
        df["event_adverse"] = df["outcome"].isin(
            ["died", "removed_too_sick"]).astype(int)
        df["event_transplant"] = (df["outcome"] == "transplanted").astype(int)
        df["censored"] = (df["outcome"] == "still_waiting").astype(int)

    if "END_DATE" in df.columns:
        df["END_DATE"] = pd.to_datetime(df["END_DATE"], errors="coerce")
        df["days_to_event"] = (df["END_DATE"] - df["INIT_DATE"]).dt.days
        # negative or absurd durations signal data-entry problems, not reality
        bad = df["days_to_event"] < 0
        if bad.any():
            print(f"\n  {bad.sum():,} rows have END_DATE before INIT_DATE "
                  f"-- set to NaN, investigate before modeling")
            df.loc[bad, "days_to_event"] = pd.NA

    # ---- write ----------------------------------------------------------
    out_csv = outdir / "kidney_waitlist_analytic.csv"
    df.to_csv(out_csv, index=False)
    size_mb = out_csv.stat().st_size / 1e6

    lines = [
        "OPTN KIDPAN_DATA -- Analytic Extract Summary",
        "=" * 48,
        f"Source file:        {src.name}",
        f"Source size:        {src.stat().st_size / 1e9:.2f} GB",
        f"Rows read:          {rows_read:,}",
        f"Rows kept:          {len(df):,}",
        f"Columns kept:       {len(df.columns)}",
        f"Listing window:     {args.min_year}-01-01 onward",
        f"Cohort:             kidney-alone waitlist registrations (WL_ORG == KI)",
        f"Output:             {out_csv.name}  ({size_mb:.1f} MB)",
        "",
    ]

    if "outcome" in df.columns:
        lines.append("Outcome distribution")
        lines.append("-" * 48)
        counts = df["outcome"].value_counts(dropna=False)
        for label, n in counts.items():
            lines.append(f"  {label:<26} {n:>9,}  ({n / len(df) * 100:5.1f}%)")
        lines.append("")

    lines.append("Missingness by column (%)")
    lines.append("-" * 48)
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    for col, pct in miss.items():
        lines.append(f"  {col:<26} {pct:6.2f}")

    summary = "\n".join(lines)
    (outdir / "extract_summary.txt").write_text(summary)

    print("\n" + summary)
    print(f"\nWrote {out_csv}  ({size_mb:.1f} MB)")
    if size_mb > 400:
        print("  Still large for Colab free tier -- consider narrowing "
              "--min-year or dropping columns.")


if __name__ == "__main__":
    main()
