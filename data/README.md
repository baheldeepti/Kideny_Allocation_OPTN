# data/ — DUA-governed extracts (LOCAL ONLY)

These CSVs are **row-level** OPTN registry extracts governed by a Data Use
Agreement. Do **not** commit, upload, or share them outside the team.

| folder | cohort | rows | notes |
|---|---|---|---|
| `extract_2015_kidney_only/` | kidney-alone, listed 2015+ (`WL_ORG==KI`) | 494,862 | **primary** — used by analysis + app |
| `extract_2015_ki_kp/` | kidney + kidney-pancreas, 2015+ | 511,390 | includes KP combos |
| `extract_2010_ki_kp/` | kidney + kidney-pancreas, 2010+ | 701,496 | widest window |

Each folder has: `kidney_waitlist_analytic.csv`, `extract_summary.txt`,
`column_manifest.csv`.

Rebuild any of these from the raw STAR file with `extraction/build_analytic_extract.py`.
