from __future__ import annotations

"""municipality_month and patient_flow_year tables:
- municipality: treated counts by residence municipality x month (cohorts A/B),
  treated municipality x month (care location)
- patient_flow_year: residence municipality -> treating CNES, weighted by
  unique AIHs within calendar year (cohort B primary; A supportive)"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--out-municipality", type=Path, required=True)
    parser.add_argument("--out-flow", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    df = pq.read_table(args.cohorts).to_pandas()
    df["res_municipio"] = df["MUNIC_RES"].astype(str).str.strip().str.zfill(6)
    df["treat_municipio"] = df["MUNIC_MOV"].astype(str).str.strip().str.zfill(6)
    df["year"] = df["competence_month"].str[:4]

    # municipality_month: residence municipality x month, counts by cohort
    rows = []
    for cohort_label in ("A", "B"):
        sub = df[df["cohort"] == cohort_label]
        res = sub.groupby(["res_municipio", "competence_month"]).size().rename("ercp_count").reset_index()
        res["cohort"] = cohort_label
        res["basis"] = "residence"
        rows.append(res)
    municipality = pd.concat(rows, ignore_index=True)
    municipality = municipality.sort_values(["cohort", "res_municipio", "competence_month"]).reset_index(drop=True)
    args.out_municipality.parent.mkdir(parents=True, exist_ok=True)
    municipality.to_parquet(args.out_municipality, index=False)

    # patient_flow_year: residence municipality -> treating CNES
    flow_rows = []
    for cohort_label in ("A", "B"):
        sub = df[df["cohort"] == cohort_label]
        flow = (
            sub.groupby(["res_municipio", "SP_CNES", "year"], as_index=False)
            .size()
            .rename(columns={"size": "n_aih"})
        )
        flow["cohort"] = cohort_label
        flow_rows.append(flow)
    patient_flow = pd.concat(flow_rows, ignore_index=True)
    patient_flow = patient_flow.sort_values(["cohort", "res_municipio", "SP_CNES", "year"]).reset_index(drop=True)
    args.out_flow.parent.mkdir(parents=True, exist_ok=True)
    patient_flow.to_parquet(args.out_flow, index=False)

    # conservation: sum of flow n_aih must equal cohort sizes
    flow_sum = {c: int(patient_flow[patient_flow["cohort"] == c]["n_aih"].sum()) for c in ("A", "B")}
    cohort_size = {c: int(df[df["cohort"] == c].shape[0]) for c in ("A", "B")}
    conservation_ok = flow_sum == cohort_size

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS" if conservation_ok else "FAIL",
        "municipality_rows": int(len(municipality)),
        "municipality_key_duplicates": int(municipality.duplicated(["cohort", "res_municipio", "competence_month"]).sum()),
        "patient_flow_rows": int(len(patient_flow)),
        "patient_flow_key_duplicates": int(patient_flow.duplicated(["cohort", "res_municipio", "SP_CNES", "year"]).sum()),
        "flow_conservation": {"flow_sum": flow_sum, "cohort_size": cohort_size, "ok": conservation_ok},
        "out_municipality": str(args.out_municipality),
        "out_flow": str(args.out_flow),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0 if conservation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())