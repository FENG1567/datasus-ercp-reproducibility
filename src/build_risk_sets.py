from __future__ import annotations

"""Eligible hospital-month risk sets from CNES (no future information):
broad = SUS inpatient (LEITHOSP=1 & VINC_SUS=1) in the same month;
primary = broad + endoscopic/optical diagnostic service (SERAP08P=1) or
clinical treatment (SERAP09P=1);
strict = primary + >=1 endoscopist (PF CBO 225250) in the same month.
Capacity: SUS beds (LT QT_SUS), endoscopists, anesthesiologists, unique
medical CBOs per hospital-month from PF."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ENDOSCOPIST_CBO = "225250"
ANESTHESIA_PREFIX = "2251"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    st_frames = []
    for year_dir in sorted((args.cnes_dir / "ST").iterdir()):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.parquet")):
            df = pq.read_table(path, columns=["CNES", "competence_month", "LEITHOSP", "VINC_SUS", "SERAP08P", "SERAP09P", "TP_UNID"]).to_pandas()
            st_frames.append(df)
    st = pd.concat(st_frames, ignore_index=True)
    st["CNES"] = st["CNES"].astype(str).str.zfill(7)
    st["LEITHOSP"] = st["LEITHOSP"].astype(str).str.strip()
    st["VINC_SUS"] = st["VINC_SUS"].astype(str).str.strip()
    st["SERAP08P"] = st["SERAP08P"].astype(str).str.strip()
    st["SERAP09P"] = st["SERAP09P"].astype(str).str.strip()

    st["inpatient_sus"] = (st["LEITHOSP"] == "1") & (st["VINC_SUS"] == "1")
    st["endoscopy_service"] = st["SERAP08P"] == "1"
    st["clinical_treatment"] = st["SERAP09P"] == "1"
    st["broad"] = st["inpatient_sus"]
    st["primary"] = st["broad"] & (st["endoscopy_service"] | st["clinical_treatment"])
    print(f"ST hospital-months total: {len(st)}; broad: {int(st['broad'].sum())}; primary: {int(st['primary'].sum())}", flush=True)

    # beds from LT
    lt_frames = []
    for year_dir in sorted((args.cnes_dir / "LT").iterdir()):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.parquet")):
            df = pq.read_table(path, columns=["CNES", "competence_month", "QT_EXIST", "QT_SUS", "QT_NSUS"]).to_pandas()
            lt_frames.append(df)
    lt = pd.concat(lt_frames, ignore_index=True)
    lt["CNES"] = lt["CNES"].astype(str).str.zfill(7)
    for col in ("QT_EXIST", "QT_SUS", "QT_NSUS"):
        lt[col] = pd.to_numeric(lt[col], errors="coerce").fillna(0)
    beds = (
        lt.groupby(["CNES", "competence_month"], as_index=False)
        .agg(beds_total=("QT_EXIST", "sum"), beds_sus=("QT_SUS", "sum"))
    )
    st = st.merge(beds, on=["CNES", "competence_month"], how="left")
    st["beds_total"] = st["beds_total"].fillna(0)
    st["beds_sus"] = st["beds_sus"].fillna(0)

    # PF workforce per hospital-month
    pf_frames = []
    for year_dir in sorted((args.cnes_dir / "PF").iterdir()):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.parquet")):
            df = pq.read_table(path, columns=["CNES", "competence_month", "CBO", "CPF_PROF"]).to_pandas()
            df = df[df["CBO"].astype(str).str.startswith(("2251", "2252", "2253"))]
            if len(df):
                pf_frames.append(df)
    pf = pd.concat(pf_frames, ignore_index=True) if pf_frames else pd.DataFrame(
        columns=["CNES", "competence_month", "CBO", "CPF_PROF"])
    pf["CNES"] = pf["CNES"].astype(str).str.zfill(7)
    pf["CBO"] = pf["CBO"].astype(str).str.strip()
    pf["endoscopist"] = pf["CBO"] == ENDOSCOPIST_CBO
    pf["anesthesia"] = pf["CBO"].str.startswith(ANESTHESIA_PREFIX)
    workforce = (
        pf.groupby(["CNES", "competence_month"], as_index=False)
        .agg(
            endoscopists=("CPF_PROF", "nunique"),
            anesthesiologists=("CPF_PROF", "nunique"),
            medical_cbos=("CBO", "nunique"),
            medical_professionals=("CPF_PROF", "nunique"),
        )
    )
    pf_grouped = pf[pf["endoscopist"]].groupby(["CNES", "competence_month"])["CPF_PROF"].nunique().reset_index(name="endoscopists")
    pf_an = pf[pf["anesthesia"]].groupby(["CNES", "competence_month"])["CPF_PROF"].nunique().reset_index(name="anesthesiologists")
    pf_cbo = pf.groupby(["CNES", "competence_month"])["CBO"].nunique().reset_index(name="medical_cbos")
    pf_prof = pf.groupby(["CNES", "competence_month"])["CPF_PROF"].nunique().reset_index(name="medical_professionals")
    workforce = pf_grouped.merge(pf_an, on=["CNES", "competence_month"], how="outer") \
        .merge(pf_cbo, on=["CNES", "competence_month"], how="outer") \
        .merge(pf_prof, on=["CNES", "competence_month"], how="outer")
    for col in ("endoscopists", "anesthesiologists", "medical_cbos", "medical_professionals"):
        workforce[col] = workforce[col].fillna(0)
    st = st.merge(workforce, on=["CNES", "competence_month"], how="left")
    for col in ("endoscopists", "anesthesiologists", "medical_cbos", "medical_professionals"):
        st[col] = st[col].fillna(0)

    st["strict"] = st["primary"] & (st["endoscopists"] >= 1)

    out_cols = ["CNES", "competence_month", "inpatient_sus", "endoscopy_service",
                "clinical_treatment", "broad", "primary", "strict",
                "beds_total", "beds_sus", "endoscopists", "anesthesiologists",
                "medical_cbos", "medical_professionals", "TP_UNID"]
    result = st[out_cols].sort_values(["CNES", "competence_month"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS",
        "st_hospital_months": int(len(st)),
        "broad_hospital_months": int(st["broad"].sum()),
        "primary_hospital_months": int(st["primary"].sum()),
        "strict_hospital_months": int(st["strict"].sum()),
        "broad_hospitals": int(st[st["broad"]]["CNES"].nunique()),
        "primary_hospitals": int(st[st["primary"]]["CNES"].nunique()),
        "strict_hospitals": int(st[st["strict"]]["CNES"].nunique()),
        "unique_key_duplicates": int(result.duplicated(["CNES", "competence_month"]).sum()),
        "no_future_info": True,
        "output": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())