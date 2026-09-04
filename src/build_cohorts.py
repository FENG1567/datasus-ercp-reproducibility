from __future__ import annotations

"""Dual-cohort build on linked ERCP AIH:
A = all indications; B = adult K80.3/K80.4/K80.5 (exclude principal C23/C24).
Diagnosis priority: DIAG_PRINC, then DIAG_SECUN, then DIAGSEC1..9.
Stratifiers: age<18, emergency, any-K80, K83, malignancy."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

K80 = {"K803", "K804", "K805"}
K80_ANY = {"K800", "K801", "K802", "K803", "K804", "K805", "K808", "K809"}
MALIGNANT = {"C23", "C24"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def diagnosis_set(row: pd.Series) -> set[str]:
    codes = set()
    for col in ("DIAG_PRINC", "DIAG_SECUN", "DIAGSEC1", "DIAGSEC2", "DIAGSEC3", "DIAGSEC4",
                "DIAGSEC5", "DIAGSEC6", "DIAGSEC7", "DIAGSEC8", "DIAGSEC9"):
        value = str(row.get(col, "")).strip().upper()
        if value and value != "0000" and value != "NAN":
            codes.add(value)
    return codes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    df = pq.read_table(args.linked).to_pandas()

    def age_years(row: pd.Series) -> float | None:
        try:
            age = float(row["IDADE"])
        except (TypeError, ValueError):
            return None
        unit = str(row.get("COD_IDADE", "")).strip()
        if unit == "4":  # SIH COD_IDADE: 4 = anos
            return age
        if unit == "1":  # dias
            return age / 365.25
        if unit == "2":  # meses
            return age / 12.0
        return None

    df["age_years"] = df.apply(age_years, axis=1)
    df["adult"] = df["age_years"].apply(lambda a: a is not None and a >= 18)
    df["diag_set"] = df.apply(diagnosis_set, axis=1)
    df["principal"] = df["DIAG_PRINC"].astype(str).str.strip().str.upper()

    # Cohort A: all indications
    # Cohort B: adult + K803/4/5 in any diagnostic position (priority:
    # principal > secondary > DIAGSEC*), exclude principal C23/C24
    def in_any(row: pd.Series, codes: set[str]) -> bool:
        return bool(codes & row["diag_set"])

    df["k80345"] = df.apply(lambda r: in_any(r, K80), axis=1)
    df["any_k80"] = df.apply(lambda r: in_any(r, K80_ANY), axis=1)
    df["malig_any"] = df.apply(lambda r: bool(MALIGNANT & r["diag_set"]), axis=1)
    df["malig_principal"] = df["principal"].isin(MALIGNANT)

    cohort_a = df.copy()

    cohort_b = df[
        df["adult"] & df["k80345"] & ~df["malig_principal"]
    ].copy()

    # Stratification layers
    def add_strata(d: pd.DataFrame, cohort_label: str) -> pd.DataFrame:
        d = d.copy()
        d["cohort"] = cohort_label
        d["strata_pediatric"] = ~d["adult"]
        d["strata_any_k80"] = d["any_k80"]
        d["strata_k83"] = d["diag_set"].apply(lambda s: bool({"K83"} & s))
        d["strata_malignancy_any"] = d["malig_any"]
        d["strata_malignancy_principal"] = d["malig_principal"]
        return d

    cohort_a = add_strata(cohort_a, "A")
    cohort_b = add_strata(cohort_b, "B")

    result = pd.concat([cohort_a, cohort_b], ignore_index=True)
    result = result.drop(columns=["diag_set"])

    # duplicates within a cohort are impossible (unique AIH key); verify
    dup_a = int(cohort_a.duplicated(["competence_month", "SP_CNES", "SP_NAIH"]).sum())
    dup_b = int(cohort_b.duplicated(["competence_month", "SP_CNES", "SP_NAIH"]).sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    # Funnel (uniqueness at cohort level)
    funnel = {
        "A_unique_aih": int(len(cohort_a)),
        "A_adult": int(cohort_a["adult"].sum()),
        "B_candidate_k80345": int(df[df["k80345"]].shape[0]),
        "B_adult_k80345": int(df[df["adult"] & df["k80345"]].shape[0]),
        "B_final": int(len(cohort_b)),
        "B_excluded_principal_malignant": int(
            df[df["adult"] & df["k80345"] & df["malig_principal"]].shape[0]
        ),
    }
    composition = {
        "A_by_year": {str(y): int(n) for y, n in cohort_a.groupby("competence_month").size().groupby(lambda m: m[:4]).sum().items()},
        "B_by_year": {str(y): int(n) for y, n in cohort_b.groupby("competence_month").size().groupby(lambda m: m[:4]).sum().items()},
        "B_by_diag_principal": {
            str(k): int(v) for k, v in cohort_b["principal"].value_counts().head(10).items()
        },
    }
    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS" if (dup_a == 0 and dup_b == 0) else "FAIL",
        "funnel": funnel,
        "duplicate_a": dup_a,
        "duplicate_b": dup_b,
        "composition": composition,
        "output": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0 if (dup_a == 0 and dup_b == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())