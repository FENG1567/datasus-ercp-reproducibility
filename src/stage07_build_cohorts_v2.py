from __future__ import annotations

"""Corrected dual-cohort builder with explicit linkage and ICD-prefix rules."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


STRICT_STONE = {"K803", "K804", "K805"}
ANY_K80 = {"K800", "K801", "K802", "K803", "K804", "K805", "K808", "K809"}
DIAGNOSIS_COLUMNS = [
    "DIAG_PRINC", "DIAG_SECUN", "DIAGSEC1", "DIAGSEC2", "DIAGSEC3",
    "DIAGSEC4", "DIAGSEC5", "DIAGSEC6", "DIAGSEC7", "DIAGSEC8", "DIAGSEC9",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_icd(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return "" if text in {"", "NAN", "NONE", "0000"} else text


def diagnosis_set(row: pd.Series) -> set[str]:
    return {
        code
        for column in DIAGNOSIS_COLUMNS
        if (code := normalise_icd(row.get(column, "")))
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    linked = pq.read_table(args.linked).to_pandas()
    key = ["competence_month", "SP_CNES", "SP_NAIH"]
    if linked.duplicated(key).any():
        raise RuntimeError("linked input contains duplicate unique-AIH keys")
    for column in DIAGNOSIS_COLUMNS:
        if column not in linked.columns:
            linked[column] = pd.NA

    age_numeric = pd.to_numeric(linked.get("IDADE"), errors="coerce")
    age_unit = linked.get("COD_IDADE").astype(str).str.strip()
    linked["age_years"] = pd.NA
    linked.loc[age_unit.eq("4"), "age_years"] = age_numeric[age_unit.eq("4")]
    linked.loc[age_unit.eq("2"), "age_years"] = age_numeric[age_unit.eq("2")] / 12.0
    linked.loc[age_unit.eq("1"), "age_years"] = age_numeric[age_unit.eq("1")] / 365.25
    linked["age_years"] = pd.to_numeric(linked["age_years"], errors="coerce")
    linked["adult"] = linked["age_years"].ge(18)
    linked["principal"] = linked["DIAG_PRINC"].map(normalise_icd)
    linked["diag_set"] = linked.apply(diagnosis_set, axis=1)
    linked["k80345"] = linked["diag_set"].map(lambda codes: bool(codes & STRICT_STONE))
    linked["any_k80"] = linked["diag_set"].map(lambda codes: bool(codes & ANY_K80))
    linked["malignancy_any"] = linked["diag_set"].map(
        lambda codes: any(code.startswith(("C23", "C24")) for code in codes)
    )
    linked["malignancy_principal"] = linked["principal"].str.startswith(
        ("C23", "C24"), na=False
    )
    linked["k83_any"] = linked["diag_set"].map(
        lambda codes: any(code.startswith("K83") for code in codes)
    )

    cohort_a = linked.copy()
    cohort_b = linked[
        linked["link_class"].eq("exact")
        & linked["adult"]
        & linked["k80345"]
        & ~linked["malignancy_principal"]
    ].copy()

    def label(frame: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
        frame = frame.copy()
        frame["cohort"] = cohort_name
        frame["strata_pediatric"] = ~frame["adult"]
        frame["strata_any_k80"] = frame["any_k80"]
        frame["strata_k83"] = frame["k83_any"]
        frame["strata_malignancy_any"] = frame["malignancy_any"]
        frame["strata_malignancy_principal"] = frame["malignancy_principal"]
        return frame

    result = pd.concat([label(cohort_a, "A"), label(cohort_b, "B")], ignore_index=True)
    result = result.drop(columns=["diag_set"])
    duplicate_a = int(result[result["cohort"].eq("A")].duplicated(key).sum())
    duplicate_b = int(result[result["cohort"].eq("B")].duplicated(key).sum())
    b_subset = set(
        map(tuple, result.loc[result["cohort"].eq("B"), key].itertuples(index=False, name=None))
    ).issubset(
        set(map(tuple, result.loc[result["cohort"].eq("A"), key].itertuples(index=False, name=None)))
    )
    status = "PASS" if duplicate_a == 0 and duplicate_b == 0 and b_subset else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    excluded_malignancy = linked[
        linked["link_class"].eq("exact")
        & linked["adult"]
        & linked["k80345"]
        & linked["malignancy_principal"]
    ]
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "funnel": {
            "A_unique_aih": int(len(cohort_a)),
            "A_exact_linked": int(cohort_a["link_class"].eq("exact").sum()),
            "A_adult": int(cohort_a["adult"].sum()),
            "B_candidate_exact_k80345": int(
                (linked["link_class"].eq("exact") & linked["k80345"]).sum()
            ),
            "B_adult_exact_k80345": int(
                (
                    linked["link_class"].eq("exact")
                    & linked["adult"]
                    & linked["k80345"]
                ).sum()
            ),
            "B_excluded_principal_C23_C24_family": int(len(excluded_malignancy)),
            "B_final": int(len(cohort_b)),
        },
        "duplicates": {"A": duplicate_a, "B": duplicate_b},
        "b_subset_of_a": b_subset,
        "composition": {
            "A_by_year": {
                str(year): int(value)
                for year, value in cohort_a.groupby(
                    cohort_a["competence_month"].astype(str).str[:4]
                ).size().items()
            },
            "B_by_year": {
                str(year): int(value)
                for year, value in cohort_b.groupby(
                    cohort_b["competence_month"].astype(str).str[:4]
                ).size().items()
            },
            "B_principal_top10": {
                str(code): int(value)
                for code, value in cohort_b["principal"].value_counts().head(10).items()
            },
        },
        "definitions": {
            "A": "All unique AIHs containing procedure 0407030255, regardless of RD linkage.",
            "B": (
                "Exact RD-linked adults aged >=18 with K803/K804/K805 in an available "
                "diagnosis field, excluding any principal ICD code beginning C23 or C24."
            ),
            "malignancy_prefix_rule": "ICD family prefix C23* or C24*, not exact-string only.",
        },
        "output": str(args.output),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
