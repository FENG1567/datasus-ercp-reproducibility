from __future__ import annotations

"""Stage-7 corrected Stage-3 freeze with explicit dual-cohort conservation."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ercp-aih", type=Path, required=True)
    parser.add_argument("--linked", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--hospital-month", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--municipality", type=Path, required=True)
    parser.add_argument("--patient-flow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    ercp = pq.read_table(
        args.ercp_aih,
        columns=["competence_month", "SP_CNES", "SP_NAIH", "matching_detail_row_count"],
    ).to_pandas()
    linked = pq.read_table(
        args.linked,
        columns=["competence_month", "SP_CNES", "SP_NAIH", "link_class"],
    ).to_pandas()
    cohorts = pq.read_table(
        args.cohorts,
        columns=["cohort", "competence_month", "SP_CNES", "SP_NAIH"],
    ).to_pandas()
    hospital_month = pq.read_table(
        args.hospital_month,
        columns=["SP_CNES", "competence_month", "ercp_count"],
    ).to_pandas()
    eligible = pq.read_table(
        args.eligible,
        columns=["CNES", "competence_month"],
    ).to_pandas()
    municipality = pq.read_table(
        args.municipality,
        columns=["cohort", "res_municipio", "competence_month", "ercp_count"],
    ).to_pandas()
    flow = pq.read_table(
        args.patient_flow,
        columns=["cohort", "res_municipio", "SP_CNES", "year", "n_aih"],
    ).to_pandas()

    key = ["competence_month", "SP_CNES", "SP_NAIH"]
    cohort_sizes = {
        label: int(cohorts[cohorts["cohort"].eq(label)].shape[0])
        for label in ("A", "B")
    }
    flow_sums = {
        label: int(flow.loc[flow["cohort"].eq(label), "n_aih"].sum())
        for label in ("A", "B")
    }
    municipality_sums = {
        label: int(
            municipality.loc[municipality["cohort"].eq(label), "ercp_count"].sum()
        )
        for label in ("A", "B")
    }
    link_counts = {
        str(label): int(value)
        for label, value in linked["link_class"].value_counts(dropna=False).items()
    }
    exact_link_rate = link_counts.get("exact", 0) / len(linked) if len(linked) else 0.0

    checks = {
        "ercp_aih_unique_key": not ercp.duplicated(key).any(),
        "ercp_aih_nonnegative_detail_counts": bool(
            ercp["matching_detail_row_count"].ge(1).all()
        ),
        "linked_unique_key": not linked.duplicated(key).any(),
        "linked_conservation_to_ercp_aih": len(linked) == len(ercp),
        "exact_link_rate_ge_0_98": exact_link_rate >= 0.98,
        "cohort_a_unique_key": not cohorts[cohorts["cohort"].eq("A")].duplicated(key).any(),
        "cohort_b_unique_key": not cohorts[cohorts["cohort"].eq("B")].duplicated(key).any(),
        "cohort_a_conservation_to_ercp_aih": cohort_sizes["A"] == len(ercp),
        "cohort_b_subset_of_a": set(
            map(tuple, cohorts.loc[cohorts["cohort"].eq("B"), key].itertuples(index=False, name=None))
        ).issubset(
            set(map(tuple, cohorts.loc[cohorts["cohort"].eq("A"), key].itertuples(index=False, name=None)))
        ),
        "hospital_month_unique_key": not hospital_month.duplicated(
            ["SP_CNES", "competence_month"]
        ).any(),
        "hospital_month_counts_nonnegative": bool(hospital_month["ercp_count"].ge(0).all()),
        "eligible_unique_key": not eligible.duplicated(["CNES", "competence_month"]).any(),
        "municipality_unique_key": not municipality.duplicated(
            ["cohort", "res_municipio", "competence_month"]
        ).any(),
        "flow_unique_key": not flow.duplicated(
            ["cohort", "res_municipio", "SP_CNES", "year"]
        ).any(),
        "flow_conservation_a": flow_sums["A"] == cohort_sizes["A"],
        "flow_conservation_b": flow_sums["B"] == cohort_sizes["B"],
        "municipality_conservation_a": municipality_sums["A"] == cohort_sizes["A"],
        "municipality_conservation_b": municipality_sums["B"] == cohort_sizes["B"],
    }
    all_ok = all(value is True for value in checks.values())
    files = {
        "ercp_aih": args.ercp_aih,
        "ercp_aih_linked": args.linked,
        "ercp_cohorts": args.cohorts,
        "hospital_month": args.hospital_month,
        "eligible_hospital_month": args.eligible,
        "municipality_month": args.municipality,
        "patient_flow_year": args.patient_flow,
    }
    manifest = {
        "schema_version": "2.0",
        "frozen_at": utc_now(),
        "status": "FROZEN" if all_ok else "NOT_FROZEN",
        "checks": checks,
        "counts": {
            "ercp_aih": len(ercp),
            "link_class": link_counts,
            "exact_link_rate": exact_link_rate,
            "cohort": cohort_sizes,
            "flow": flow_sums,
            "municipality": municipality_sums,
        },
        "tables": {name: str(path) for name, path in files.items()},
        "hashes": {name: sha256_stream(path) for name, path in files.items()},
    }
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": "PASS" if all_ok else "FAIL",
        "checks": checks,
        "counts": manifest["counts"],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
