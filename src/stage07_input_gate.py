from __future__ import annotations

"""Stage-7 entrance gate for SIH partition completeness and window validity."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


UFS = "AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO".split()
YEARS = range(2021, 2026)
MONTHS = range(1, 13)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(dataset_dir: Path, dataset: str) -> pd.DataFrame:
    observed: dict[tuple[str, int, int], Path] = {}
    pattern = re.compile(r"([A-Z]{2})_(20\d{2})(\d{2})\.parquet$", re.I)
    for path in dataset_dir.rglob("*.parquet"):
        match = pattern.search(path.name)
        if match:
            observed[(match.group(1).upper(), int(match.group(2)), int(match.group(3)))] = path
    rows = []
    for uf in UFS:
        for year in YEARS:
            for month in MONTHS:
                path = observed.get((uf, year, month))
                rows.append(
                    {
                        "dataset": dataset,
                        "uf": uf,
                        "year": year,
                        "month": month,
                        "present": path is not None,
                        "rows": pq.ParquetFile(path).metadata.num_rows if path else None,
                        "path": str(path) if path else None,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-dir", type=Path, required=True)
    parser.add_argument("--rd-dir", type=Path, required=True)
    parser.add_argument("--ercp-aih", type=Path, required=True)
    parser.add_argument("--supplemented-raw", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    partition_inventory = pd.concat(
        [inventory(args.sp_dir, "SP"), inventory(args.rd_dir, "RD")],
        ignore_index=True,
    )
    expected_per_dataset = len(UFS) * len(list(YEARS)) * len(list(MONTHS))
    present_counts = partition_inventory.groupby("dataset")["present"].sum().to_dict()
    missing = partition_inventory[~partition_inventory["present"]][
        ["dataset", "uf", "year", "month"]
    ].to_dict("records")

    ercp = pq.read_table(
        args.ercp_aih,
        columns=["competence_month", "SP_CNES", "SP_NAIH"],
    ).to_pandas()
    ercp["competence_month"] = ercp["competence_month"].astype(str)
    national_month = (
        ercp.groupby("competence_month").size().rename("unique_aih").reset_index()
    )
    full_months = pd.DataFrame(
        {"competence_month": [f"{year}{month:02d}" for year in YEARS for month in MONTHS]}
    )
    national_month = full_months.merge(national_month, on="competence_month", how="left")
    national_month["unique_aih"] = national_month["unique_aih"].fillna(0).astype(int)
    last_six = national_month.tail(6).copy()
    late_2025_nonzero = bool(last_six["unique_aih"].gt(0).all())
    median_2025 = float(
        national_month[national_month["competence_month"].str.startswith("2025")][
            "unique_aih"
        ].median()
    )
    december_ratio = (
        float(last_six.iloc[-1]["unique_aih"] / median_2025) if median_2025 else 0.0
    )

    raw_files = sorted(args.supplemented_raw.rglob("*.dbc"))
    raw_manifest = [
        {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in raw_files
    ]
    checks = {
        "sp_partitions_1620_of_1620": present_counts.get("SP", 0) == expected_per_dataset,
        "rd_partitions_1620_of_1620": present_counts.get("RD", 0) == expected_per_dataset,
        "no_missing_partitions": len(missing) == 0,
        "last_six_2025_months_nonzero": late_2025_nonzero,
        "december_2025_not_below_half_annual_median": december_ratio >= 0.5,
        "supplemented_raw_files_hashed": len(raw_manifest) == 6,
    }
    completeness_pass = all(checks.values())
    audit = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "PASS_WITH_MANDATORY_DOWNGRADE" if completeness_pass else "FIX",
        "partition_checks": checks,
        "present_partitions": {key: int(value) for key, value in present_counts.items()},
        "expected_partitions_per_dataset": expected_per_dataset,
        "missing_partitions": missing,
        "national_unique_aih_by_month": national_month.to_dict("records"),
        "december_2025_to_2025_median_ratio": december_ratio,
        "supplemented_raw_manifest": raw_manifest,
        "window_validity": {
            "status": "DOWNGRADE",
            "reason": (
                "The dedicated procedure code is not a homogeneous measure before 2021. "
                "First observed use in 2021-2025 cannot establish the true date of local "
                "technology adoption, especially at the January 2021 boundary."
            ),
            "required_language": (
                "Use 'first observed coded use' or 'observed uptake'; label January 2021 "
                "performers as prevalent-at-window-start; do not claim incident adoption "
                "or a causal effect of national incorporation."
            ),
        },
        "invalidates_prior_outputs": {
            "stages": [3, 4, 5, 6],
            "reason": (
                "Six SP state-month partitions were absent from the original analytic input, "
                "including five São Paulo partitions in 2025 and Santa Catarina 2025-12."
            ),
        },
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    partition_inventory.to_csv(args.inventory, index=False)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if completeness_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
