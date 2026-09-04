from __future__ import annotations

"""Stage 3 freeze: verify uniqueness/conservation on the four analytic
tables, compute SHA-256, and emit a freeze manifest."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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
    parser.add_argument("--hospital-month", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--municipality", type=Path, required=True)
    parser.add_argument("--patient-flow", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    ercp = pq.read_table(args.ercp_aih).to_pandas()
    hm = pq.read_table(args.hospital_month).to_pandas()
    elig = pq.read_table(args.eligible).to_pandas()
    mun = pq.read_table(args.municipality).to_pandas()
    flow = pq.read_table(args.patient_flow).to_pandas()

    checks = {}
    checks["ercp_aih_unique_key"] = int(ercp.duplicated(["competence_month", "SP_CNES", "SP_NAIH"]).sum()) == 0
    checks["ercp_aih_nonneg_counts"] = bool((ercp["matching_detail_row_count"] >= 1).all())
    checks["hospital_month_unique_key"] = int(hm.duplicated(["SP_CNES", "competence_month"]).sum()) == 0
    checks["hospital_month_counts_nonneg"] = bool((hm["ercp_count"] >= 0).all())
    checks["eligible_unique_key"] = int(elig.duplicated(["CNES", "competence_month"]).sum()) == 0
    checks["municipality_unique_key"] = int(mun.duplicated(["cohort", "res_municipio", "competence_month"]).sum()) == 0
    checks["flow_unique_key"] = int(flow.duplicated(["cohort", "res_municipio", "SP_CNES", "year"]).sum()) == 0
    checks["flow_conservation_A"] = int(flow[flow["cohort"] == "A"]["n_aih"].sum()) == int(ercp.shape[0])
    checks["flow_conservation_B"] = int(flow[flow["cohort"] == "B"]["n_aih"].sum()) == int(
        ercp[ercp["principal"].isin(["K803", "K804", "K805"]) & ercp["adult"] & ~ercp["malig_principal"]].shape[0]
    ) if "principal" in ercp.columns and "adult" in ercp.columns else None
    checks["municipality_conservation_A"] = int(mun[mun["cohort"] == "A"]["ercp_count"].sum()) == int(ercp.shape[0])
    checks["municipality_conservation_B"] = int(mun[mun["cohort"] == "B"]["ercp_count"].sum()) == int(
        ercp[ercp["principal"].isin(["K803", "K804", "K805"]) & ercp["adult"] & ~ercp["malig_principal"]].shape[0]
    ) if "principal" in ercp.columns and "adult" in ercp.columns else None

    files = {
        "ercp_aih": args.ercp_aih,
        "hospital_month": args.hospital_month,
        "eligible_hospital_month": args.eligible,
        "municipality_month": args.municipality,
        "patient_flow_year": args.patient_flow,
    }
    hashes = {name: sha256_stream(path) for name, path in files.items()}
    all_ok = all(v is not False for v in checks.values())
    manifest = {
        "schema_version": "1.0",
        "frozen_at": utc_now(),
        "status": "FROZEN" if all_ok else "NOT_FROZEN",
        "hashes": hashes,
        "checks": checks,
        "tables": {name: str(path) for name, path in files.items()},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps({"status": "PASS" if all_ok else "FAIL", "checks": checks},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())