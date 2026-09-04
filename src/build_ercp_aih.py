from __future__ import annotations

"""Build the ercp_aih base table: filter SP partitions to procedure
0407030255, collapse to one row per unique AIH
(competence_month + SP_CNES + SP_NAIH), keep procedure presence plus a
detail summary (never summing acts into cases)."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ERCP_CODE = "0407030255"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    audit_rows = []
    for year_dir in sorted(args.sp_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.parquet")):
            df = pq.read_table(path).to_pandas()
            hit = df[df["SP_PROCREA_NORM"] == ERCP_CODE]
            if len(hit) == 0:
                continue
            before = len(hit)
            hit = hit.copy()
            hit["SP_QTD_ATO"] = pd.to_numeric(hit["SP_QTD_ATO"], errors="coerce").fillna(0)
            hit["SP_VALATO"] = pd.to_numeric(hit["SP_VALATO"], errors="coerce").fillna(0)
            grouped = (
                hit.groupby(["competence_month", "SP_CNES", "SP_NAIH"], as_index=False)
                .agg(
                    matching_detail_row_count=("SP_PROCREA", "size"),
                    sum_SP_QTD_ATO_for_audit_only=("SP_QTD_ATO", "sum"),
                    distinct_SP_ATOPROF_count=("SP_ATOPROF", "nunique"),
                    distinct_SP_PF_CBO_count=("SP_PF_CBO", "nunique"),
                    min_SP_DTINTER=("SP_DTINTER", "min"),
                    max_SP_DTSAIDA=("SP_DTSAIDA", "max"),
                    first_SP_CIDPRI=("SP_CIDPRI", "first"),
                    first_SP_CIDSEC=("SP_CIDSEC", "first"),
                )
            )
            grouped["procedure_presence"] = 1
            frames.append(grouped)
            audit_rows.append(
                {
                    "file": path.name,
                    "sp_rows_with_code": before,
                    "unique_aih": len(grouped),
                }
            )
            print(f"PASS {path.name}: {before} rows -> {len(grouped)} unique AIH", flush=True)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=[
                "competence_month", "SP_CNES", "SP_NAIH", "matching_detail_row_count",
                "sum_SP_QTD_ATO_for_audit_only", "distinct_SP_ATOPROF_count",
                "distinct_SP_PF_CBO_count", "min_SP_DTINTER", "max_SP_DTSAIDA",
                "first_SP_CIDPRI", "first_SP_CIDSEC", "procedure_presence",
            ]
        )

    dup = int(combined.duplicated(["competence_month", "SP_CNES", "SP_NAIH"]).sum())
    combined = combined.sort_values(["competence_month", "SP_CNES", "SP_NAIH"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    output_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS" if dup == 0 else "FAIL",
        "rows_total": int(len(combined)),
        "duplicate_unique_aih_keys": dup,
        "per_partition": audit_rows,
        "output": str(args.output),
        "output_sha256": output_hash,
        "counting_rule": "one row per unique (competence_month, SP_CNES, SP_NAIH); SP_QTD_ATO and detail rows are audit-only summaries",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0 if dup == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())