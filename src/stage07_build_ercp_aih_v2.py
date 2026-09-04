from __future__ import annotations

"""Build one ERCP AIH per key while retaining its source state-month partition."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ERCP_CODE = "0407030255"
PARTITION_PATTERN = re.compile(r"([A-Z]{2})_(20\d{2})(\d{2})\.parquet$", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-audit", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    partition_records: list[dict] = []
    for path in sorted(args.sp_dir.rglob("*.parquet")):
        match = PARTITION_PATTERN.fullmatch(path.name)
        if not match:
            continue
        source_uf = match.group(1).upper()
        source_month = f"{match.group(2)}{match.group(3)}"
        table = pq.read_table(path).to_pandas()
        hit = table[table["SP_PROCREA_NORM"].astype(str).eq(ERCP_CODE)].copy()
        if not hit.empty:
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
            grouped["source_uf"] = source_uf
            grouped["source_partition"] = path.name
            frames.append(grouped)
        partition_records.append(
            {
                "source_uf": source_uf,
                "competence_month": source_month,
                "source_partition": path.name,
                "sp_detail_rows": int(len(table)),
                "matching_detail_rows": int(len(hit)),
                "unique_aih": int(
                    hit[["competence_month", "SP_CNES", "SP_NAIH"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    key = ["competence_month", "SP_CNES", "SP_NAIH"]
    duplicate_keys = int(result.duplicated(key, keep=False).sum()) if len(result) else 0
    source_month_mismatch = int(
        (result["competence_month"].astype(str) != result["source_partition"].str[3:9]).sum()
    ) if len(result) else 0
    result = result.sort_values(key).reset_index(drop=True)
    status = "PASS" if duplicate_keys == 0 and source_month_mismatch == 0 else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.partition_audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    partition_frame = pd.DataFrame(partition_records).sort_values(
        ["competence_month", "source_uf"]
    )
    partition_frame.to_parquet(args.partition_audit, index=False)
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "partitions_observed": int(len(partition_frame)),
        "partitions_with_code": int(partition_frame["unique_aih"].gt(0).sum()),
        "matching_detail_rows": int(partition_frame["matching_detail_rows"].sum()),
        "rows_total_unique_aih": int(len(result)),
        "duplicate_unique_aih_key_rows": duplicate_keys,
        "source_month_mismatch": source_month_mismatch,
        "counting_rule": (
            "One row per unique (competence_month, SP_CNES, SP_NAIH); SP detail "
            "rows and SP_QTD_ATO are audit-only and never treated as cases."
        ),
        "source_partition_retained": True,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
