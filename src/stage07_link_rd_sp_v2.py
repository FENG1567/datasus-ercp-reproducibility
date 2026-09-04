from __future__ import annotations

"""Partition-aware RD-SP linkage with state-month linkage gates."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


RD_FIELDS = [
    "DIAG_PRINC", "DIAG_SECUN", "DIAGSEC1", "DIAGSEC2", "DIAGSEC3", "DIAGSEC4",
    "DIAGSEC5", "DIAGSEC6", "DIAGSEC7", "DIAGSEC8", "DIAGSEC9",
    "TPDISEC1", "TPDISEC2", "TPDISEC3", "TPDISEC4", "TPDISEC5",
    "TPDISEC6", "TPDISEC7", "TPDISEC8", "TPDISEC9",
    "MORTE", "IDADE", "COD_IDADE", "SEXO", "RACA_COR", "INSTRU", "ETNIA",
    "MUNIC_RES", "MUNIC_MOV", "DT_INTER", "DT_SAIDA", "NATUREZA", "CAR_INT",
    "UTI_MES_IN", "UTI_MES_AN", "UTI_MES_AL", "UTI_MES_TO", "MARCA_UTI",
    "VAL_TOT", "CID_MORTE", "CID_ASSO", "DIAS_PERM", "TP_UNID",
]


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
    parser.add_argument("--rd-dir", type=Path, required=True)
    parser.add_argument("--ercp-aih", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-audit", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--overall-min", type=float, default=0.98)
    parser.add_argument("--partition-min", type=float, default=0.95)
    args = parser.parse_args()

    sp = pq.read_table(args.ercp_aih).to_pandas()
    key_sp = ["competence_month", "SP_CNES", "SP_NAIH"]
    key_rd = ["competence_month", "CNES", "N_AIH"]
    outputs: list[pd.DataFrame] = []
    partition_records: list[dict] = []
    missing_rd_partitions: list[str] = []

    for (source_uf, competence_month), group in sp.groupby(
        ["source_uf", "competence_month"], sort=True
    ):
        year = str(competence_month)[:4]
        rd_path = args.rd_dir / year / f"{source_uf}_{competence_month}.parquet"
        if not rd_path.exists():
            missing_rd_partitions.append(str(rd_path))
            unmatched = group.copy()
            unmatched["link_class"] = "unmatched_missing_rd_partition"
            outputs.append(unmatched)
            partition_records.append(
                {
                    "source_uf": source_uf,
                    "competence_month": competence_month,
                    "n_sp_unique_aih": len(group),
                    "n_exact": 0,
                    "n_ambiguous": 0,
                    "n_unmatched": len(group),
                    "exact_rate": 0.0,
                    "rd_partition_present": False,
                }
            )
            continue

        rd = pq.read_table(rd_path).to_pandas()
        rd["competence_month"] = rd["competence_month"].astype(str)
        rd["CNES"] = rd["CNES"].astype(str).str.zfill(7)
        rd["N_AIH"] = rd["N_AIH"].astype(str).str.strip()
        duplicate_rd = rd.duplicated(key_rd, keep=False)
        ambiguous_keys = set(map(tuple, rd.loc[duplicate_rd, key_rd].itertuples(index=False, name=None)))
        rd_unique = rd.loc[~duplicate_rd].copy()
        merge_columns = key_rd + [column for column in RD_FIELDS if column in rd.columns]
        exact = group.merge(
            rd_unique[merge_columns],
            left_on=key_sp,
            right_on=key_rd,
            how="inner",
            validate="one_to_one",
        ).drop(columns=["CNES", "N_AIH"])
        exact["link_class"] = "exact"

        group_keys = pd.MultiIndex.from_frame(group[key_sp])
        exact_keys = pd.MultiIndex.from_frame(exact[key_sp])
        remaining = group.loc[~group_keys.isin(exact_keys)].copy()
        remaining_tuple = list(map(tuple, remaining[key_sp].itertuples(index=False, name=None)))
        ambiguous_mask = [
            (row[0], row[1], row[2]) in ambiguous_keys for row in remaining_tuple
        ]
        ambiguous = remaining.loc[ambiguous_mask].copy()
        unmatched = remaining.loc[[not value for value in ambiguous_mask]].copy()
        ambiguous["link_class"] = "ambiguous"
        unmatched["link_class"] = "unmatched"
        outputs.extend([exact, ambiguous, unmatched])
        exact_rate = len(exact) / len(group) if len(group) else 1.0
        partition_records.append(
            {
                "source_uf": source_uf,
                "competence_month": competence_month,
                "n_sp_unique_aih": len(group),
                "n_exact": len(exact),
                "n_ambiguous": len(ambiguous),
                "n_unmatched": len(unmatched),
                "exact_rate": exact_rate,
                "rd_partition_present": True,
            }
        )

    linked = pd.concat(outputs, ignore_index=True, sort=False)
    linked = linked.sort_values(key_sp).reset_index(drop=True)
    partition_audit = pd.DataFrame(partition_records).sort_values(
        ["competence_month", "source_uf"]
    )
    overall_exact = int(linked["link_class"].eq("exact").sum())
    overall_rate = overall_exact / len(sp) if len(sp) else 0.0
    below_gate = partition_audit[
        partition_audit["exact_rate"].lt(args.partition_min)
    ].copy()
    conservation = len(linked) == len(sp) and not linked.duplicated(key_sp).any()
    checks = {
        "no_missing_rd_partitions": not missing_rd_partitions,
        "linked_row_conservation": conservation,
        "overall_exact_rate_gate": overall_rate >= args.overall_min,
        "each_nonzero_state_month_exact_rate_gate": below_gate.empty,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.partition_audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    linked.to_parquet(args.output, index=False)
    partition_audit.to_parquet(args.partition_audit, index=False)
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "checks": checks,
        "sp_unique_aih": int(len(sp)),
        "linked_rows": int(len(linked)),
        "link_class_counts": {
            str(key): int(value)
            for key, value in linked["link_class"].value_counts(dropna=False).items()
        },
        "overall_exact_rate": overall_rate,
        "state_month_partitions_with_code": int(len(partition_audit)),
        "state_month_partitions_below_gate": below_gate.to_dict("records"),
        "missing_rd_partitions": missing_rd_partitions,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
