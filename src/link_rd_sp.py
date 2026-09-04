from __future__ import annotations

"""RD-SP linkage: join unique ERCP AIH (SP) to RD records by
(competence_month, CNES, N_AIH). Outputs four classes:
exact, ambiguous, unmatched, cross-month candidate."""

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
    parser.add_argument("--rd-dir", type=Path, required=True)
    parser.add_argument("--ercp-aih", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    sp = pq.read_table(args.ercp_aih).to_pandas()
    sp["rd_key"] = sp["competence_month"] + "_" + sp["SP_CNES"] + "_" + sp["SP_NAIH"]

    rd_frames = []
    for year_dir in sorted(args.rd_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.parquet")):
            rd_frames.append(pq.read_table(path).to_pandas())
    rd = pd.concat(rd_frames, ignore_index=True)
    rd["rd_key"] = rd["competence_month"] + "_" + rd["CNES"] + "_" + rd["N_AIH"]

    RD_FIELDS = [
        "DIAG_PRINC", "DIAG_SECUN", "DIAGSEC1", "DIAGSEC2", "DIAGSEC3", "DIAGSEC4",
        "DIAGSEC5", "DIAGSEC6", "DIAGSEC7", "DIAGSEC8", "DIAGSEC9",
        "TPDISEC1", "TPDISEC2", "TPDISEC3", "TPDISEC4", "TPDISEC5",
        "TPDISEC6", "TPDISEC7", "TPDISEC8", "TPDISEC9",
        "MORTE", "IDADE", "COD_IDADE", "SEXO", "RACA_COR", "INSTRU", "ETNIA",
        "MUNIC_RES", "MUNIC_MOV", "DT_INTER", "DT_SAIDA", "NATUREZA",
        "UTI_MES_IN", "UTI_MES_AN", "UTI_MES_AL", "UTI_MES_TO",
        "MARCA_UTI", "VAL_TOT", "CID_MORTE", "CID_ASSO",
    ]
    rd_dup = rd["rd_key"].duplicated(keep=False)
    rd_unique = rd[~rd_dup].set_index("rd_key")
    rd_dup_keys = rd.loc[rd_dup, "rd_key"].drop_duplicates()

    exact_mask = sp["rd_key"].isin(rd_unique.index)
    ambiguous_mask = sp["rd_key"].isin(rd_dup_keys)
    unmatched_mask = ~(exact_mask | ambiguous_mask)

    rd_merge_cols = [c for c in RD_FIELDS if c in rd.columns]
    exact = sp[exact_mask].merge(
        rd_unique[rd_merge_cols].reset_index(),
        on="rd_key", how="left",
    ).drop(columns=["rd_key"])
    ambiguous = sp[ambiguous_mask].copy()
    unmatched = sp[unmatched_mask].copy()

    cross_month = unmatched[
        unmatched["SP_CNES"].isin(rd["CNES"])
        & unmatched["SP_NAIH"].isin(rd["N_AIH"])
    ].copy()

    exact = exact.drop_duplicates(["competence_month", "SP_CNES", "SP_NAIH"])
    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "sp_unique_aih": int(len(sp)),
        "rd_rows_total": int(len(rd)),
        "rd_unique_keys": int(len(rd_unique)),
        "rd_duplicate_key_groups": int(len(rd_dup_keys)),
        "link_exact": int(len(exact)),
        "link_ambiguous": int(len(ambiguous)),
        "link_unmatched": int(len(unmatched)),
        "cross_month_candidates": int(len(cross_month)),
        "link_rate_exact": round(len(exact) / len(sp), 6) if len(sp) else None,
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([exact.assign(link_class="exact"),
                          ambiguous.assign(link_class="ambiguous"),
                          unmatched.assign(link_class="unmatched")], ignore_index=True)
    combined.to_parquet(args.output, index=False)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())