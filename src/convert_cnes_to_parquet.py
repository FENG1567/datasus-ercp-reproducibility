from __future__ import annotations

"""CNES module DBC -> Parquet conversion (ST/LT/EQ/SR/HB/EP/PF, 2021-2025)
with column pruning, resume, and 8-worker parallelism."""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dbfread import DBF
from pyreaddbc import dbc2dbf

UF_LIST = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]

COMMON_COLUMNS = ["CNES", "COMPETENCIA", "CODUFMUN"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_dbc_records(dbc_path: Path, columns: list[str] | None = None) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="dbc_") as tmp:
        tmp_dbc = Path(tmp) / dbc_path.name
        shutil.copyfile(dbc_path, tmp_dbc)
        dbf_path = tmp_dbc.with_suffix(".dbf")
        dbc2dbf(str(tmp_dbc), str(dbf_path))
        table = DBF(str(dbf_path), encoding="latin-1", load=False)
        rows = []
        for record in table:
            if columns is None:
                rows.append(dict(record))
            else:
                rows.append({k: record[k] for k in columns if k in record})
        return rows


def build_partition(args_tuple: tuple) -> dict:
    module, year, month, uf, dbc_path, output_dir = args_tuple
    rows = read_dbc_records(dbc_path)
    if not rows:
        raise RuntimeError(f"empty conversion for {dbc_path.name}")
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    competence = f"{year}{month:02d}"
    df["competence_month"] = competence
    df["CNES"] = df["CNES"].str.strip().str.zfill(7)
    mismatched = 0
    for col in ("COMPETENCIA", "COMPET"):
        if col in df.columns:
            mismatched = max(mismatched, int((df[col].astype(str).str.strip() != competence).sum()))
    if mismatched:
        raise RuntimeError(f"competence mismatch in {dbc_path.name}: {mismatched} rows")

    output_path = output_dir / module / str(year) / f"{uf}_{year}{month:02d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd")
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

    return {
        "module": module,
        "year": year,
        "month": month,
        "uf": uf,
        "file": dbc_path.name,
        "input_rows": len(rows),
        "output_rows": int(len(df)),
        "columns": list(df.columns),
        "unique_cnes": int(df["CNES"].nunique()),
        "output_path": str(output_path),
        "output_sha256": output_hash,
        "converted_at": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["ST", "LT", "EQ", "SR", "HB", "EP", "PF"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qc-log", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            for uf in UF_LIST:
                prefix = f"{args.module}{uf}{year % 100}{month:02d}"
                dbc_path = args.raw_root / args.module / str(year) / uf / f"{prefix}.dbc"
                output_path = args.output_dir / args.module / str(year) / f"{uf}_{year}{month:02d}.parquet"
                if output_path.exists():
                    continue
                if not dbc_path.exists():
                    continue
                tasks.append((args.module, year, month, uf, dbc_path, args.output_dir))

    qc_records = []
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_partition, t): t[4].name for t in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                qc = future.result()
                qc_records.append(qc)
                print(json.dumps({"status": "PASS", "file": name}, ensure_ascii=False), flush=True)
            except Exception as exc:
                failed.append({"file": name, "reason": type(exc).__name__, "detail": str(exc)[:200]})
                print(json.dumps({"status": "FAIL", "file": name, "error": str(exc)[:200]},
                                 ensure_ascii=False), flush=True)

    qc_df = pd.DataFrame(qc_records)
    args.qc_log.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_parquet(args.qc_log, index=False)
    summary = {
        "module": args.module,
        "partitions_ok": len(qc_records),
        "partitions_failed": len(failed),
        "failures": failed,
        "total_input_rows": int(qc_df["input_rows"].sum()) if len(qc_df) else 0,
        "total_unique_cnes_rows": int(qc_df["unique_cnes"].sum()) if len(qc_df) else 0,
    }
    (args.qc_log.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())