from __future__ import annotations

"""State-month streaming DBC -> Parquet conversion for SIH-RD and SIH-SP
2021-2025, with: column pruning, resume (skip existing partitions), and
process-level parallelism capped at 8 workers."""

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
ERCP_CODE = "0407030255"

RD_COLUMNS = [
    "ANO_CMPT", "MES_CMPT", "CNES", "N_AIH", "IDENT", "MUNIC_RES", "MUNIC_MOV",
    "DT_INTER", "DT_SAIDA", "DIAG_PRINC", "DIAG_SECUN",
    "DIAGSEC1", "DIAGSEC2", "DIAGSEC3", "DIAGSEC4", "DIAGSEC5",
    "DIAGSEC6", "DIAGSEC7", "DIAGSEC8", "DIAGSEC9",
    "TPDISEC1", "TPDISEC2", "TPDISEC3", "TPDISEC4", "TPDISEC5",
    "TPDISEC6", "TPDISEC7", "TPDISEC8", "TPDISEC9",
    "RACA_COR", "INSTRU", "ETNIA", "SEXO", "IDADE", "COD_IDADE", "MORTE",
    "UTI_MES_IN", "UTI_MES_AN", "UTI_MES_AL", "UTI_MES_TO",
    "MARCA_UTI", "NATUREZA", "CAR_INT", "COMPLEX", "FINANC",
    "VAL_TOT", "CID_MORTE", "CID_ASSO", "SEQ_AIH5", "DIAS_PERM",
]
SP_COLUMNS = [
    "SP_AA", "SP_MM", "SP_UF", "SP_CNES", "SP_NAIH", "SP_PROCREA",
    "SP_CIDPRI", "SP_CIDSEC", "SP_DTINTER", "SP_DTSAIDA",
    "SP_QTD_ATO", "SP_ATOPROF", "SP_TP_ATO", "SP_PF_CBO", "SP_VALATO",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_procedure(code: str) -> str:
    digits = "".join(ch for ch in (code or "") if ch.isdigit())
    return digits.zfill(10)


def read_dbc_records(dbc_path: Path, columns: list[str]) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="dbc_") as tmp:
        tmp_dbc = Path(tmp) / dbc_path.name
        shutil.copyfile(dbc_path, tmp_dbc)
        dbf_path = tmp_dbc.with_suffix(".dbf")
        dbc2dbf(str(tmp_dbc), str(dbf_path))
        table = DBF(str(dbf_path), encoding="latin-1", load=False)
        rows = []
        for record in table:
            rows.append({k: record[k] for k in columns if k in record})
        return rows


def build_partition(args_tuple: tuple) -> dict:
    dataset, year, month, uf, dbc_path, output_dir = args_tuple
    columns = RD_COLUMNS if dataset == "RD" else SP_COLUMNS
    rows = read_dbc_records(dbc_path, columns)
    if not rows:
        raise RuntimeError(f"empty conversion for {dbc_path.name}")
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    competence = f"{year}{month:02d}"
    if dataset == "RD":
        df["competence_month"] = df["ANO_CMPT"] + df["MES_CMPT"]
        df["CNES"] = df["CNES"].str.strip().str.zfill(7)
        df["N_AIH"] = df["N_AIH"].str.strip()
        code_hits = 0
    else:
        df["competence_month"] = df["SP_AA"] + df["SP_MM"]
        df["SP_CNES"] = df["SP_CNES"].str.strip().str.zfill(7)
        df["SP_NAIH"] = df["SP_NAIH"].str.strip()
        df["SP_PROCREA_NORM"] = df["SP_PROCREA"].map(normalize_procedure)
        code_hits = int((df["SP_PROCREA_NORM"] == ERCP_CODE).sum())

    mismatched = int((df["competence_month"] != competence).sum())
    if mismatched:
        raise RuntimeError(
            f"competence mismatch in {dbc_path.name}: {mismatched} rows disagree with filename"
        )

    if dataset == "RD":
        unique_key = int(df[["competence_month", "CNES", "N_AIH"]].drop_duplicates().shape[0])
    else:
        unique_key = int(df[["competence_month", "SP_CNES", "SP_NAIH"]].drop_duplicates().shape[0])
    dup_keys = len(df) - unique_key

    output_path = output_dir / dataset / str(year) / f"{uf}_{year}{month:02d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd")
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

    date_min = date_max = ""
    for col in ("DT_INTER", "DT_SAIDA"):
        if col in df.columns:
            valid = df[col].str.fullmatch(r"\d{8}")
            if valid.any():
                dates = df.loc[valid, col]
                date_min = min(date_min, dates.min()) if date_min else dates.min()
                date_max = max(date_max, dates.max()) if date_max else dates.max()

    return {
        "dataset": dataset,
        "year": year,
        "month": month,
        "uf": uf,
        "file": dbc_path.name,
        "input_rows": len(rows),
        "output_rows": int(len(df)),
        "columns": list(df.columns),
        "code_hits_0407030255": code_hits,
        "unique_aih": unique_key,
        "duplicate_key_rows": dup_keys,
        "date_min": date_min,
        "date_max": date_max,
        "output_path": str(output_path),
        "output_sha256": output_hash,
        "converted_at": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["RD", "SP"], required=True)
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
                prefix = f"{'RD' if args.dataset == 'RD' else 'SP'}{uf}{year % 100}{month:02d}"
                dbc_path = args.raw_root / str(year) / uf / f"{prefix}.dbc"
                output_path = args.output_dir / args.dataset / str(year) / f"{uf}_{year}{month:02d}.parquet"
                if output_path.exists():
                    continue
                if not dbc_path.exists():
                    continue
                tasks.append((args.dataset, year, month, uf, dbc_path, args.output_dir))

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
        "dataset": args.dataset,
        "partitions_ok": len(qc_records),
        "partitions_failed": len(failed),
        "failures": failed,
        "total_input_rows": int(qc_df["input_rows"].sum()) if len(qc_df) else 0,
        "total_output_rows": int(qc_df["output_rows"].sum()) if len(qc_df) else 0,
        "total_unique_aih": int(qc_df["unique_aih"].sum()) if len(qc_df) else 0,
        "total_code_hits": int(qc_df["code_hits_0407030255"].sum()) if len(qc_df) else 0,
    }
    (args.qc_log.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())