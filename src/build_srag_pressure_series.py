from __future__ import annotations

"""Build a state x month SUS pressure series from SIVEP-Gripe SRAG parquet
files (2021-2025). Primary metric: SRAG hospitalisations (notified cases)
per state and month; hospitalisation proxies for acute respiratory pressure
on the SUS hospital network."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

YEAR_COLUMN = "SG_UF_NOT"
NAME_PATTERN = re.compile(r"INFLUD(\d{2})-(\d{2})-(\d{2})-(\d{4})\.parquet")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    per_file = {}
    for path in sorted(args.input_dir.glob("INFLUD*.parquet")):
        df = pd.read_parquet(path, columns=None)
        match = NAME_PATTERN.match(path.name)
        if not match:
            raise RuntimeError(f"unexpected filename: {path.name}")
        year = 2000 + int(match.group(1))
        frame = df.copy()
        month_col = None
        for candidate in ["DT_SIN_PRI", "DT_NOTIFIC", "DT_INTERNA"]:
            if candidate in frame.columns:
                month_col = candidate
                break
        if month_col is None:
            raise RuntimeError(f"no date column in {path.name}")
        frame["ano"] = int(year)
        frame["mes"] = pd.to_datetime(frame[month_col], errors="coerce").dt.month
        per_file[path.name] = {
            "rows": int(len(frame)),
            "date_column": month_col,
            "with_valid_month": int(frame["mes"].notna().sum()),
            "with_uf": int(frame[YEAR_COLUMN].notna().sum()),
        }
        frame = frame.groupby([YEAR_COLUMN, "ano", "mes"], dropna=False).size().rename("srag_cases").reset_index()
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={YEAR_COLUMN: "uf"})
    combined = combined.sort_values(["uf", "ano", "mes"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "source": "SIVEP-Gripe SRAG official parquet (dadosabertos.saude.gov.br)",
        "per_file": per_file,
        "rows_total_state_month": int(len(combined)),
        "ufs": sorted(combined["uf"].dropna().unique().tolist()),
        "years": sorted(combined["ano"].dropna().unique().tolist()),
        "output": str(args.output),
        "metric": "SRAG notified hospitalisation cases (all classifications) by state and month",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())