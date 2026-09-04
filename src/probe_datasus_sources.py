from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from ftplib import FTP
from pathlib import Path


HOST = "ftp.datasus.gov.br"
SOURCES = {
    "CNES_PF": "/dissemin/publicos/CNES/200508_/Dados/PF",
    "SIASUS_PA": "/dissemin/publicos/SIASUS/200801_/Dados",
}
PATTERNS = {
    "CNES_PF": re.compile(r"^PF([A-Z]{2})(\d{2})(\d{2})\.dbc$", re.I),
    "SIASUS_PA": re.compile(r"^PA([A-Z]{2})(\d{2})(\d{2})[a-z]?\.dbc$", re.I),
}
VALID_UF = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
}


def list_mlsd(path: str, attempts: int = 3) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with FTP() as ftp:
                ftp.connect(HOST, 21, timeout=60)
                ftp.login()
                ftp.set_pasv(True)
                ftp.cwd(path)
                rows = []
                for name, facts in ftp.mlsd():
                    if facts.get("type") != "file":
                        continue
                    rows.append(
                        {
                            "name": name,
                            "size": int(facts["size"]) if facts.get("size") else None,
                            "modified": facts.get("modify"),
                        }
                    )
                return rows
        except Exception as exc:  # provenance retains class, not server text
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"FTP listing failed after {attempts} attempts: {type(last_error).__name__}")


def normalize_year(two_digit: str) -> int:
    value = int(two_digit)
    return 2000 + value if value <= 99 else value


def summarize(dataset: str, rows: list[dict]) -> dict:
    pattern = PATTERNS[dataset]
    parsed = []
    for row in rows:
        match = pattern.match(row["name"])
        if not match:
            continue
        uf, yy, mm = match.groups()
        year, month = normalize_year(yy), int(mm)
        if uf.upper() not in VALID_UF or not 1 <= month <= 12:
            continue
        parsed.append({**row, "uf": uf.upper(), "year": year, "month": month})

    windows = {}
    for label, start, end in [("primary_2021_2025", 2021, 2025), ("history_2016_2025", 2016, 2025)]:
        chosen = [row for row in parsed if start <= row["year"] <= end]
        observed = {(row["uf"], row["year"], row["month"]) for row in chosen}
        expected = {(uf, year, month) for uf in VALID_UF for year in range(start, end + 1) for month in range(1, 13)}
        windows[label] = {
            "expected_partitions": len(expected),
            "observed_partitions": len(observed),
            "missing_partitions": [f"{uf}-{year:04d}-{month:02d}" for uf, year, month in sorted(expected - observed)],
            "duplicate_partition_count": len(chosen) - len(observed),
            "total_bytes": sum(row["size"] or 0 for row in chosen),
        }
    return {
        "dataset": dataset,
        "host": HOST,
        "remote_path": SOURCES[dataset],
        "matching_files_all_years": len(parsed),
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("provenance/datasus_source_probe.json"))
    args = parser.parse_args()
    result = {
        "schema_version": "1.0",
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "retrieval": "anonymous FTP MLSD, three bounded attempts per directory",
        "sources": [],
    }
    failed = False
    for dataset, path in SOURCES.items():
        try:
            result["sources"].append(summarize(dataset, list_mlsd(path)))
        except Exception as exc:
            failed = True
            result["sources"].append(
                {
                    "dataset": dataset,
                    "host": HOST,
                    "remote_path": path,
                    "status": "unavailable",
                    "error_class": type(exc).__name__,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

