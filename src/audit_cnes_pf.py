from __future__ import annotations

"""Per-file acceptance audit for CNES-PF: every manifest-complete DBC must
exist with matching size and SHA-256; filenames must be unique; the set of
(year, month, UF) must be exactly 2021-01..2025-12 x 27 UF (1,620 files)."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

UF_LIST = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8-sig"))]
    complete = [r for r in rows if r.get("status") == "complete"]
    problems = []
    verified = 0
    for row in complete:
        p = args.raw_root / str(row["year"]) / row["uf"] / row["filename"]
        if not p.exists():
            problems.append({"filename": row["filename"], "problem": "missing_file"})
            continue
        if p.stat().st_size != int(row["local_size_bytes"]):
            problems.append({"filename": row["filename"], "problem": "size_mismatch",
                             "expected": int(row["local_size_bytes"]), "actual": p.stat().st_size})
            continue
        if sha256(p).lower() != row["sha256"].lower():
            problems.append({"filename": row["filename"], "problem": "hash_mismatch"})
            continue
        verified += 1

    names = [r["filename"] for r in complete]
    dup_names = [name for name, count in Counter(names).items() if count > 1]
    covered = {(r["year"], str(r["month"]).zfill(2), r["uf"]) for r in complete}
    expected = {(str(y), f"{m:02d}", uf) for y in range(2021, 2026) for m in range(1, 13) for uf in UF_LIST}
    missing = sorted(expected - covered)
    extra = sorted(covered - expected)

    passed = (
        len(problems) == 0
        and not dup_names
        and not missing
        and not extra
        and len(complete) == 1620
    )
    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS" if passed else "FAIL",
        "manifest_rows": len(rows),
        "complete_rows": len(complete),
        "files_verified_ok": verified,
        "duplicate_filenames": dup_names,
        "missing_combinations": missing[:10],
        "extra_combinations": extra[:10],
        "problems": problems[:20],
        "problem_count": len(problems),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())