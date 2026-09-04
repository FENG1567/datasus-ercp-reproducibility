#!/usr/bin/env python3
"""Generate a deterministic file inventory and SHA-256 checksum list."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "public_release_manifest_v1.json"
CHECKSUMS = ROOT / "manifests" / "checksums.sha256"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def listed_files(exclude: set[Path]) -> list[Path]:
    def is_transient(path: Path) -> bool:
        relative = path.relative_to(ROOT)
        return (
            ".git" in relative.parts
            or "__pycache__" in relative.parts
            or ".pytest_cache" in relative.parts
            or path.suffix.lower() in {".pyc", ".pyo"}
        )

    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and path not in exclude and not is_transient(path)
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def main() -> int:
    files = listed_files({MANIFEST, CHECKSUMS})
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    ]
    payload = {
        "schema_version": "datasus_ercp_public_release_manifest_v1",
        "generated_date": date.today().isoformat(),
        "release_status": "LOCAL_GITHUB_UPLOAD_CANDIDATE",
        "licence_status": "SELECTION_REQUIRED",
        "repository_url_status": "NOT_ASSIGNED",
        "doi_status": "NOT_ASSIGNED",
        "scientific_release": "frozen analysis outputs and final editor package",
        "file_count_excluding_manifest_and_checksum_list": len(records),
        "total_bytes_excluding_manifest_and_checksum_list": sum(r["bytes"] for r in records),
        "files": records,
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    all_files = listed_files({CHECKSUMS})
    CHECKSUMS.write_text(
        "".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in all_files),
        encoding="utf-8",
        newline="\n",
    )
    print(f"MANIFEST_PASS files={len(all_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
