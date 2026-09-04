#!/usr/bin/env python3
"""Fail-closed validation for the public DATASUS ERCP GitHub package."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_data"


def public_paths():
    """Yield repository content while excluding local Git and test caches."""
    excluded_parts = {".git", "__pycache__", ".pytest_cache"}
    return (
        path
        for path in ROOT.rglob("*")
        if not excluded_parts.intersection(path.relative_to(ROOT).parts)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def main() -> int:
    errors: list[str] = []
    required = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "LICENSE",
        ROOT / "LICENSE_SCOPE.md",
        SOURCE / "source_data_manifest.json",
        SOURCE / "source_data_audit.json",
        ROOT / "docs" / "REPRODUCIBILITY.md",
        ROOT / "config" / "threading.env",
        ROOT / "manuscript" / "manuscript_for_editor_english_v3.docx",
        ROOT / "manuscript" / "supplementary_material_for_editor_clean_v2.docx",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing required file: {path.relative_to(ROOT)}", errors)

    manifest = json.loads((SOURCE / "source_data_manifest.json").read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    if len(outputs) != 22:
        fail(f"expected 22 frozen CSV outputs, observed {len(outputs)}", errors)
    for name, record in outputs.items():
        path = SOURCE / name
        if not path.is_file():
            fail(f"missing frozen source data: {name}", errors)
            continue
        expected = record.get("sha256")
        observed = sha256(path)
        if expected != observed:
            fail(f"source-data hash mismatch: {name}", errors)

    audit = json.loads((SOURCE / "source_data_audit.json").read_text(encoding="utf-8"))
    for name, status in audit.get("checks", {}).items():
        if status != "PASS":
            fail(f"source-data audit is not PASS: {name}={status}", errors)

    threading = (ROOT / "config" / "threading.env").read_text(encoding="utf-8")
    if "ERCP_MAX_WORKERS=8" not in threading:
        fail("eight-thread ceiling missing", errors)

    forbidden_names = {"data_raw", "data_stage", "data_analytic", "restricted", "private"}
    for path in public_paths():
        if path.is_dir() and path.name.lower() in forbidden_names:
            fail(f"forbidden directory present: {path.relative_to(ROOT)}", errors)

    text_suffixes = {".py", ".r", ".md", ".yaml", ".yml", ".json", ".env", ".txt", ".csv", ".cff"}
    secret_pattern = re.compile(
        r"(?i)(ssh_pass\s*=|password\s*=\s*[^<\s][^\r\n]*|api[_-]?key\s*=\s*[^<\s][^\r\n]*|secret\s*=\s*[^<\s][^\r\n]*|/home/data/t[0-9]+/)"
    )
    for path in public_paths():
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if secret_pattern.search(text):
            fail(f"credential or internal server-path pattern: {path.relative_to(ROOT)}", errors)

    if errors:
        print("RELEASE_VALIDATION_FAIL")
        for item in errors:
            print(f"- {item}")
        return 1
    print(f"RELEASE_VALIDATION_PASS files={sum(1 for p in public_paths() if p.is_file())} source_csv=22")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
