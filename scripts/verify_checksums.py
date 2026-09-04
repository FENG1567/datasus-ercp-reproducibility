#!/usr/bin/env python3
"""Verify every file listed in manifests/checksums.sha256."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = ROOT / "manifests" / "checksums.sha256"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    lines = CHECKSUMS.read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = ROOT / Path(relative)
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif digest(path) != expected:
            errors.append(f"hash mismatch: {relative}")
    if errors:
        print("CHECKSUMS_FAIL")
        for item in errors:
            print(f"- {item}")
        return 1
    print(f"CHECKSUMS_PASS files={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
