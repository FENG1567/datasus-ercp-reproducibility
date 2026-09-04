#!/usr/bin/env python3
"""Create or verify a column-level dictionary for public source-data CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_data"
OUTPUT = ROOT / "data" / "source_data_dictionary.csv"


def rows() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in sorted(SOURCE.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
        for name in header:
            result.append(
                {
                    "file": path.name,
                    "column": name,
                    "definition": "See the frozen source-data manifest, SAP, and figure/table legend; no additional meaning inferred.",
                    "unit_or_allowed_values": "Declared in the file where explicit; otherwise consult the SAP.",
                    "missing_value": "Blank/NA where applicable; privacy-suppressed display cells may use <5.",
                }
            )
    return result


def render(items: list[dict[str, str]]) -> str:
    from io import StringIO

    out = StringIO()
    writer = csv.DictWriter(out, fieldnames=list(items[0]))
    writer.writeheader()
    writer.writerows(items)
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render(rows())
    if args.check:
        observed = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if observed.replace("\r\n", "\n") != text.replace("\r\n", "\n"):
            print("DATA_DICTIONARY_MISMATCH")
            return 1
        print("DATA_DICTIONARY_PASS")
        return 0
    OUTPUT.write_text(text, encoding="utf-8", newline="")
    print(f"WROTE {OUTPUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
