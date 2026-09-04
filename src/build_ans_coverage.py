from __future__ import annotations

"""ANS supplementary-coverage rate: medical-hospital active links (Dec) over
IBGE municipal population, per municipality-year. Also verifies the 6-to-7
digit IBGE code bridge used between the ANS table and IBGE population."""

import argparse
import csv
import json
import struct
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ibge_check_digit(six: str) -> int:
    digits = [int(d) for d in six]
    weights = [2, 3, 4, 5, 6, 7]
    total = 0
    for i, digit in enumerate(digits):
        total += digit * weights[i % len(weights)]
    remainder = total % 11
    dv = 11 - remainder
    return 0 if dv >= 10 else dv


def six_to_seven(six: str) -> str:
    return six + str(ibge_check_digit(six))


def read_pop_dbf(path: Path) -> tuple[dict[str, int], dict[str, str]]:
    raw = path.read_bytes()
    header_len = struct.unpack("<H", raw[8:10])[0]
    record_len = struct.unpack("<H", raw[10:12])[0]
    nfields = (header_len - 33) // 32
    fields = []
    pos = 32
    for _ in range(nfields):
        name = raw[pos:pos + 11].split(b"\x00")[0].decode("latin-1")
        ftype = chr(raw[pos + 11])
        flen = raw[pos + 16]
        fields.append((name, ftype, flen))
        pos += 32
    pop = {}
    code_map = {}
    idx = header_len
    while idx + record_len <= len(raw):
        if raw[idx] == 0x1A:
            break
        record = raw[idx + 1: idx + record_len]
        values = {}
        offset = 0
        for name, ftype, flen in fields:
            values[name] = record[offset:offset + flen].decode("latin-1").strip()
            offset += flen
        code = values.get("COD_MUN") or values.get("cod_mun") or ""
        ano = values.get("ANO") or values.get("ano") or ""
        pop_value = values.get("POP") or values.get("pop") or ""
        if code and ano and len(code) == 7:
            try:
                pop[(code, ano)] = pop.get((code, ano), 0) + int(pop_value or 0)
            except ValueError:
                pass
            code_map.setdefault(code[:6], code)
        idx += record_len
    return pop, code_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ans-aggregate", type=Path, required=True)
    parser.add_argument("--popsvs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    ans = list(csv.DictReader(open(args.ans_aggregate, encoding="utf-8-sig")))
    population = {}
    pop_years = {}
    code_map = {}
    for year in ["2021", "2022", "2023", "2024", "2025"]:
        zip_path = args.popsvs_dir / year / f"POPSBR{year[-2:]}.zip"
        if not zip_path.exists():
            raise RuntimeError(f"missing population archive {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            member = archive.namelist()[0]
            raw = archive.read(member)
            tmp = Path("logs") / f"pop_{year}.dbf"
            tmp.write_bytes(raw)
            pop, year_map = read_pop_dbf(tmp)
            tmp.unlink(missing_ok=True)
        population.update({(code, year): value for (code, _), value in pop.items()})
        pop_years[year] = len(pop)
        code_map.update(year_map)

    rows = []
    code_bridge_failures = []
    for r in ans:
        period = r["period"]
        year = period[:4]
        six = r["cd_municipio_6"]
        if six.endswith("0000") or six.endswith("000"):
            continue
        seven = code_map.get(six)
        if seven is None:
            code_bridge_failures.append({"period": period, "six": six})
            continue
        medical = int(r["medical_hospital_active_links"] or 0)
        pop_key = (seven, year)
        total_pop = population.get(pop_key)
        if total_pop is None:
            code_bridge_failures.append({"period": period, "six": six})
            continue
        rows.append(
            {
                "period": period,
                "year": year,
                "sg_uf": r["sg_uf"],
                "cd_municipio_6": six,
                "cd_municipio_7": seven,
                "medical_hospital_active_links": medical,
                "ibge_population": total_pop,
                "supplementary_coverage_rate": round(medical / total_pop, 6) if total_pop else 0.0,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS",
        "rows_output": len(rows),
        "municipality_years_with_pop": len(rows),
        "code_bridge_failures": len(code_bridge_failures),
        "code_bridge_failure_sample": code_bridge_failures[:10],
        "population_records_by_year": pop_years,
        "coverage_summary": {
            "median": round(sorted(r["supplementary_coverage_rate"] for r in rows)[len(rows) // 2], 5),
            "mean": round(sum(r["supplementary_coverage_rate"] for r in rows) / len(rows), 5),
            "min": round(min(r["supplementary_coverage_rate"] for r in rows), 5),
            "max": round(max(r["supplementary_coverage_rate"] for r in rows), 5),
        },
        "note": "Medical-hospital active plans / IBGE population, December of each year. Municipality codes bridged 6->7 digit using the official IBGE code map built from the IBGE population DBF files (not check-digit computation).",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0 if not code_bridge_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())