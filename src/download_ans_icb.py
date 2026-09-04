from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


BASE = "https://dadosabertos.ans.gov.br/FTP/PDA/informacoes_consolidadas_de_beneficiarios-024"
USER_AGENT = "DATASUS-ERCP-reproducible-research/1.0"
UF_LIST = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO", "XX",
]
FILE_RE = re.compile(r"^pda-024-icb-([A-Z]{2})-(\d{4})_(\d{2})\.zip$")
MANIFEST_FIELDS = [
    "period", "uf", "filename", "source_url", "accessed_at", "last_modified", "etag",
    "remote_size_bytes", "local_relative_path", "local_size_bytes", "sha256", "status", "error_class",
]
AGG_FIELDS = [
    "period", "sg_uf", "cd_municipio_6", "medical_hospital_active_links",
    "dental_active_links", "all_active_links", "source_row_count",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


def list_period(period: str, attempts: int = 3) -> list[dict]:
    url = f"{BASE}/{period}/"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                html = response.read().decode("utf-8", errors="replace")
            names = sorted(set(re.findall(r'href="([^"]+\.zip)"', html, flags=re.I)))
            selected = []
            for name in names:
                match = FILE_RE.match(name)
                if not match:
                    continue
                uf, year, month = match.groups()
                if f"{year}{month}" != period or uf not in UF_LIST:
                    continue
                selected.append({"period": period, "uf": uf, "filename": name, "source_url": url + name})
            observed = [row["uf"] for row in selected]
            if sorted(observed) != sorted(UF_LIST) or len(observed) != len(set(observed)):
                raise RuntimeError(f"Incomplete ANS listing for {period}: observed={observed}")
            return sorted(selected, key=lambda row: UF_LIST.index(row["uf"]))
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"ANS listing failed after {attempts} attempts: {type(last_error).__name__}") from last_error


def download(row: dict, destination: Path, expected_csv: str, attempts: int = 3) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(row["source_url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=900) as response, partial.open("wb") as output:
                headers = {
                    "last_modified": response.headers.get("Last-Modified", ""),
                    "etag": response.headers.get("ETag", ""),
                    "remote_size_bytes": response.headers.get("Content-Length", ""),
                }
                shutil.copyfileobj(response, output, length=1024 * 1024)
            validate_archive(partial, expected_csv)
            os.replace(partial, destination)
            return headers
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"ANS file download failed after {attempts} attempts: {type(last_error).__name__}") from last_error


def validate_archive(path: Path, expected_csv: str) -> None:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != 1 or Path(members[0].filename).name != expected_csv:
            raise RuntimeError(f"Unexpected ANS archive members: {[member.filename for member in members]}")
        member_path = Path(members[0].filename.replace("\\", "/"))
        if member_path.is_absolute() or ".." in member_path.parts or len(member_path.parts) != 1:
            raise RuntimeError(f"Unsafe ANS ZIP member: {members[0].filename}")


def aggregate_archive(path: Path, period: str, uf: str, aggregate: dict, categories: set[str]) -> None:
    expected_csv = path.name.removesuffix(".zip") + ".csv"
    validate_archive(path, expected_csv)
    with zipfile.ZipFile(path) as archive, archive.open(expected_csv) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text, delimiter=";")
        required = {
            "ID_CMPT_MOVEL", "SG_UF", "CD_MUNICIPIO", "COBERTURA_ASSIST_PLAN", "QT_BENEFICIARIO_ATIVO"
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"ANS schema mismatch in {path.name}: {reader.fieldnames}")
        for row in reader:
            if row["ID_CMPT_MOVEL"].replace("-", "") != period:
                raise RuntimeError(f"Period mismatch in {path.name}: {row['ID_CMPT_MOVEL']}")
            row_uf = row["SG_UF"].strip().upper()
            if uf != "XX" and row_uf != uf:
                raise RuntimeError(f"UF mismatch in {path.name}: {row_uf}")
            municipality = row["CD_MUNICIPIO"].strip()
            if uf == "XX" or not re.fullmatch(r"\d{6}", municipality):
                continue
            category = row["COBERTURA_ASSIST_PLAN"].strip()
            categories.add(category)
            active = int(row["QT_BENEFICIARIO_ATIVO"] or 0)
            key = (period, row_uf, municipality)
            target = aggregate[key]
            target["all_active_links"] += active
            target["source_row_count"] += 1
            if category == "Médico-hospitalar":
                target["medical_hospital_active_links"] += active
            elif category == "Odontológico":
                target["dental_active_links"] += active
            else:
                raise RuntimeError(f"Unrecognized COBERTURA_ASSIST_PLAN={category!r} in {path.name}")


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--month", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("provenance/ans_icb_manifest.csv"))
    parser.add_argument("--aggregate-output", type=Path, default=Path("data_stage/ans_icb_municipality_year.csv"))
    parser.add_argument("--event-log", type=Path, default=Path("provenance/ans_icb_events.jsonl"))
    parser.add_argument("--inter-file-delay", type=float, default=1.0)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year cannot exceed --end-year")
    if args.inter_file_delay < 0:
        parser.error("--inter-file-delay cannot be negative")

    manifest = load_manifest(args.manifest)
    aggregate = defaultdict(lambda: defaultdict(int))
    categories: set[str] = set()
    for year in range(args.start_year, args.end_year + 1):
        period = f"{year}{args.month:02d}"
        listing = list_period(period)
        append_event(args.event_log, {"status": "listing_pass", "period": period, "file_count": len(listing)})
        for row in listing:
            destination = args.raw_root / period / row["filename"]
            expected_csv = row["filename"].removesuffix(".zip") + ".csv"
            if destination.exists():
                try:
                    validate_archive(destination, expected_csv)
                    headers = manifest.get(row["filename"], {})
                except Exception as exc:
                    quarantine = (
                        args.raw_root.parent
                        / "icb_quarantine"
                        / period
                        / f"{row['filename']}.invalid_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, quarantine)
                    append_event(
                        args.event_log,
                        {
                            "status": "invalid_existing_quarantined",
                            "period": period,
                            "uf": row["uf"],
                            "filename": row["filename"],
                            "error_class": type(exc).__name__,
                            "quarantine": str(quarantine),
                        },
                    )
                    headers = download(row, destination, expected_csv)
            else:
                headers = download(row, destination, expected_csv)
            record = {
                **row,
                "accessed_at": headers.get("accessed_at", utc_now()),
                "last_modified": headers.get("last_modified", ""),
                "etag": headers.get("etag", ""),
                "remote_size_bytes": headers.get("remote_size_bytes", ""),
                "local_relative_path": str(destination.relative_to(args.raw_root)).replace("\\", "/"),
                "local_size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "status": "complete",
                "error_class": "",
            }
            manifest[row["filename"]] = record
            save_csv(
                args.manifest,
                MANIFEST_FIELDS,
                sorted(manifest.values(), key=lambda item: (item["period"], UF_LIST.index(item["uf"]))),
            )
            aggregate_archive(destination, period, row["uf"], aggregate, categories)
            append_event(
                args.event_log,
                {"status": "file_pass", "period": period, "uf": row["uf"], "filename": row["filename"]},
            )
            if args.inter_file_delay:
                time.sleep(args.inter_file_delay)

    aggregate_rows = []
    for (period, uf, municipality), values in sorted(aggregate.items()):
        aggregate_rows.append(
            {
                "period": period,
                "sg_uf": uf,
                "cd_municipio_6": municipality,
                **{field: values[field] for field in AGG_FIELDS[3:]},
            }
        )
    save_csv(args.aggregate_output, AGG_FIELDS, aggregate_rows)
    append_event(
        args.event_log,
        {
            "status": "complete",
            "period_count": args.end_year - args.start_year + 1,
            "manifest_files": len(manifest),
            "aggregate_rows": len(aggregate_rows),
            "coverage_categories": sorted(categories),
            "aggregate_sha256": sha256(args.aggregate_output),
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_files": len(manifest),
                "aggregate_rows": len(aggregate_rows),
                "coverage_categories": sorted(categories),
                "aggregate_output": str(args.aggregate_output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
