from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class SingleInstanceLock:
    """Cross-process mutual exclusion for the downloader via O_EXCL lock file."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def acquire(self) -> None:
        try:
            self._handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._handle, str(os.getpid()).encode("ascii"))
        except FileExistsError as exc:
            raise RuntimeError(f"Another downloader instance holds {self.path}") from exc

    def release(self) -> None:
        if self._handle is not None:
            os.close(self._handle)
            self._handle = None
        self.path.unlink(missing_ok=True)


LIST_ENDPOINT = "https://datasus.saude.gov.br/wp-content/ftp.php"
BUNDLE_ENDPOINT = "https://datasus.saude.gov.br/wp-content/download.php"
PORTAL_PAGE = "https://datasus.saude.gov.br/transferencia-de-arquivos/"
USER_AGENT = "DATASUS-ERCP-reproducible-research/1.0"
UF_LIST = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]
MANIFEST_FIELDS = [
    "dataset", "filename", "uf", "year", "month", "source_ftp_url", "portal_page",
    "list_endpoint", "bundle_endpoint", "bundle_url", "bundle_size_bytes", "bundle_sha256",
    "local_relative_path", "local_size_bytes", "sha256", "accessed_at", "status", "error_class",
]


class PortalRequestError(RuntimeError):
    """A sanitized, provenance-safe portal request failure."""

    def __init__(self, operation: str, error_class: str, http_status: int | None = None):
        self.operation = operation
        self.error_class = error_class
        self.http_status = http_status
        status = f", http_status={http_status}" if http_status is not None else ""
        super().__init__(f"{operation} failed: {error_class}{status}")


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
    record = {"timestamp": utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def post_form(
    url: str,
    pairs: list[tuple[str, str]],
    operation: str,
    timeout: int = 600,
    attempts: int = 5,
) -> bytes:
    body = urllib.parse.urlencode(pairs).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                data=body,
                headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                retry_after = None
                if isinstance(exc, HTTPError) and exc.headers:
                    retry_after = exc.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * 2 ** (attempt - 1))
                time.sleep(delay)
    http_status = last_error.code if isinstance(last_error, HTTPError) else None
    raise PortalRequestError(operation, type(last_error).__name__, http_status) from last_error


def list_files(dataset: str, year: int, months: list[int]) -> list[dict]:
    if dataset == "CNES_PF":
        source, file_type = "CNES", "PF"
        pattern = re.compile(r"^PF([A-Z]{2})(\d{2})(\d{2})\.dbc$", re.I)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    pairs: list[tuple[str, str]] = [
        ("tipo_arquivo[]", file_type),
        ("modalidade[]", "1"),
        ("fonte[]", source),
        ("ano[]", str(year)),
    ]
    pairs.extend(("mes[]", f"{month:02d}") for month in months)
    pairs.extend(("uf[]", uf) for uf in UF_LIST)
    payload = json.loads(post_form(LIST_ENDPOINT, pairs, operation="file_listing").decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("File-list response is not a JSON list")
    selected: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("arquivo", ""))
        match = pattern.match(name)
        if not match:
            continue
        uf, yy, mm = match.groups()
        parsed_year, parsed_month = 2000 + int(yy), int(mm)
        address = str(item.get("endereco", ""))
        if parsed_year != year or parsed_month not in months or uf.upper() not in UF_LIST:
            continue
        if not address.startswith("ftp://ftp.datasus.gov.br/"):
            raise ValueError(f"Unexpected source address for {name}")
        selected.append(
            {"filename": name, "uf": uf.upper(), "year": parsed_year, "month": parsed_month, "source_ftp_url": address}
        )
    selected.sort(key=lambda row: (row["year"], row["month"], row["uf"], row["filename"]))
    expected = {(uf, month) for uf in UF_LIST for month in months}
    observed = {(row["uf"], row["month"]) for row in selected}
    if observed != expected or len(selected) != len(expected):
        missing = sorted(expected - observed)
        duplicates = len(selected) - len(observed)
        raise RuntimeError(f"Incomplete official listing: missing={missing[:10]}, duplicate_count={duplicates}")
    return selected


def find_first_url(value):
    if isinstance(value, str) and value.startswith("https://"):
        return value.replace("\\/", "/")
    if isinstance(value, list):
        for item in value:
            found = find_first_url(item)
            if found:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = find_first_url(item)
            if found:
                return found
    return None


def create_bundle(files: list[dict]) -> str:
    pairs: list[tuple[str, str]] = []
    for index, row in enumerate(files):
        pairs.append((f"dados[{index}][arquivo]", row["filename"]))
        pairs.append((f"dados[{index}][link]", row["source_ftp_url"]))
    payload = json.loads(
        post_form(BUNDLE_ENDPOINT, pairs, operation="bundle_creation", timeout=900).decode("utf-8")
    )
    url = find_first_url(payload)
    if not url or not url.startswith("https://datasus.saude.gov.br/wp-content/zipupload/"):
        raise RuntimeError("Official bundle endpoint did not return an accepted HTTPS URL")
    return url


def server_ready(url: str, timeout: float = 30.0) -> bool:
    """Quick probe: the official bundle endpoint packages archives
    asynchronously, and connections are dropped or stall while a bundle is
    still being built. A HEAD/GET probe avoids burning a 900-second
    urlopen timeout on a not-yet-ready archive."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return length is not None and length.isdigit() and int(length) > 0
    except Exception:
        return False


def expected_size(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit():
                return int(length)
    except Exception:
        pass
    return None


def download(url: str, destination: Path, attempts: int = 0, base_delay: float = 20.0) -> None:
    """Endless resumable download (Range-aware) robust to intermittent TLS
    resets observed on outbound connections to the official portal.
    The completed file is verified against the declared remote size; a
    mismatch restarts the transfer."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    attempt = 1
    while True:
        resume_from = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                content_length = response.headers.get("Content-Length")
                if resume_from and response.status == 206 and content_length:
                    declared = resume_from + int(content_length)
                    if declared < resume_from:
                        partial.unlink(missing_ok=True)
                        resume_from = 0
                        response.close()
                        raise RuntimeError("invalid range state, restarting")
                mode = "ab" if resume_from and response.status == 206 else "wb"
                if resume_from and response.status != 206:
                    resume_from = 0
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            expected = expected_size(url)
            if expected is not None and partial.stat().st_size != expected:
                print(
                    f"size mismatch: got {partial.stat().st_size}, expected {expected}; restarting", flush=True
                )
                partial.unlink(missing_ok=True)
                attempt += 1
                time.sleep(base_delay * min(attempt, 20))
                continue
            os.replace(partial, destination)
            return
        except Exception as exc:
            print(f"attempt {attempt} failed: {type(exc).__name__} (resumed={resume_from})", flush=True)
            attempt += 1
            if attempts and attempt > attempts:
                raise RuntimeError(
                    f"Bundle download failed after {attempts} attempts: {type(exc).__name__}"
                ) from exc
            time.sleep(base_delay * min(attempt, 20))


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def raw_file_is_complete(row: dict, raw_root: Path, manifest: dict[str, dict]) -> bool:
    recorded = manifest.get(row["filename"])
    if not recorded or recorded.get("status") != "complete":
        return False
    destination = raw_root / str(row["year"]) / row["uf"] / row["filename"]
    if not destination.exists():
        return False
    expected_size = str(recorded.get("local_size_bytes", ""))
    expected_hash = str(recorded.get("sha256", "")).lower()
    return (
        expected_size.isdigit()
        and destination.stat().st_size == int(expected_size)
        and bool(expected_hash)
        and sha256(destination).lower() == expected_hash
    )


def save_manifest(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (int(row["year"]), int(row["month"]), row["uf"])))
    os.replace(temporary, path)


def safe_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for member in archive.infolist():
        normalized = Path(member.filename.replace("\\", "/"))
        if member.is_dir():
            continue
        if normalized.is_absolute() or ".." in normalized.parts or len(normalized.parts) != 1:
            raise RuntimeError(f"Unsafe or unexpected ZIP member: {member.filename}")
        yield member


def extract_batch(
    dataset: str,
    files: list[dict],
    bundle: Path,
    bundle_url: str,
    raw_root: Path,
    manifest_path: Path,
) -> None:
    expected = {row["filename"]: row for row in files}
    bundle_hash = sha256(bundle)
    bundle_size = bundle.stat().st_size
    manifest = load_manifest(manifest_path)
    accessed_at = utc_now()
    with zipfile.ZipFile(bundle) as archive:
        members = list(safe_members(archive))
        names = {Path(member.filename).name for member in members}
        if names != set(expected):
            raise RuntimeError(f"ZIP/list mismatch: missing={sorted(set(expected)-names)[:10]}, extra={sorted(names-set(expected))[:10]}")
        for member in members:
            name = Path(member.filename).name
            metadata = expected[name]
            destination = raw_root / str(metadata["year"]) / metadata["uf"] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".dbc.part")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            new_hash = sha256(temporary)
            if destination.exists():
                if sha256(destination) != new_hash:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f"Existing immutable raw file differs: {destination}")
                temporary.unlink(missing_ok=True)
            else:
                os.replace(temporary, destination)
            manifest[name] = {
                "dataset": dataset,
                "filename": name,
                "uf": metadata["uf"],
                "year": metadata["year"],
                "month": metadata["month"],
                "source_ftp_url": metadata["source_ftp_url"],
                "portal_page": PORTAL_PAGE,
                "list_endpoint": LIST_ENDPOINT,
                "bundle_endpoint": BUNDLE_ENDPOINT,
                "bundle_url": bundle_url,
                "bundle_size_bytes": bundle_size,
                "bundle_sha256": bundle_hash,
                "local_relative_path": str(destination.relative_to(raw_root.parent.parent)).replace("\\", "/"),
                "local_size_bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "accessed_at": accessed_at,
                "status": "complete",
                "error_class": "",
            }
    save_manifest(manifest_path, manifest)


def file_batches(files: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return [files[index:index + batch_size] for index in range(0, len(files), batch_size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CNES_PF"], default="CNES_PF")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--only-month", type=int, choices=list(range(1, 13)))
    parser.add_argument("--files-per-bundle", type=int, choices=range(1, 19), default=18)
    parser.add_argument(
        "--inter-bundle-delay",
        type=float,
        default=2.0,
        help="Polite delay in seconds after each completed bundle.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        help="Bounded endpoint probe: process only the first N listed files in each selected year.",
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, default=Path("data_raw/cnes_pf_bundles"))
    parser.add_argument("--manifest", type=Path, default=Path("provenance/cnes_pf_download_manifest.csv"))
    parser.add_argument("--event-log", type=Path, default=Path("provenance/cnes_pf_download_events.jsonl"))
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path("logs/cnes_pf_download.lock"),
        help="O_EXCL lock file guarding a single downloader instance.",
    )
    args = parser.parse_args()

    if args.limit_files is not None and args.limit_files < 1:
        parser.error("--limit-files must be positive")
    if args.inter_bundle_delay < 0:
        parser.error("--inter-bundle-delay cannot be negative")

    lock = SingleInstanceLock(args.lock_file)
    try:
        lock.acquire()
    except RuntimeError as exc:
        print(f"ABORT {exc}", flush=True)
        return 2
    try:
        _run(args)
    finally:
        lock.release()
    return 0


def _run(args: argparse.Namespace) -> None:

    for year in range(args.start_year, args.end_year + 1):
        selected_months = [args.only_month] if args.only_month else list(range(1, 13))
        try:
            # Exactly one authoritative listing request per year. The response is
            # then split locally to keep the total run at 95 portal API calls:
            # 5 listings + 90 bundles of 18 files for 2021-2025.
            files_year = list_files(args.dataset, year, selected_months)
            if args.limit_files is not None:
                files_year = files_year[: args.limit_files]
            batches = file_batches(files_year, args.files_per_bundle)
            append_event(
                args.event_log,
                {
                    "status": "listing_pass",
                    "dataset": args.dataset,
                    "year": year,
                    "selected_months": selected_months,
                    "listed_files": len(files_year),
                    "batch_size": args.files_per_bundle,
                    "batch_count": len(batches),
                },
            )
            for batch_number, files in enumerate(batches, start=1):
                current_manifest = load_manifest(args.manifest)
                pending_files = [
                    row for row in files if not raw_file_is_complete(row, args.raw_root, current_manifest)
                ]
                if not pending_files:
                    append_event(
                        args.event_log,
                        {
                            "status": "batch_skipped_complete",
                            "dataset": args.dataset,
                            "year": year,
                            "batch_number": batch_number,
                            "batch_size": len(files),
                            "first_file": files[0]["filename"],
                            "last_file": files[-1]["filename"],
                        },
                    )
                    continue
                first_stem = Path(pending_files[0]["filename"]).stem
                last_stem = Path(pending_files[-1]["filename"]).stem
                batch_ok = False
                for attempt in range(1, 4):
                    suffix = f"B{batch_number:03d}_{first_stem}-{last_stem}"
                    bundle = args.bundle_root / f"CNES_PF_{year}_{suffix}.zip"
                    if bundle.exists() and zipfile.is_zipfile(bundle):
                        bundle_url = current_manifest.get(pending_files[0]["filename"], {}).get(
                            "bundle_url", "reused-local-bundle"
                        )
                    else:
                        bundle_url = create_bundle(pending_files)
                        append_event(
                            args.event_log,
                            {
                                "status": "bundle_created",
                                "dataset": args.dataset,
                                "year": year,
                                "batch_number": batch_number,
                                "batch_size": len(pending_files),
                                "first_file": pending_files[0]["filename"],
                                "last_file": pending_files[-1]["filename"],
                                "bundle_url": bundle_url,
                            },
                        )
                        download(bundle_url, bundle)
                    try:
                        extract_batch(
                            args.dataset, pending_files, bundle, bundle_url, args.raw_root, args.manifest
                        )
                        batch_ok = True
                        event = {
                            "status": "bundle_pass",
                            "dataset": args.dataset,
                            "year": year,
                            "batch_number": batch_number,
                            "batch_size": len(pending_files),
                            "first_file": pending_files[0]["filename"],
                            "last_file": pending_files[-1]["filename"],
                            "bundle": str(bundle),
                        }
                        append_event(args.event_log, event)
                        print(json.dumps({"status": "PASS", **event}, ensure_ascii=False), flush=True)
                        break
                    except Exception as exc:
                        bundle.unlink(missing_ok=True)
                        for name in pending_files:
                            raw_file = args.raw_root / str(name["year"]) / name["uf"] / name["filename"]
                            raw_file.unlink(missing_ok=True)
                        append_event(
                            args.event_log,
                            {
                                "status": "batch_retry",
                                "dataset": args.dataset,
                                "year": year,
                                "batch_number": batch_number,
                                "attempt": attempt,
                                "error_class": type(exc).__name__,
                            },
                        )
                        print(
                            json.dumps(
                                {"status": "BATCH_RETRY", "batch_number": batch_number,
                                 "attempt": attempt, "error_class": type(exc).__name__},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        time.sleep(20 * attempt)
                if not batch_ok:
                    append_event(
                        args.event_log,
                        {
                            "status": "batch_failed_after_retries",
                            "dataset": args.dataset,
                            "year": year,
                            "batch_number": batch_number,
                            "first_file": pending_files[0]["filename"],
                            "last_file": pending_files[-1]["filename"],
                        },
                    )
                if args.inter_bundle_delay:
                    time.sleep(args.inter_bundle_delay)
        except Exception as exc:
            append_event(
                args.event_log,
                {
                    "status": "failed",
                    "dataset": args.dataset,
                    "year": year,
                    "selected_months": selected_months,
                    "batch_size": args.files_per_bundle,
                    "error_class": getattr(exc, "error_class", type(exc).__name__),
                    "http_status": getattr(exc, "http_status", None),
                    "operation": getattr(exc, "operation", "local_validation_or_download"),
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
