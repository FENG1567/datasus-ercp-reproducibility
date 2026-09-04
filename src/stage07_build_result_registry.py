#!/usr/bin/env python3
"""Build a fail-closed, release-candidate result registry.

This module deliberately contains no study-specific result identifiers or paths.
It accepts only a declared specification and recomputes every registered value
from a hashed, versioned source artifact before writing an immutable release
directory.  The registry is intended to be the sole numeric source for later
tables, figures, and manuscript candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import stat
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SPEC_SCHEMA = "stage07_result_registry_spec_v1"
REGISTRY_SCHEMA = "stage07_result_registry_v1"
GOVERNANCE_MARKER = "prespecified reporting controls"
ALLOWED_EVIDENCE = {"descriptive", "associational"}
REJECTED_EVIDENCE = {"causal-eligible"}
ALLOWED_CONSUMERS = {"main_text", "table", "figure", "supplement"}
COUNT_UNITS = {"count", "counts", "patient", "patients", "case", "cases", "event", "events", "death", "deaths", "aih", "aihs", "admission", "admissions"}


class RegistryError(RuntimeError):
    """Raised for a contract violation that must block a release."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hash() -> str:
    return sha256_file(Path(__file__).resolve())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"{context} must be a finite numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise RegistryError(f"{context} cannot be NaN or Inf")
    return number


def _relative_source(project_root: Path, raw_path: Any) -> tuple[Path, str]:
    _require(isinstance(raw_path, str) and raw_path, "source_artifact must be a nonempty relative path")
    path = Path(raw_path)
    _require(not path.is_absolute(), "absolute source_artifact paths are forbidden")
    resolved_root = project_root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RegistryError("source_artifact escapes declared project root") from exc
    _require(resolved.is_file(), f"source_artifact does not exist: {relative.as_posix()}")
    return resolved, relative.as_posix()


def _load_spec(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"registry specification does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise RegistryError(f"invalid registry specification: {exc}") from exc
    _require(isinstance(loaded, dict), "registry specification must be a mapping")
    return loaded


def _json_pointer(document: Any, pointer: str) -> Any:
    _require(isinstance(pointer, str) and pointer.startswith("/"), "JSON pointer must begin with '/'")
    current = document
    for encoded in pointer.split("/")[1:]:
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            _require(key in current, f"JSON pointer token not found: {key}")
            current = current[key]
        elif isinstance(current, list):
            _require(key.isdigit(), f"JSON array pointer token must be an index: {key}")
            index = int(key)
            _require(0 <= index < len(current), f"JSON array index out of range: {key}")
            current = current[index]
        else:
            raise RegistryError(f"JSON pointer traverses a scalar before token: {key}")
    return current


def _schema_from_json(path: Path, expected: str) -> tuple[Any, str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid JSON source artifact: {path.name}") from exc
    _require(isinstance(document, dict), "JSON source artifact must be an object with schema_version")
    actual = document.get("schema_version")
    _require(actual == expected, f"source schema mismatch for {path.name}: expected {expected!r}, got {actual!r}")
    return document, str(actual)


def _schema_from_csv(path: Path, expected: str) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    _require("schema_version" in frame.columns, "CSV source must contain a schema_version column")
    versions = set(frame["schema_version"].dropna().astype(str).unique())
    _require(versions == {expected}, f"source schema mismatch for {path.name}: expected only {expected!r}, got {sorted(versions)!r}")
    return frame, expected


def _schema_from_parquet(path: Path, expected: str) -> tuple[pd.DataFrame, str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency is explicit in the project environment
        raise RegistryError("pyarrow is required for Parquet result sources") from exc
    metadata = pq.ParquetFile(path).metadata.metadata or {}
    actual_raw = metadata.get(b"schema_version") or metadata.get(b"stage07_schema_version")
    actual = actual_raw.decode("utf-8") if actual_raw is not None else None
    _require(actual == expected, f"source schema mismatch for {path.name}: expected {expected!r}, got {actual!r}")
    return pd.read_parquet(path), actual


def _entry_source(entry: dict[str, Any], project_root: Path) -> tuple[Path, str, str, str]:
    path, relative = _relative_source(project_root, entry.get("source_artifact"))
    expected_hash = entry.get("expected_sha256")
    _require(isinstance(expected_hash, str) and len(expected_hash) == 64, "expected_sha256 must be a SHA-256 hex digest")
    actual_hash = sha256_file(path)
    _require(actual_hash.lower() == expected_hash.lower(), f"source SHA-256 mismatch for {relative}")
    expected_schema = entry.get("source_schema_version")
    _require(isinstance(expected_schema, str) and expected_schema, "source_schema_version is required")
    return path, relative, actual_hash, expected_schema


def _extract_value(entry: dict[str, Any], project_root: Path) -> tuple[Any, dict[str, Any]]:
    path, relative, actual_hash, expected_schema = _entry_source(entry, project_root)
    locator = entry.get("locator")
    _require(isinstance(locator, dict), "locator must be a mapping")
    kind = locator.get("type")
    expected_kind = entry.get("value_type")
    _require(expected_kind in {"numeric", "string"}, "value_type must be 'numeric' or 'string'")
    provenance = {
        "source_artifact": relative,
        "source_sha256": actual_hash,
        "source_schema_version": expected_schema,
        "locator": locator,
    }
    if kind == "json_pointer":
        document, actual_schema = _schema_from_json(path, expected_schema)
        provenance["observed_schema_version"] = actual_schema
        value = _json_pointer(document, locator.get("pointer"))
    elif kind == "csv_row_key":
        frame, actual_schema = _schema_from_csv(path, expected_schema)
        provenance["observed_schema_version"] = actual_schema
        row_key, column = locator.get("row_key"), locator.get("column")
        _require(isinstance(row_key, dict) and row_key, "CSV locator row_key must be a nonempty mapping")
        _require(isinstance(column, str) and column in frame.columns, "CSV locator column is absent")
        for column_name, expected_value in row_key.items():
            _require(column_name in frame.columns, f"CSV row_key column is absent: {column_name}")
            frame = frame.loc[frame[column_name].astype(str) == str(expected_value)]
        _require(len(frame) == 1, f"CSV row_key must select exactly one row; selected {len(frame)}")
        value = frame.iloc[0][column]
    elif kind == "parquet_aggregate":
        frame, actual_schema = _schema_from_parquet(path, expected_schema)
        provenance["observed_schema_version"] = actual_schema
        filters, column, aggregation = locator.get("filters"), locator.get("column"), locator.get("aggregation")
        _require(isinstance(filters, dict) and filters, "Parquet locator filters must be a nonempty equality mapping")
        _require(isinstance(column, str) and column in frame.columns, "Parquet locator column is absent")
        _require(aggregation in {"count", "sum", "mean", "median", "min", "max", "unique_count"}, "unsupported Parquet aggregation")
        for column_name, expected_value in filters.items():
            _require(column_name in frame.columns, f"Parquet filter column is absent: {column_name}")
            frame = frame.loc[frame[column_name].astype(str) == str(expected_value)]
        _require(len(frame) > 0, "Parquet filters selected zero rows")
        series = frame[column]
        if aggregation == "count":
            value = int(series.count())
        elif aggregation == "unique_count":
            value = int(series.nunique(dropna=True))
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            _require(numeric.notna().all(), "Parquet numeric aggregation encountered missing/non-numeric values")
            value = {"sum": numeric.sum, "mean": numeric.mean, "median": numeric.median, "min": numeric.min, "max": numeric.max}[aggregation]()
    elif kind == "literal_status":
        _require(expected_kind == "string" and entry.get("explicitly_nonnumeric") is True, "literal_status is allowed only for explicitly nonnumeric values")
        value = locator.get("value")
        _require(isinstance(value, str) and value, "literal_status value must be a nonempty string")
        provenance["observed_schema_version"] = "literal_status"
    else:
        raise RegistryError(f"unsupported locator type: {kind!r}")
    if expected_kind == "numeric":
        value = _finite_number(value, f"recomputed value for {entry.get('result_id')!r}")
    else:
        _require(isinstance(value, str) and value, "recomputed string values must be nonempty")
    return value, provenance


def _same_value(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _validate_entry_contract(entry: Any, project_root: Path) -> None:
    _require(isinstance(entry, dict), "each registry entry must be a mapping")
    required = ["result_id", "display_label", "evidence_level", "estimand_or_description", "value", "value_type", "unit", "missing_status", "suppression_status", "source_artifact", "expected_sha256", "source_schema_version", "locator", "tolerance", "allowed_consumers"]
    for name in required:
        _require(name in entry, f"registry entry is missing required field: {name}")
    _require(isinstance(entry["result_id"], str) and entry["result_id"].strip(), "result_id must be a nonempty stable string")
    _require(isinstance(entry["display_label"], str) and entry["display_label"].strip(), "display_label must be nonempty")
    evidence = entry["evidence_level"]
    _require(evidence not in REJECTED_EVIDENCE and evidence in ALLOWED_EVIDENCE, "only descriptive or associational evidence is authorized")
    _require(isinstance(entry["estimand_or_description"], str) and entry["estimand_or_description"].strip(), "estimand_or_description must be nonempty")
    _require(isinstance(entry["unit"], str) and entry["unit"].strip(), "unit must be nonempty")
    _require(entry["missing_status"] in {"not_missing", "missing", "not_applicable"}, "invalid missing_status")
    _require(entry["suppression_status"] in {"not_suppressed", "suppressed", "missing", "not_applicable"}, "invalid suppression_status")
    tolerance = _finite_number(entry["tolerance"], "tolerance")
    _require(tolerance >= 0, "tolerance must be nonnegative")
    consumers = entry["allowed_consumers"]
    _require(isinstance(consumers, list) and consumers, "allowed_consumers must be a nonempty list")
    _require(set(consumers).issubset(ALLOWED_CONSUMERS) and len(set(consumers)) == len(consumers), "allowed_consumers contains an invalid or duplicate consumer")
    if entry.get("denominator") is not None:
        denominator = entry["denominator"]
        _require(isinstance(denominator, dict) and {"value", "unit"}.issubset(denominator), "denominator must include value and unit")
        _finite_number(denominator["value"], "denominator.value")
        _require(isinstance(denominator["unit"], str) and denominator["unit"].strip(), "denominator.unit must be nonempty")
    # A raw source is never a release-consumer source: release candidates must
    # point to staged/frozen analytical artifacts, not immutable raw inputs.
    relative = Path(str(entry["source_artifact"]).replace("\\", "/"))
    if relative.parts and relative.parts[0] == "data_raw" and set(consumers) & ALLOWED_CONSUMERS:
        raise RegistryError("release consumers cannot source a registry value directly from data_raw")
    _relative_source(project_root, entry["source_artifact"])


def _privacy_gate(entry: dict[str, Any], value: Any) -> None:
    if entry["value_type"] != "numeric" or entry["suppression_status"] != "not_suppressed":
        return
    is_count = entry.get("value_is_count") is True or entry["unit"].strip().lower() in COUNT_UNITS
    if is_count and 0 <= float(value) < 5:
        raise RegistryError(f"unsuppressed small count (<5) is prohibited for {entry['result_id']}")


def _consumer_consistency(entry: dict[str, Any], value: Any) -> dict[str, Any]:
    declared = entry.get("consumer_values", {})
    _require(isinstance(declared, dict), "consumer_values must be a mapping when supplied")
    extra = set(declared) - set(entry["allowed_consumers"])
    _require(not extra, f"consumer_values declares unallowed consumers: {sorted(extra)}")
    tolerance = float(entry["tolerance"])
    audits: dict[str, Any] = {}
    for consumer, candidate in declared.items():
        candidate_value = candidate.get("value") if isinstance(candidate, dict) else candidate
        _require(_same_value(candidate_value, value, tolerance), f"consumer value diverges for {entry['result_id']} in {consumer}")
        audits[consumer] = {"value": candidate_value, "matches_registry": True}
    return audits


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _make_readonly(path: Path) -> None:
    path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def build_registry(spec_path: Path, project_root: Path, output_dir: Path) -> dict[str, Any]:
    """Validate *spec_path* and emit an immutable, self-contained registry.

    ``output_dir`` must exist or be newly created and must be empty.  This
    prevents accidental mutation of a prior release candidate.
    """
    spec_path = spec_path.resolve()
    root = project_root.resolve()
    _require(root.is_dir(), "declared project root does not exist")
    if output_dir.exists():
        _require(output_dir.is_dir(), "output path exists but is not a directory")
        _require(not any(output_dir.iterdir()), "output directory must be empty; prior releases are immutable")
    else:
        output_dir.mkdir(parents=True)
    spec = _load_spec(spec_path)
    _require(spec.get("schema_version") == SPEC_SCHEMA, f"spec schema_version must be {SPEC_SCHEMA!r}")
    candidate_version = spec.get("candidate_version")
    _require(isinstance(candidate_version, str) and candidate_version.strip(), "candidate_version is required")
    entries = spec.get("entries")
    _require(isinstance(entries, list) and entries, "entries must be a nonempty list")
    result_ids: set[str] = set()
    registry_entries: list[dict[str, Any]] = []
    trace_entries: list[dict[str, Any]] = []
    consistency: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for entry in entries:
        _validate_entry_contract(entry, root)
        result_id = entry["result_id"]
        _require(result_id not in result_ids, f"duplicate result_id: {result_id}")
        result_ids.add(result_id)
        recomputed, provenance = _extract_value(entry, root)
        declared_value = entry["value"]
        if entry["value_type"] == "numeric":
            declared_value = _finite_number(declared_value, f"declared value for {result_id}")
        else:
            _require(isinstance(declared_value, str) and declared_value, "declared string value must be nonempty")
        _require(_same_value(recomputed, declared_value, float(entry["tolerance"])), f"declared value does not match recomputed source for {result_id}")
        _privacy_gate(entry, recomputed)
        audits = _consumer_consistency(entry, recomputed)
        source_hashes[provenance["source_artifact"]] = provenance["source_sha256"]
        public = {
            "result_id": result_id,
            "display_label": entry["display_label"],
            "evidence_level": entry["evidence_level"],
            "estimand_or_description": entry["estimand_or_description"],
            "value": recomputed,
            "value_type": entry["value_type"],
            "unit": entry["unit"],
            "denominator": entry.get("denominator"),
            "suppression_status": entry["suppression_status"],
            "missing_status": entry.get("missing_status", "not_missing"),
            "allowed_consumers": entry["allowed_consumers"],
            "limitation": entry.get("limitation", "Not specified."),
            "governance_marker": GOVERNANCE_MARKER,
            "source": provenance,
        }
        registry_entries.append(public)
        trace_entries.append({"result_id": result_id, "declared_value": declared_value, "recomputed_value": recomputed, "matches": True, "tolerance": entry["tolerance"], "source": provenance, "governance_marker": GOVERNANCE_MARKER})
        consistency[result_id] = {"allowed_consumers": entry["allowed_consumers"], "checked_consumers": audits, "consistent": True, "governance_marker": GOVERNANCE_MARKER}
    spec_hash = sha256_file(spec_path)
    invalidation_fingerprint = sha256_bytes(canonical_json({"candidate_version": candidate_version, "spec_sha256": spec_hash, "code_sha256": code_hash(), "source_sha256": source_hashes}).encode("utf-8"))
    registry = {"schema_version": REGISTRY_SCHEMA, "candidate_version": candidate_version, "invalidation_fingerprint": invalidation_fingerprint, "governance_marker": GOVERNANCE_MARKER, "entries": registry_entries}
    traceability = {"schema_version": "stage07_numeric_traceability_v1", "candidate_version": candidate_version, "invalidation_fingerprint": invalidation_fingerprint, "governance_marker": GOVERNANCE_MARKER, "entries": trace_entries}
    consumer_report = {"schema_version": "stage07_consumer_consistency_v1", "candidate_version": candidate_version, "invalidation_fingerprint": invalidation_fingerprint, "governance_marker": GOVERNANCE_MARKER, "results": consistency}
    json_path = output_dir / "result_registry.json"
    csv_path = output_dir / "result_registry.csv"
    trace_path = output_dir / "numeric_traceability_report.json"
    consistency_path = output_dir / "consumer_consistency.json"
    _write_json(json_path, registry)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["result_id", "display_label", "evidence_level", "value", "value_type", "unit", "denominator_value", "denominator_unit", "suppression_status", "missing_status", "allowed_consumers", "invalidation_fingerprint"])
        writer.writeheader()
        for item in registry_entries:
            denominator = item["denominator"] or {}
            writer.writerow({"result_id": item["result_id"], "display_label": item["display_label"], "evidence_level": item["evidence_level"], "value": item["value"], "value_type": item["value_type"], "unit": item["unit"], "denominator_value": denominator.get("value"), "denominator_unit": denominator.get("unit"), "suppression_status": item["suppression_status"], "missing_status": item["missing_status"], "allowed_consumers": ";".join(item["allowed_consumers"]), "invalidation_fingerprint": invalidation_fingerprint})
    _write_json(trace_path, traceability)
    _write_json(consistency_path, consumer_report)
    outputs = {path.name: sha256_file(path) for path in (json_path, csv_path, trace_path, consistency_path)}
    manifest = {"schema_version": "stage07_registry_manifest_v1", "candidate_version": candidate_version, "governance_marker": GOVERNANCE_MARKER, "code_sha256": code_hash(), "input_spec_sha256": spec_hash, "source_sha256": source_hashes, "output_sha256": outputs, "invalidation_fingerprint": invalidation_fingerprint}
    manifest_path = output_dir / "registry_manifest.json"
    _write_json(manifest_path, manifest)
    for path in (json_path, csv_path, trace_path, consistency_path, manifest_path):
        _make_readonly(path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON or YAML registry specification")
    parser.add_argument("--project-root", type=Path, required=True, help="declared project root containing relative sources")
    parser.add_argument("--output-dir", type=Path, required=True, help="empty directory for immutable registry artifacts")
    args = parser.parse_args(argv)
    try:
        manifest = build_registry(args.spec, args.project_root, args.output_dir)
    except RegistryError as exc:
        print(f"REGISTRY BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(canonical_json({"status": "PASS", "registry_manifest": str(args.output_dir / "registry_manifest.json"), "invalidation_fingerprint": manifest["invalidation_fingerprint"], "governance_marker": GOVERNANCE_MARKER}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
