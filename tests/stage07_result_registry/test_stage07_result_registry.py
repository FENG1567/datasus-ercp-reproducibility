from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "stage07_build_result_registry.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage07_build_result_registry", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


target = load_module()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_source(root: Path, payload: dict) -> Path:
    source = root / "analytic" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(payload), encoding="utf-8")
    return source


def base_entry(source: Path, root: Path) -> dict:
    return {
        "result_id": "synthetic.primary_count",
        "display_label": "Synthetic primary count",
        "evidence_level": "descriptive",
        "estimand_or_description": "Synthetic contract test only.",
        "value": 12,
        "value_type": "numeric",
        "unit": "count",
        "denominator": {"value": 100, "unit": "records"},
        "suppression_status": "not_suppressed",
        "missing_status": "not_missing",
        "source_artifact": source.relative_to(root).as_posix(),
        "expected_sha256": sha(source),
        "source_schema_version": "synthetic_json_v1",
        "locator": {"type": "json_pointer", "pointer": "/metrics/primary_count"},
        "tolerance": 0,
        "allowed_consumers": ["main_text", "table"],
        "consumer_values": {"main_text": {"value": 12}, "table": {"value": 12}},
        "limitation": "Synthetic only.",
    }


def spec_for(entry: dict, version: str = "synthetic_rc_v1") -> dict:
    return {"schema_version": target.SPEC_SCHEMA, "candidate_version": version, "entries": [entry]}


def write_spec(root: Path, spec: dict, suffix: str = ".json") -> Path:
    path = root / f"spec{suffix}"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_json_pointer_traceability_and_immutable_release_first_artifact(tmp_path):
    source = write_json_source(tmp_path, {"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 12}})
    spec_path = write_spec(tmp_path, spec_for(base_entry(source, tmp_path)))
    output = tmp_path / "release"
    manifest = target.build_registry(spec_path, tmp_path, output)
    registry = json.loads((output / "result_registry.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "numeric_traceability_report.json").read_text(encoding="utf-8"))
    assert registry["entries"][0]["value"] == 12.0
    assert trace["entries"][0]["recomputed_value"] == 12.0
    assert target.GOVERNANCE_MARKER in registry["governance_marker"]
    assert manifest["code_sha256"] == sha(SOURCE)
    assert set(manifest["output_sha256"]) == {"result_registry.json", "result_registry.csv", "numeric_traceability_report.json", "consumer_consistency.json"}
    with pytest.raises(target.RegistryError, match="must be empty"):
        target.build_registry(spec_path, tmp_path, output)


def test_csv_row_key_and_tolerance(tmp_path):
    source = tmp_path / "analytic" / "source.csv"
    source.parent.mkdir()
    source.write_text("schema_version,metric,value\nsynthetic_csv_v1,rate,12.5004\n", encoding="utf-8")
    entry = base_entry(source, tmp_path)
    entry.update({"result_id": "synthetic.csv_rate", "unit": "rate", "value": 12.5, "tolerance": 0.001, "source_schema_version": "synthetic_csv_v1", "locator": {"type": "csv_row_key", "row_key": {"metric": "rate"}, "column": "value"}, "consumer_values": {"main_text": {"value": 12.5}, "table": {"value": 12.5}}})
    spec_path = write_spec(tmp_path, spec_for(entry))
    target.build_registry(spec_path, tmp_path, tmp_path / "release")


def test_parquet_equality_filter_aggregations(tmp_path):
    frame = target.pd.DataFrame({"group": ["A", "A", "B"], "value": [4.0, 6.0, 99.0]})
    source = tmp_path / "analytic" / "source.parquet"
    source.parent.mkdir()
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(frame).replace_schema_metadata({b"schema_version": b"synthetic_parquet_v1"})
    pq.write_table(table, source)
    entry = base_entry(source, tmp_path)
    entry.update({"result_id": "synthetic.parquet_mean", "unit": "rate", "value": 5.0, "source_schema_version": "synthetic_parquet_v1", "locator": {"type": "parquet_aggregate", "filters": {"group": "A"}, "column": "value", "aggregation": "mean"}})
    entry["expected_sha256"] = sha(source)
    entry["consumer_values"] = {"main_text": {"value": 5.0}, "table": {"value": 5.0}}
    spec_path = write_spec(tmp_path, spec_for(entry))
    target.build_registry(spec_path, tmp_path, tmp_path / "release")


def test_explicit_nonnumeric_literal_status_is_traceable(tmp_path):
    source = write_json_source(tmp_path, {"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 12}})
    entry = base_entry(source, tmp_path)
    entry.update({"result_id": "synthetic.literal", "value_type": "string", "value": "NOT_ESTIMABLE", "unit": "status", "explicitly_nonnumeric": True, "locator": {"type": "literal_status", "value": "NOT_ESTIMABLE"}, "consumer_values": {"supplement": {"value": "NOT_ESTIMABLE"}}, "allowed_consumers": ["supplement"]})
    target.build_registry(write_spec(tmp_path, spec_for(entry)), tmp_path, tmp_path / "release")


@pytest.mark.parametrize("mutator,pattern", [
    (lambda entry: entry.update(expected_sha256="0" * 64), "SHA-256 mismatch"),
    (lambda entry: entry.update(source_schema_version="wrong"), "schema mismatch"),
    (lambda entry: entry.update(source_artifact="../escape.json"), "escapes declared project root"),
    (lambda entry: entry.update(source_artifact="C:/outside/source.json"), "absolute source_artifact"),
    (lambda entry: entry.update(evidence_level="causal-eligible"), "only descriptive or associational"),
    (lambda entry: entry.update(value=3), "does not match recomputed"),
])
def test_fail_closed_hash_schema_path_evidence_and_value(tmp_path, mutator, pattern):
    source = write_json_source(tmp_path, {"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 12}})
    entry = base_entry(source, tmp_path)
    mutator(entry)
    with pytest.raises(target.RegistryError, match=pattern):
        target.build_registry(write_spec(tmp_path, spec_for(entry)), tmp_path, tmp_path / "release")


def test_duplicate_consumer_mismatch_and_privacy_are_rejected(tmp_path):
    source = write_json_source(tmp_path, {"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 12}})
    first = base_entry(source, tmp_path)
    duplicate = dict(first)
    with pytest.raises(target.RegistryError, match="duplicate result_id"):
        target.build_registry(write_spec(tmp_path, {"schema_version": target.SPEC_SCHEMA, "candidate_version": "v1", "entries": [first, duplicate]}), tmp_path, tmp_path / "duplicate")
    mismatch = base_entry(source, tmp_path)
    mismatch["consumer_values"] = {"main_text": {"value": 13}}
    with pytest.raises(target.RegistryError, match="consumer value diverges"):
        target.build_registry(write_spec(tmp_path, spec_for(mismatch)), tmp_path, tmp_path / "mismatch")
    small = base_entry(source, tmp_path)
    small["value"] = 3
    small["locator"] = {"type": "literal_status", "value": "NOT_ALLOWED"}
    small["value_type"] = "string"
    small["explicitly_nonnumeric"] = True
    # Use a separate source-valued numeric entry for the privacy gate.
    source.write_text(json.dumps({"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 3}}), encoding="utf-8")
    small = base_entry(source, tmp_path)
    small["value"] = 3
    small["expected_sha256"] = sha(source)
    small["consumer_values"] = {"main_text": {"value": 3}}
    with pytest.raises(target.RegistryError, match="small count"):
        target.build_registry(write_spec(tmp_path, spec_for(small)), tmp_path, tmp_path / "privacy")


def test_multiple_csv_rows_and_unsupported_aggregation_fail_closed(tmp_path):
    source = tmp_path / "analytic" / "source.csv"
    source.parent.mkdir()
    source.write_text("schema_version,metric,value\nsynthetic_csv_v1,rate,12\nsynthetic_csv_v1,rate,12\n", encoding="utf-8")
    entry = base_entry(source, tmp_path)
    entry.update({"source_schema_version": "synthetic_csv_v1", "locator": {"type": "csv_row_key", "row_key": {"metric": "rate"}, "column": "value"}, "expected_sha256": sha(source)})
    with pytest.raises(target.RegistryError, match="exactly one row"):
        target.build_registry(write_spec(tmp_path, spec_for(entry)), tmp_path, tmp_path / "rows")

    frame = target.pd.DataFrame({"group": ["A"], "value": [12.0]})
    parquet = tmp_path / "analytic" / "source.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(frame).replace_schema_metadata({b"schema_version": b"synthetic_parquet_v1"}), parquet)
    entry = base_entry(parquet, tmp_path)
    entry.update({"source_schema_version": "synthetic_parquet_v1", "expected_sha256": sha(parquet), "locator": {"type": "parquet_aggregate", "filters": {"group": "A"}, "column": "value", "aggregation": "variance"}})
    with pytest.raises(target.RegistryError, match="unsupported Parquet aggregation"):
        target.build_registry(write_spec(tmp_path, spec_for(entry)), tmp_path, tmp_path / "aggregation")


def test_literal_status_requires_explicit_nonnumeric_and_raw_sources_rejected(tmp_path):
    source = write_json_source(tmp_path, {"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 12}})
    entry = base_entry(source, tmp_path)
    entry.update({"value_type": "string", "value": "NOT_ESTIMABLE", "unit": "status", "locator": {"type": "literal_status", "value": "NOT_ESTIMABLE"}})
    with pytest.raises(target.RegistryError, match="explicitly nonnumeric"):
        target.build_registry(write_spec(tmp_path, spec_for(entry)), tmp_path, tmp_path / "literal")
    raw = tmp_path / "data_raw" / "source.json"
    raw.parent.mkdir()
    raw.write_text(json.dumps({"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 12}}), encoding="utf-8")
    entry = base_entry(raw, tmp_path)
    with pytest.raises(target.RegistryError, match="data_raw"):
        target.build_registry(write_spec(tmp_path, spec_for(entry)), tmp_path, tmp_path / "raw")


def test_invalidation_fingerprint_changes_for_candidate_version_and_source(tmp_path):
    source = write_json_source(tmp_path, {"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 12}})
    entry = base_entry(source, tmp_path)
    manifest_one = target.build_registry(write_spec(tmp_path, spec_for(entry, "v1")), tmp_path, tmp_path / "one")
    entry_two = base_entry(source, tmp_path)
    manifest_two = target.build_registry(write_spec(tmp_path, spec_for(entry_two, "v2")), tmp_path, tmp_path / "two")
    assert manifest_one["invalidation_fingerprint"] != manifest_two["invalidation_fingerprint"]
    source.write_text(json.dumps({"schema_version": "synthetic_json_v1", "metrics": {"primary_count": 13}}), encoding="utf-8")
    entry_three = base_entry(source, tmp_path)
    entry_three["value"] = 13
    entry_three["consumer_values"] = {"main_text": {"value": 13}, "table": {"value": 13}}
    manifest_three = target.build_registry(write_spec(tmp_path, spec_for(entry_three, "v1")), tmp_path, tmp_path / "three")
    assert manifest_one["invalidation_fingerprint"] != manifest_three["invalidation_fingerprint"]
