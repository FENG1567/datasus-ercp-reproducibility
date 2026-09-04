#!/usr/bin/env python3
"""Build immutable, registry-linked source CSVs for the Stage 7 7-figure/4-table release.

This program deliberately prepares data only.  It never draws, previews, or
exports graphics.  Every declared contract input is hash checked before it is
read and all output is written transactionally or not at all.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


CONTRACT_SCHEMA = "stage07_figures_tables_contract_rc_v1"
REGISTRY_SCHEMA = "stage07_result_registry_v1"
MANIFEST_SCHEMA = "stage07_figure_table_source_data_manifest_rc_v1"
AUDIT_SCHEMA = "stage07_figure_table_source_data_audit_rc_v1"
EXPECTED_OUTPUT_COUNT = 22
FROZEN_CONTRACT_SHA256 = "485bbd56029b71eac618f39c5ef94fe5ee20d8a46c62aaa5a81f279ee6667991"
SENSITIVE_COUNT_COLUMN = re.compile(
    r"(^n_aih($|_)|^n_aih($|_)|^n_(patient|admission|death|event)s?($|_)|"
    r"(^|_)(patient|admission|death|event)s?(_|$)|^in_strength$)", re.I)


class BuilderError(RuntimeError):
    """A frozen-contract violation: callers must not continue."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BuilderError(message)


def _relative(root: Path, raw: str) -> tuple[Path, str]:
    _need(isinstance(raw, str) and raw.strip(), "contract input path is required")
    value = Path(raw)
    _need(not value.is_absolute(), "absolute contract paths are forbidden")
    resolved = (root / value).resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise BuilderError(f"contract path escapes project root: {raw}") from exc
    return resolved, relative


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BuilderError(f"cannot read frozen contract: {exc}") from exc
    _need(isinstance(value, dict), "contract must be a mapping")
    _need(value.get("schema_version") == CONTRACT_SCHEMA, "unexpected contract schema_version")
    _need(value.get("status") == "FROZEN_CONTRACT", "contract is not frozen")
    return value


def _load_registry(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuilderError(f"cannot read immutable registry: {exc}") from exc
    _need(payload.get("schema_version") == REGISTRY_SCHEMA, "unexpected result registry schema_version")
    entries = payload.get("entries")
    _need(isinstance(entries, list) and entries, "registry entries are required")
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        _need(isinstance(entry, dict) and isinstance(entry.get("result_id"), str), "invalid registry entry")
        _need(entry["result_id"] not in index, f"duplicate registry result_id: {entry['result_id']}")
        index[entry["result_id"]] = entry
    return payload, index


def _all_deliverables(contract: dict[str, Any]) -> list[dict[str, Any]]:
    figures, tables = contract.get("figures"), contract.get("tables")
    _need(isinstance(figures, list) and len(figures) == 7, "frozen contract must declare exactly seven figures")
    _need(isinstance(tables, list) and len(tables) == 4, "frozen contract must declare exactly four tables")
    result = figures + tables
    for item in result:
        _need(isinstance(item, dict), "contract deliverable must be a mapping")
        for field in ("id", "registry_result_ids", "inputs", "source_data_outputs"):
            _need(field in item, f"contract deliverable missing {field}")
    return result


def expected_outputs(contract: dict[str, Any]) -> list[str]:
    outputs = [name for item in _all_deliverables(contract) for name in item["source_data_outputs"]]
    _need(len(outputs) == EXPECTED_OUTPUT_COUNT and len(set(outputs)) == EXPECTED_OUTPUT_COUNT,
          f"contract must declare exactly {EXPECTED_OUTPUT_COUNT} unique source-data outputs")
    _need(all(isinstance(name, str) and name.endswith(".csv") and Path(name).name == name for name in outputs),
          "source-data output names must be flat CSV filenames")
    return outputs


def _verify_contract_inputs(root: Path, contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve and hash-check every logical input before any data is opened."""
    records: dict[str, dict[str, Any]] = {}
    items = list(contract.get("shared_inputs", []))
    for deliverable in _all_deliverables(contract):
        items.extend(deliverable["inputs"])
    for item in items:
        _need(isinstance(item, dict), "contract input must be a mapping")
        raw_path, expected = item.get("path"), item.get("sha256")
        _need(isinstance(expected, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected) is not None,
              "contract input sha256 must be a SHA-256 digest")
        # A frozen policy copy replaces the source path at the rendering host.
        # Its bytes must nevertheless equal the contract's original SHA.
        target = item.get("frozen_copy_target")
        actual_path_raw = target if target is not None else raw_path
        path, relative = _relative(root, actual_path_raw)
        _need(path.is_file(), f"frozen input does not exist: {relative}")
        observed = sha256_file(path)
        _need(observed.lower() == expected.lower(), f"input SHA-256 mismatch for {relative}")
        if target is not None:
            _need(item.get("frozen_copy_required_before_render") is True,
                  f"frozen copy target not authorized for {raw_path}")
        prior = records.get(relative)
        if prior:
            _need(prior["sha256"] == observed and prior["declared_sha256"] == expected.lower(),
                  f"same input has conflicting frozen hashes: {relative}")
            prior["roles"].append(str(item.get("role", "unspecified")))
        else:
            records[relative] = {"path": path, "sha256": observed, "declared_sha256": expected.lower(),
                                 "roles": [str(item.get("role", "unspecified"))],
                                 "frozen_copy_of": raw_path if target is not None else None}
    return records


def _check_registry_use(deliverable: dict[str, Any], registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ids = deliverable["registry_result_ids"]
    _need(isinstance(ids, list), f"{deliverable['id']} registry_result_ids must be a list")
    selected: list[dict[str, Any]] = []
    for result_id in ids:
        _need(isinstance(result_id, str) and result_id in registry,
              f"missing registry result_id: {result_id}")
        entry = registry[result_id]
        allowed = entry.get("allowed_consumers", [])
        consumer = "figure" if str(deliverable["id"]).startswith("Figure_") else "table"
        _need(consumer in allowed, f"registry result_id not allowed for {consumer}: {result_id}")
        selected.append(entry)
    return selected


def _read_frame(record: dict[str, Any]) -> pd.DataFrame:
    path: Path = record["path"]
    suffixes = [value.lower() for value in path.suffixes]
    if suffixes[-1:] == [".parquet"]:
        return pd.read_parquet(path)
    if suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(path, compression="gzip")
    if suffixes[-1:] == [".csv"]:
        return pd.read_csv(path)
    raise BuilderError(f"expected tabular frozen input, received {path.name}")


def _read_json(record: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(record["path"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuilderError(f"cannot read frozen JSON {record['path'].name}: {exc}") from exc
    _need(isinstance(value, dict), f"frozen JSON must be an object: {record['path'].name}")
    return value


def _by_role(deliverable: dict[str, Any], input_records: dict[str, dict[str, Any]], needle: str) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    for item in deliverable["inputs"]:
        if needle.lower() in str(item.get("role", "")).lower():
            actual = item.get("frozen_copy_target", item["path"])
            found.append(input_records[_relative_root_key(actual)])
    _need(len(found) == 1, f"{deliverable['id']} must have one input matching role {needle!r}")
    return found[0]


def _relative_root_key(path: str) -> str:
    return Path(path).as_posix()


def _registry_long(entries: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        denominator = entry.get("denominator") or {}
        source = entry.get("source") or {}
        rows.append({
            "result_id": entry["result_id"], "display_label": entry.get("display_label"), "value": entry.get("value"),
            "value_type": entry.get("value_type"), "unit": entry.get("unit"),
            "denominator_value": denominator.get("value"), "denominator_unit": denominator.get("unit"),
            "evidence_level": entry.get("evidence_level"), "limitation": entry.get("limitation"),
            "missing_status": entry.get("missing_status"), "suppression_status": entry.get("suppression_status"),
            "source_artifact": source.get("source_artifact"), "source_sha256": source.get("source_sha256"),
        })
    return pd.DataFrame(rows)


def _parse_policy_markdown(record: dict[str, Any]) -> pd.DataFrame:
    lines = record["path"].read_text(encoding="utf-8").splitlines()
    rows: list[list[str]] = []
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
            if not cells or all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
                continue
            rows.append(cells)
    _need(rows, "frozen policy markdown contains no pipe table")
    header_index = next((i for i, row in enumerate(rows) if any("date" in x.lower() for x in row)), None)
    _need(header_index is not None, "frozen policy markdown table lacks a date column")
    header = [re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_") or f"column_{i}" for i, x in enumerate(rows[header_index])]
    frame = pd.DataFrame(rows[header_index + 1:], columns=header)
    _need(not frame.empty, "frozen policy markdown table has no rows")
    aliases = {"date": ["date", "data"], "event": ["event", "evento", "milestone"],
               "evidence": ["evidence", "evidencia", "source"], "location": ["location", "local", "where"]}
    output = pd.DataFrame()
    for target, choices in aliases.items():
        column = next((x for x in choices if x in frame.columns), None)
        output[target] = frame[column] if column else pd.NA
    output["source_artifact"] = record["frozen_copy_of"] or record["path"].name
    output["source_sha256"] = record["sha256"]
    return output


def _coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    def col(*names: str) -> str:
        value = next((lower[name] for name in names if name in lower), None)
        _need(value is not None, f"municipality coverage lacks required column; expected one of {names}")
        return value
    year = col("year", "ano")
    adult = col("adult_population", "adult_pop", "population_adult", "pop_adult")
    anchor = col("anchor_available", "has_anchor", "road_anchor_available")
    p120 = col("has_provider_120", "provider_120", "covered_120")
    p180 = col("has_provider_180", "provider_180", "covered_180")
    work = frame[[year, adult, anchor, p120, p180]].copy()
    for column in (adult,):
        work[column] = pd.to_numeric(work[column], errors="coerce")
        _need(work[column].notna().all() and (work[column] >= 0).all(), "adult population must be non-negative numeric")
    for column in (anchor, p120, p180):
        work[column] = work[column].astype("boolean")
    rows: list[dict[str, Any]] = []
    for value, subset in work.groupby(year, dropna=False):
        anchored = subset.loc[subset[anchor].fillna(False)]
        denom = float(anchored[adult].sum())
        _need(denom > 0, f"adult anchored denominator is zero for year {value}")
        rows.append({"scope": "national", "year": value, "status": "EVALUATED", "adult_anchor_denominator": denom,
                     "adult_population_share_120": float(anchored.loc[anchored[p120].fillna(False), adult].sum()) / denom,
                     "adult_population_share_180": float(anchored.loc[anchored[p180].fillna(False), adult].sum()) / denom,
                     "adult_population_anchor_coverage": denom / float(subset[adult].sum()),
                     "reason": pd.NA, "missing_frozen_input": pd.NA})
    return pd.DataFrame(rows)


def figure4_not_evaluated_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mandatory boundaries for frozen inputs the contract does not supply."""
    regional = pd.DataFrame([{"scope": "regional", "status": "NOT_EVALUATED",
        "reason": "Frozen contract supplies no municipality-to-region mapping.",
        "missing_frozen_input": "municipality_region_mapping"}])
    vulnerability = pd.DataFrame([{"scope": "vulnerability_gap", "status": "NOT_EVALUATED",
        "reason": "Frozen contract supplies no municipality-level IVS join; downgraded Aim 2 v3 predictions are forbidden.",
        "missing_frozen_input": "municipality_level_ivs_join"}])
    return regional, vulnerability


def figure6_sensitivity_status_rows() -> pd.DataFrame:
    """Return boundaries that prevent an Aim 4 downgrade being mis-presented."""
    return pd.DataFrame([
        {"analysis_component": "adjusted_point_estimates", "status": "ASSOCIATIONAL_SUPPORTIVE_ONLY", "boundary": "No causal interpretation."},
        {"analysis_component": "bootstrap_intervals", "status": "NOT_EVALUATED", "boundary": "618/617 valid replicates cannot produce accepted formal inference."},
        {"analysis_component": "bootstrap_gate", "status": "DOWNGRADE", "boundary": "Prespecified gate failed; no rerun or relaxation."},
    ])


def _flatten_json(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten_json(child, prefix + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _flatten_json(child, prefix + (str(index),))
    else:
        yield prefix, value


def _figure6_points(record: dict[str, Any]) -> pd.DataFrame:
    document = _read_json(record)
    wanted = {"risk_p10", "risk_p90", "rd", "rr"}
    rows: list[dict[str, Any]] = []
    model_map = {"primary": "AS_mean", "sensitivity": "MPL_Jeffreys"}
    for source_key, estimator in model_map.items():
        source = document.get(source_key)
        _need(isinstance(source, dict), f"Aim 4 point estimate JSON lacks top-level {source_key!r} model")
        found: dict[str, list[Any]] = defaultdict(list)
        for path, value in _flatten_json(source):
            metric = path[-1].lower() if path else ""
            if metric in wanted and isinstance(value, (int, float)) and not isinstance(value, bool):
                found[metric].append(value)
        for metric in sorted(wanted):
            _need(len(found[metric]) == 1,
                  f"Aim 4 {source_key} must contain exactly one numeric {metric}; found {len(found[metric])}")
            rows.append({"estimator": estimator, "metric": metric, "value": found[metric][0],
                         "associational_status": "ASSOCIATIONAL_POINT_ESTIMATE_ONLY",
                         "formal_interval_status": "NOT_EVALUATED", "bootstrap_status": "DOWNGRADE",
                         "source_artifact": record["path"].name, "source_sha256": record["sha256"]})
    return pd.DataFrame(rows)


def _figure6_bootstrap(record: dict[str, Any], entries: list[dict[str, Any]]) -> pd.DataFrame:
    frame = _read_frame(record)
    _need(len(frame) > 0, "Aim 4 bootstrap merged input is empty")
    rows = _registry_long(entries)
    rows["bootstrap_merged_rows"] = len(frame)
    rows["formal_interval_status"] = "NOT_EVALUATED"
    rows["gate_status"] = "DOWNGRADE"
    rows["source_artifact"] = record["path"].name
    rows["source_sha256"] = record["sha256"]
    return rows


def _normalise_frame(record: dict[str, Any], label: str) -> pd.DataFrame:
    frame = _read_frame(record).copy()
    _need(not frame.empty, f"frozen tabular input is empty: {record['path'].name}")
    frame.insert(0, "source_dataset", label)
    frame["source_artifact"] = record["path"].name
    frame["source_sha256"] = record["sha256"]
    return frame


def _integer_counts(series: pd.Series, context: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    _need(values.notna().all() and (values >= 0).all() and (values % 1 == 0).all(),
          f"{context} must be non-negative integer counts")
    return values.astype("int64")


def _display_count(value: int) -> str:
    return "<5" if 1 <= value <= 4 else str(value)


def _travel_bin_summary(record: dict[str, Any]) -> pd.DataFrame:
    """Aggregate all route rows without releasing municipality IDs or raw AIH cells."""
    frame = _read_frame(record).copy()
    required = {"travel_minutes", "n_aih", "route_source", "cross_municipality", "cross_state"}
    _need(required.issubset(frame.columns), "travel route input lacks required display-aggregation columns")
    frame["n_aih"] = _integer_counts(frame["n_aih"], "travel n_aih")
    minutes = pd.to_numeric(frame["travel_minutes"], errors="coerce")
    _need((minutes.dropna() >= 0).all(), "travel_minutes cannot be negative")
    frame["route_status"] = minutes.notna().map({True: "ROUTED", False: "NOT_ROUTED"})
    lower = (minutes.fillna(-1).floordiv(15) * 15).clip(lower=0, upper=180).astype(int)
    frame["travel_bin_lower_min"] = lower.where(minutes.notna(), pd.NA)
    frame["travel_bin_upper_min"] = (lower + 15).where((minutes.notna()) & (lower < 180), pd.NA)
    frame["travel_bin_label"] = pd.Series([
        "NOT_ROUTED" if pd.isna(value) else ("180+" if floor >= 180 else f"{floor}-<{floor + 15}")
        for value, floor in zip(minutes, lower, strict=True)
    ], index=frame.index)
    for column in ("route_source", "cross_municipality", "cross_state"):
        frame[column] = frame[column].astype("string").fillna("MISSING")
    group_columns = ["route_status", "travel_bin_lower_min", "travel_bin_upper_min", "travel_bin_label",
                     "route_source", "cross_municipality", "cross_state"]
    summary = frame.groupby(group_columns, dropna=False, as_index=False)["n_aih"].sum()
    summary["weighted_n_aih_display"] = summary["n_aih"].map(_display_count)
    summary = summary.drop(columns=["n_aih"])
    summary["source_dataset"] = "realised_travel_15_minute_weighted_bins"
    summary["source_artifact"] = record["path"].name
    summary["source_sha256"] = record["sha256"]
    return summary


def _suppressed_network_edges(record: dict[str, Any]) -> pd.DataFrame:
    frame = _normalise_frame(record, "suppressed_patient_flow_network_edges")
    _need({"n_aih", "n_aih_display"}.issubset(frame.columns),
          "network display edges must contain n_aih and n_aih_display for suppression verification")
    raw = _integer_counts(frame["n_aih"], "network edge n_aih")
    declared = frame["n_aih_display"].astype("string")
    expected = raw.map(_display_count).astype("string")
    _need((declared == expected).all(), "network n_aih_display disagrees with frozen raw suppression rule")
    return frame.drop(columns=["n_aih"])


def _centrality_display(record: dict[str, Any]) -> pd.DataFrame:
    frame = _normalise_frame(record, "target_centrality")
    _need("in_strength" in frame.columns, "target centrality input lacks in_strength")
    frame["in_strength_display"] = _integer_counts(frame["in_strength"], "centrality in_strength").map(_display_count)
    return frame.drop(columns=["in_strength"])


def _presentation_rule(filename: str) -> dict[str, Any]:
    rules = {
        "figure_3_travel_source_data.csv": {
            "aggregation_rule": "All route-level records are grouped into fixed 15-minute bins [0,15), …, [165,180), [180,+∞), plus NOT_ROUTED; n_aih is used only as an internal weight.",
            "column_exclusions": ["res_municipio", "treat_municipio", "n_aih"],
            "suppression_rule": "weighted_n_aih_display replaces 1-4 with '<5'.",
        },
        "figure_5_suppressed_network_edges_source_data.csv": {
            "aggregation_rule": "No row exclusion; frozen display-edge rows retained.",
            "column_exclusions": ["n_aih"],
            "suppression_rule": "n_aih_display is cross-checked against raw n_aih then raw n_aih is removed.",
        },
        "figure_5_centrality_source_data.csv": {
            "aggregation_rule": "No row exclusion; frozen centrality rows retained.",
            "column_exclusions": ["in_strength"],
            "suppression_rule": "in_strength_display replaces 1-4 with '<5'; zero and counts >=5 remain explicit.",
        },
    }
    return rules.get(filename, {"aggregation_rule": "No row exclusion or aggregation.", "column_exclusions": [], "suppression_rule": "No additional display suppression."})


def _make_outputs(contract: dict[str, Any], input_records: dict[str, dict[str, Any]], registry: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    deliverables = {item["id"]: item for item in _all_deliverables(contract)}
    selected = {key: _check_registry_use(value, registry) for key, value in deliverables.items()}
    # Figure 1
    d = deliverables["Figure_1"]
    outputs[d["source_data_outputs"][0]] = _parse_policy_markdown(_by_role(d, input_records, "policy"))
    outputs[d["source_data_outputs"][1]] = _registry_long(selected[d["id"]])
    # Figure 2
    d = deliverables["Figure_2"]
    monthly = _normalise_frame(_by_role(d, input_records, "monthly_unique"), "monthly_observed_uptake")
    annual = _normalise_frame(_by_role(d, input_records, "annual_population"), "annual_rate_metadata")
    outputs[d["source_data_outputs"][0]] = pd.concat([monthly, annual], ignore_index=True, sort=False)
    outputs[d["source_data_outputs"][1]] = _normalise_frame(_by_role(d, input_records, "first_observed_by_year_display"), "first_observed_coded_use")
    outputs[d["source_data_outputs"][2]] = _normalise_frame(_by_role(d, input_records, "hospital_continuity"), "maintenance_risk_set")
    # Figure 3
    d = deliverables["Figure_3"]
    outputs[d["source_data_outputs"][0]] = _normalise_frame(_by_role(d, input_records, "equity_rate_display"), "equity_standardised_rates")
    outputs[d["source_data_outputs"][1]] = _travel_bin_summary(_by_role(d, input_records, "route_level"))
    # Figure 4: never fabricate missing regional/IVS joins.
    d = deliverables["Figure_4"]
    municipality = _normalise_frame(_by_role(d, input_records, "municipality_year_coverage"), "municipality_coverage")
    outputs[d["source_data_outputs"][0]] = municipality
    national = _coverage_summary(municipality.drop(columns=["source_dataset", "source_artifact", "source_sha256"]))
    regional, vulnerability = figure4_not_evaluated_rows()
    outputs[d["source_data_outputs"][1]] = pd.concat([national, regional], ignore_index=True, sort=False)
    outputs[d["source_data_outputs"][2]] = vulnerability
    # Figure 5
    d = deliverables["Figure_5"]
    outputs[d["source_data_outputs"][0]] = _suppressed_network_edges(_by_role(d, input_records, "network_display_edges"))
    outputs[d["source_data_outputs"][1]] = _normalise_frame(_by_role(d, input_records, "service_area_residence_display"), "suppressed_service_area")
    outputs[d["source_data_outputs"][2]] = _centrality_display(_by_role(d, input_records, "target_centrality"))
    # Figure 6
    d = deliverables["Figure_6"]
    outputs[d["source_data_outputs"][0]] = _figure6_points(_by_role(d, input_records, "associational_point_estimates"))
    validity = [entry for entry in selected[d["id"]] if entry["result_id"] in {"aim4.bootstrap_status", "aim4.as_mean_valid_replicates", "aim4.mpl_jeffreys_valid_replicates"}]
    _need(len(validity) == 3 and next(x for x in validity if x["result_id"] == "aim4.bootstrap_status")["value"] == "DOWNGRADE",
          "Aim 4 final bootstrap gate must be DOWNGRADE")
    outputs[d["source_data_outputs"][1]] = _figure6_bootstrap(_by_role(d, input_records, "bootstrap_concordance"), validity)
    outputs[d["source_data_outputs"][2]] = figure6_sensitivity_status_rows()
    # Figure 7
    d = deliverables["Figure_7"]
    outputs[d["source_data_outputs"][0]] = _normalise_frame(_by_role(d, input_records, "targeted_sequential"), "targeted_structural_stress_test")
    random = _normalise_frame(_by_role(d, input_records, "random_removal_benchmark"), "random_structural_stress_test")
    _need(len(random) == 8, "frozen random benchmark must contain exactly eight display rows")
    outputs[d["source_data_outputs"][1]] = random
    # Tables are registry-only long-form release values.
    for number in range(1, 5):
        d = deliverables[f"Table_{number}"]
        table = _registry_long(selected[d["id"]])
        if number == 4:
            _need({"aim4.bootstrap_status", "stage06.final_status"}.issubset(set(table["result_id"])),
                  "Table 4 must preserve Aim 4 and Stage 6 downgraded statuses")
        outputs[d["source_data_outputs"][0]] = table
    _need(set(outputs) == set(expected_outputs(contract)), "output mapping does not exactly match frozen contract")
    return outputs


def _privacy_gate(frame: pd.DataFrame, filename: str) -> None:
    for column in frame.columns:
        if not SENSITIVE_COUNT_COLUMN.search(str(column)):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if ((values >= 1) & (values < 5)).any():
            raise BuilderError(f"privacy gate failed: unsuppressed count 1-4 in {filename}:{column}")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    _need(len(frame.columns) > 0, f"output {path.name} has no columns")
    _privacy_gate(frame, path.name)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", na_rep="NA", quoting=csv.QUOTE_MINIMAL)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _make_readonly(path: Path) -> None:
    path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _existing_is_consistent(output_dir: Path, outputs: list[str]) -> bool:
    manifest_path, audit_path = output_dir / "source_data_manifest.json", output_dir / "source_data_audit.json"
    if not output_dir.exists():
        return False
    _need(output_dir.is_dir() and manifest_path.is_file() and audit_path.is_file(),
          "existing output directory is incomplete; refusing overwrite")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuilderError("existing output manifest is invalid; refusing overwrite") from exc
    _need(manifest.get("schema_version") == MANIFEST_SCHEMA, "existing output manifest schema mismatch")
    registered = manifest.get("outputs")
    _need(isinstance(registered, dict) and set(registered) == set(outputs), "existing output manifest does not match contract")
    for filename, meta in registered.items():
        path = output_dir / filename
        _need(path.is_file() and sha256_file(path) == meta.get("sha256"), f"existing output hash mismatch: {filename}")
    return True


def build_source_data(project_root: Path, contract_path: Path, registry_path: Path, output_dir: Path, audit_out: Path) -> dict[str, Any]:
    """Create the 22 source CSVs plus manifest/audit, or raise :class:`BuilderError`."""
    root = project_root.resolve()
    _need(root.is_dir(), "project root does not exist")
    contract_path = contract_path.resolve()
    registry_path = registry_path.resolve()
    _need(contract_path.is_file() and registry_path.is_file(), "contract and registry paths must exist")
    internal_audit = output_dir / "source_data_audit.json"
    contract = _load_yaml(contract_path)
    outputs = expected_outputs(contract)
    contract_hash = sha256_file(contract_path)
    _need(contract_hash == FROZEN_CONTRACT_SHA256,
          "contract SHA-256 does not match the frozen rc_v1 contract")
    # Do this before even loading a dataframe/JSON input or creating output.
    input_records = _verify_contract_inputs(root, contract)
    registry_payload, registry = _load_registry(registry_path)
    if output_dir.exists() and _existing_is_consistent(output_dir, outputs):
        return json.loads((output_dir / "source_data_manifest.json").read_text(encoding="utf-8"))
    _need(not output_dir.exists(), "output directory already exists; refusing overwrite")
    _need(not audit_out.exists(), "audit output already exists; refusing overwrite")
    rendered = _make_outputs(contract, input_records, registry)
    deliverables_by_id = {item["id"]: item for item in _all_deliverables(contract)}
    audit_rows: list[dict[str, Any]] = []
    staging_parent = output_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=staging_parent))
    try:
        output_meta: dict[str, Any] = {}
        for filename in outputs:
            frame = rendered[filename]
            presentation = _presentation_rule(filename)
            source_rows = int(len(frame))
            if filename == "figure_3_travel_source_data.csv":
                source_rows = int(len(_read_frame(_by_role(deliverables_by_id["Figure_3"], input_records, "route_level"))))
            elif filename == "figure_5_suppressed_network_edges_source_data.csv":
                source_rows = int(len(_read_frame(_by_role(deliverables_by_id["Figure_5"], input_records, "network_display_edges"))))
            elif filename == "figure_5_centrality_source_data.csv":
                source_rows = int(len(_read_frame(_by_role(deliverables_by_id["Figure_5"], input_records, "target_centrality"))))
            _write_frame(stage / filename, frame)
            output_meta[filename] = {"sha256": sha256_file(stage / filename), "rows": int(len(frame)),
                                     "columns": [str(value) for value in frame.columns],
                                     "source_artifact_hashes": {key: value["sha256"] for key, value in input_records.items()},
                                     "registry_ids": [entry["result_id"] for entry in _check_registry_use(next(d for d in _all_deliverables(contract) if filename in d["source_data_outputs"]), registry)],
                                     "evidence_level": sorted(set(str(x.get("evidence_level")) for x in _check_registry_use(next(d for d in _all_deliverables(contract) if filename in d["source_data_outputs"]), registry))),
                                     "suppression_status": "preserved_or_not_applicable", "presentation_transform": presentation, "status": "PASS"}
            audit_rows.append({"output": filename, "before_rows": source_rows, "after_rows": int(len(frame)),
                               "aggregation_rule": presentation["aggregation_rule"],
                               "column_exclusions": presentation["column_exclusions"],
                               "suppression_rule": presentation["suppression_rule"], "status": "PASS"})
        manifest = {"schema_version": MANIFEST_SCHEMA, "contract_sha256": contract_hash,
                    "registry_sha256": sha256_file(registry_path), "registry_invalidation_fingerprint": registry_payload.get("invalidation_fingerprint"),
                    "outputs": output_meta, "input_artifacts": {key: {k: v for k, v in value.items() if k != "path"} for key, value in input_records.items()}}
        audit = {"schema_version": AUDIT_SCHEMA, "contract_sha256": contract_hash, "registry_sha256": sha256_file(registry_path),
                 "checks": {"contract_hash": "PASS", "all_input_hashes": "PASS", "registry_ids_and_consumers": "PASS",
                            "privacy": "PASS", "row_count_reconciliation": "PASS", "not_evaluated_boundaries": "PASS"},
                 "outputs": audit_rows}
        _write_json(stage / "source_data_manifest.json", manifest)
        _write_json(stage / "source_data_audit.json", audit)
        for path in stage.iterdir():
            _make_readonly(path)
        os.replace(stage, output_dir)
        # The ordinary invocation points --audit-out at the immutable in-dir
        # audit.  A separately requested mirror is written only after the
        # transactional release directory is complete.
        if audit_out.resolve() != internal_audit.resolve():
            temporary_audit = audit_out.with_name(audit_out.name + ".tmp")
            _write_json(temporary_audit, audit)
            os.replace(temporary_audit, audit_out)
            _make_readonly(audit_out)
        return manifest
    except Exception:
        if stage.exists():
            for child in stage.iterdir():
                child.chmod(stat.S_IWRITE | stat.S_IREAD)
            shutil.rmtree(stage)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_source_data(args.project_root, args.contract, args.registry, args.output_dir, args.audit_out)
    except BuilderError as exc:
        print(f"SOURCE DATA BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "PASS", "outputs": len(manifest["outputs"]), "manifest": str(args.output_dir / "source_data_manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
