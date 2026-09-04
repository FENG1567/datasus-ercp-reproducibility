"""Fail-closed regression tests for the Stage 7 figure/table source builder."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "stage07_build_figure_table_inputs_rc_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage07_inputs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_minimal_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the smallest contract that reaches input verification."""
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": "stage07_result_registry_v1", "entries": [{
        "result_id": "cohort.a_unique_aih", "display_label": "A", "value": 1,
        "value_type": "numeric", "unit": "AIHs", "evidence_level": "descriptive",
        "denominator": None, "suppression_status": "not_suppressed",
        "missing_status": "not_missing", "allowed_consumers": ["figure", "table"],
        "limitation": "fixture", "source": {"source_artifact": "source.csv", "source_sha256": _sha(source)},
    }]}), encoding="utf-8")
    contract = tmp_path / "contract.yaml"
    output_sets = [
        ["figure_1_policy_timeline_source_data.csv", "figure_1_cohort_funnel_source_data.csv"],
        ["figure_2_monthly_uptake_source_data.csv", "figure_2_first_observed_source_data.csv", "figure_2_maintenance_source_data.csv"],
        ["figure_3_equity_source_data.csv", "figure_3_travel_source_data.csv"],
        ["figure_4_municipality_coverage_source_data.csv", "figure_4_national_regional_coverage_source_data.csv", "figure_4_vulnerability_gap_source_data.csv"],
        ["figure_5_suppressed_network_edges_source_data.csv", "figure_5_service_area_source_data.csv", "figure_5_centrality_source_data.csv"],
        ["figure_6_adjusted_point_estimates_source_data.csv", "figure_6_bootstrap_validity_source_data.csv", "figure_6_sensitivity_status_source_data.csv"],
        ["figure_7_targeted_removal_source_data.csv", "figure_7_random_benchmark_source_data.csv"],
    ]
    payload = {
        "schema_version": "stage07_figures_tables_contract_rc_v1",
        "candidate_version": "rc_v1", "status": "FROZEN_CONTRACT",
        "shared_inputs": [], "figures": [
            {"id": f"Figure_{i + 1}", "registry_result_ids": ["cohort.a_unique_aih"],
             "inputs": ([{"path": "source.csv", "sha256": _sha(source), "role": "fixture"}] if i == 0 else []),
             "source_data_outputs": names}
            for i, names in enumerate(output_sets)
        ], "tables": [
            {"id": f"Table_{i}", "registry_result_ids": ["cohort.a_unique_aih"], "inputs": [],
             "source_data_outputs": [f"table_{i}_source_data.csv"]}
            for i in range(1, 5)
        ],
    }
    contract.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return tmp_path, contract, registry


def test_input_hash_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RED checkpoint: a declared frozen input hash must block all output."""
    module = _load_module()
    root, contract, registry = _write_minimal_fixture(tmp_path)
    monkeypatch.setattr(module, "FROZEN_CONTRACT_SHA256", _sha(contract))
    source = root / "source.csv"
    source.write_text("value\nTAMPERED\n", encoding="utf-8")
    with pytest.raises(module.BuilderError, match="SHA-256 mismatch"):
        module.build_source_data(root, contract, registry, root / "out", root / "audit.json")
    assert not (root / "out").exists()


def test_missing_registry_id_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A contract identifier absent from the immutable registry is blocking."""
    module = _load_module()
    root, contract, registry = _write_minimal_fixture(tmp_path)
    payload = yaml.safe_load(contract.read_text(encoding="utf-8"))
    payload["figures"][0]["registry_result_ids"] = ["not.in.registry"]
    contract.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(module, "FROZEN_CONTRACT_SHA256", _sha(contract))
    with pytest.raises(module.BuilderError, match="missing registry result_id"):
        module.build_source_data(root, contract, registry, root / "out", root / "audit.json")
    assert not (root / "out").exists()


def test_contract_has_machine_locked_22_outputs() -> None:
    module = _load_module()
    contract = yaml.safe_load((ROOT / "config" / "stage07_figures_tables_contract_rc_v1.yaml").read_text(encoding="utf-8"))
    assert len(module.expected_outputs(contract)) == 22


def test_figure4_and_aim4_downgrade_boundaries_are_explicit() -> None:
    module = _load_module()
    regional, vulnerability = module.figure4_not_evaluated_rows()
    assert regional.iloc[0]["status"] == "NOT_EVALUATED"
    assert vulnerability.iloc[0]["missing_frozen_input"] == "municipality_level_ivs_join"
    sensitivity = module.figure6_sensitivity_status_rows()
    assert {"DOWNGRADE", "NOT_EVALUATED"}.issubset(set(sensitivity["status"]))
    assert not any("confidence interval" in value.lower() for value in sensitivity["boundary"])


def _parquet_record(tmp_path: Path, name: str, frame: pd.DataFrame) -> dict[str, object]:
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return {"path": path, "sha256": _sha(path)}


def test_real_aim4_primary_and_sensitivity_shape_is_explicitly_mapped(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "point.json"
    path.write_text(json.dumps({
        "primary": {"risk_p10": 0.01, "risk_p90": 0.03, "rd": 0.02, "rr": 3.0},
        "sensitivity": {"risk_p10": 0.02, "risk_p90": 0.04, "rd": 0.02, "rr": 2.0},
    }), encoding="utf-8")
    points = module._figure6_points({"path": path, "sha256": _sha(path)})
    assert set(points["estimator"]) == {"AS_mean", "MPL_Jeffreys"}
    assert set(points["metric"]) == {"risk_p10", "risk_p90", "rd", "rr"}


def test_travel_network_and_centrality_presentation_remove_raw_small_cells(tmp_path: Path) -> None:
    module = _load_module()
    travel = _parquet_record(tmp_path, "travel.parquet", pd.DataFrame({
        "res_municipio": ["110001", "110002", "110003"], "treat_municipio": ["330001", "330002", "330003"],
        "travel_minutes": [5.0, 17.0, None], "n_aih": [1, 4, 5],
        "route_source": ["cached", "cached", "routed"], "cross_municipality": [True, True, False],
        "cross_state": [True, False, False],
    }))
    summary = module._travel_bin_summary(travel)
    assert "n_aih" not in summary and not {"res_municipio", "treat_municipio"}.intersection(summary.columns)
    assert set(summary["weighted_n_aih_display"]) == {"<5", "5"}
    network = _parquet_record(tmp_path, "network.parquet", pd.DataFrame({"edge": ["a", "b"], "n_aih": [3, 5], "n_aih_display": ["<5", "5"]}))
    edges = module._suppressed_network_edges(network)
    assert "n_aih" not in edges and edges["n_aih_display"].tolist() == ["<5", "5"]
    inconsistent = _parquet_record(tmp_path, "inconsistent.parquet", pd.DataFrame({"edge": ["a"], "n_aih": [3], "n_aih_display": ["3"]}))
    with pytest.raises(module.BuilderError, match="disagrees"):
        module._suppressed_network_edges(inconsistent)
    centrality = _parquet_record(tmp_path, "centrality.parquet", pd.DataFrame({"node": ["a", "b", "c"], "in_strength": [0, 2, 6], "betweenness": [0.0, 0.2, 0.4]}))
    display = module._centrality_display(centrality)
    assert "in_strength" not in display and display["in_strength_display"].tolist() == ["0", "<5", "6"]


def test_sensitive_count_gate_blocks_raw_aih_but_exempts_structural_counts(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(module.BuilderError, match="privacy gate failed"):
        module._write_frame(tmp_path / "blocked.csv", pd.DataFrame({"n_aih": [3]}))
    module._write_frame(tmp_path / "structural.csv", pd.DataFrame({"n_provider_municipalities": [2], "n_removed": [3], "random_replicates": [1]}))
    assert (tmp_path / "structural.csv").is_file()


def test_small_real_shape_fixture_builds_all_22_privacy_safe_frames(tmp_path: Path) -> None:
    """Exercise the complete fixed mapping without needing production data."""
    module = _load_module()
    records: dict[str, dict[str, object]] = {}
    def add_frame(name: str, frame: pd.DataFrame) -> str:
        path = tmp_path / name
        if name.endswith(".csv"):
            frame.to_csv(path, index=False)
        elif name.endswith(".csv.gz"):
            frame.to_csv(path, index=False, compression="gzip")
        else:
            frame.to_parquet(path, index=False)
        records[name] = {"path": path, "sha256": _sha(path), "frozen_copy_of": None}
        return name
    policy = tmp_path / "policy.md"
    policy.write_text("| date | event | evidence | location |\n|---|---|---|---|\n|2021-01|window|audit|Brazil|\n", encoding="utf-8")
    records["policy.md"] = {"path": policy, "sha256": _sha(policy), "frozen_copy_of": None}
    basic = add_frame("basic.parquet", pd.DataFrame({"value": [5]}))
    equity = add_frame("equity.csv", pd.DataFrame({"stratum": ["q1"], "rate": [1.0]}))
    travel = add_frame("travel.parquet", pd.DataFrame({"res_municipio": ["1"], "treat_municipio": ["2"], "travel_minutes": [7.0], "n_aih": [3], "route_source": ["cache"], "cross_municipality": [True], "cross_state": [False]}))
    coverage = add_frame("coverage.parquet", pd.DataFrame({"year": [2021], "adult_population": [100], "anchor_available": [True], "has_provider_120": [True], "has_provider_180": [True]}))
    edges = add_frame("edges.csv", pd.DataFrame({"edge": ["a"], "n_aih": [3], "n_aih_display": ["<5"]}))
    service = add_frame("service.csv", pd.DataFrame({"area": ["a"], "n_aih_display": ["<5"]}))
    centrality = add_frame("centrality.parquet", pd.DataFrame({"node": ["a"], "in_strength": [2], "betweenness": [0.1]}))
    point = tmp_path / "point.json"
    point.write_text(json.dumps({"primary": {"risk_p10": .01, "risk_p90": .02, "rd": .01, "rr": 2}, "sensitivity": {"risk_p10": .01, "risk_p90": .03, "rd": .02, "rr": 3}}), encoding="utf-8")
    records["point.json"] = {"path": point, "sha256": _sha(point), "frozen_copy_of": None}
    bootstrap = add_frame("bootstrap.csv.gz", pd.DataFrame({"replicate": [1]}))
    sequential = add_frame("sequential.csv", pd.DataFrame({"n_removed": [1]}))
    random = add_frame("random.parquet", pd.DataFrame({"scenario": list(range(8)), "n_removed": [1] * 8}))
    outputs = [
        ["figure_1_policy_timeline_source_data.csv", "figure_1_cohort_funnel_source_data.csv"],
        ["figure_2_monthly_uptake_source_data.csv", "figure_2_first_observed_source_data.csv", "figure_2_maintenance_source_data.csv"],
        ["figure_3_equity_source_data.csv", "figure_3_travel_source_data.csv"],
        ["figure_4_municipality_coverage_source_data.csv", "figure_4_national_regional_coverage_source_data.csv", "figure_4_vulnerability_gap_source_data.csv"],
        ["figure_5_suppressed_network_edges_source_data.csv", "figure_5_service_area_source_data.csv", "figure_5_centrality_source_data.csv"],
        ["figure_6_adjusted_point_estimates_source_data.csv", "figure_6_bootstrap_validity_source_data.csv", "figure_6_sensitivity_status_source_data.csv"],
        ["figure_7_targeted_removal_source_data.csv", "figure_7_random_benchmark_source_data.csv"],
    ]
    role_paths = [
        [("policy", "policy.md")],
        [("monthly_unique", basic), ("annual_population", basic), ("first_observed_by_year_display", basic), ("hospital_continuity", basic)],
        [("equity_rate_display", equity), ("route_level", travel)],
        [("municipality_year_coverage", coverage)],
        [("network_display_edges", edges), ("service_area_residence_display", service), ("target_centrality", centrality)],
        [("associational_point_estimates", "point.json"), ("bootstrap_concordance", bootstrap)],
        [("targeted_sequential", sequential), ("random_removal_benchmark", random)],
    ]
    figures = [{"id": f"Figure_{i+1}", "registry_result_ids": ["cohort.a_unique_aih"] if i != 5 else ["aim4.bootstrap_status", "aim4.as_mean_valid_replicates", "aim4.mpl_jeffreys_valid_replicates"], "inputs": [{"role": role, "path": path} for role, path in role_paths[i]], "source_data_outputs": output} for i, output in enumerate(outputs)]
    tables = [{"id": f"Table_{i}", "registry_result_ids": (["aim4.bootstrap_status", "stage06.final_status"] if i == 4 else ["cohort.a_unique_aih"]), "inputs": [], "source_data_outputs": [f"table_{i}_source_data.csv"]} for i in range(1, 5)]
    contract = {"figures": figures, "tables": tables}
    def entry(result_id: str, value: object = 5) -> dict[str, object]:
        return {"result_id": result_id, "display_label": result_id, "value": value, "value_type": "string" if isinstance(value, str) else "numeric", "unit": "status" if isinstance(value, str) else "count", "evidence_level": "associational", "denominator": None, "suppression_status": "not_applicable", "missing_status": "not_applicable", "allowed_consumers": ["figure", "table"], "limitation": "fixture", "source": {"source_artifact": "fixture", "source_sha256": "0" * 64}}
    registry = {"cohort.a_unique_aih": entry("cohort.a_unique_aih"), "aim4.bootstrap_status": entry("aim4.bootstrap_status", "DOWNGRADE"), "aim4.as_mean_valid_replicates": entry("aim4.as_mean_valid_replicates", 618), "aim4.mpl_jeffreys_valid_replicates": entry("aim4.mpl_jeffreys_valid_replicates", 617), "stage06.final_status": entry("stage06.final_status", "COMPLETE_WITH_QUASI_CAUSAL_DOWNGRADE")}
    rendered = module._make_outputs(contract, records, registry)
    assert len(rendered) == 22
    assert "n_aih" not in rendered["figure_3_travel_source_data.csv"]
    assert "n_aih" not in rendered["figure_5_suppressed_network_edges_source_data.csv"]
    assert "in_strength" not in rendered["figure_5_centrality_source_data.csv"]
