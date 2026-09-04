"""Governance and frozen-binding tests for the rc_v1_1 source-data release."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "stage07_build_figure_table_inputs_rc_v1_1.py"
OLD_CONTRACT = ROOT / "config" / "stage07_figures_tables_contract_rc_v1.yaml"
CONTRACT = ROOT / "config" / "stage07_figures_tables_contract_rc_v1_1.yaml"
REGISTRY_SPEC = ROOT / "config" / "stage07_result_registry_spec_rc_v1.json"
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("stage07_rc_v1_1", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_bindings_and_public_contract_are_unchanged() -> None:
    assert _sha(OLD_CONTRACT) == "485bbd56029b71eac618f39c5ef94fe5ee20d8a46c62aaa5a81f279ee6667991"
    assert _sha(CONTRACT) == module.FROZEN_CONTRACT_SHA256
    module._configure_base()


def test_contract_is_frozen_and_has_complete_release_shape() -> None:
    module._configure_base()
    contract = module.base._load_yaml(CONTRACT)
    assert contract["status"] == "FROZEN_CONTRACT"
    assert contract["release_decision"]["registry_permissions_unchanged"] is True
    assert len(contract["figures"]) == 7
    assert len(contract["tables"]) == 4
    assert len(module.base.expected_outputs(contract)) == 22


def test_revision_removes_exactly_the_six_unauthorised_figure_consumers() -> None:
    old = yaml.safe_load(OLD_CONTRACT.read_text(encoding="utf-8"))
    new = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    old_d = {item["id"]: item for item in old["figures"] + old["tables"]}
    new_d = {item["id"]: item for item in new["figures"] + new["tables"]}
    removed = {
        (deliverable_id, result_id)
        for deliverable_id, deliverable in old_d.items()
        for result_id in deliverable["registry_result_ids"]
        if result_id not in new_d[deliverable_id]["registry_result_ids"]
    }
    assert removed == {
        ("Figure_4", "aim2.primary_family_status"),
        ("Figure_5", "potential_access.complete_matrix_rows"),
        ("Figure_5", "potential_access.reachable_rows"),
        ("Figure_5", "service_area.ivs_selected_municipalities"),
        ("Figure_7", "potential_access.complete_matrix_rows"),
        ("Figure_7", "potential_access.reachable_rows"),
    }
    assert not {
        (deliverable_id, result_id)
        for deliverable_id, deliverable in new_d.items()
        for result_id in deliverable["registry_result_ids"]
        if result_id not in old_d[deliverable_id]["registry_result_ids"]
    }


def test_all_35_registry_ids_remain_used_with_authorised_consumers() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_SPEC.read_text(encoding="utf-8"))
    index = {entry["result_id"]: entry for entry in registry["entries"]}
    used: set[str] = set()
    for deliverable in contract["figures"] + contract["tables"]:
        consumer = "figure" if deliverable["id"].startswith("Figure_") else "table"
        for result_id in deliverable["registry_result_ids"]:
            assert result_id in index
            assert consumer in index[result_id]["allowed_consumers"]
            used.add(result_id)
    assert used == set(index)


def test_every_builder_role_selector_has_exactly_one_contract_input() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    required = {
        "Figure_1": ["policy"],
        "Figure_2": ["monthly_unique", "annual_population", "first_observed_by_year_display", "hospital_continuity"],
        "Figure_3": ["equity_rate_display", "route_level"],
        "Figure_4": ["municipality_year_coverage"],
        "Figure_5": ["network_display_edges", "service_area_residence_display", "target_centrality"],
        "Figure_6": ["associational_point_estimates", "bootstrap_concordance"],
        "Figure_7": ["targeted_sequential", "random_removal_benchmark"],
    }
    figures = {item["id"]: item for item in contract["figures"]}
    for figure_id, needles in required.items():
        roles = [str(item.get("role", "")).lower() for item in figures[figure_id]["inputs"]]
        for needle in needles:
            assert sum(needle.lower() in role for role in roles) == 1, (figure_id, needle, roles)


def test_no_python_plotting_backend_is_imported() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    for forbidden in ("matplotlib", "seaborn", "plotly", "pillow", "imageio"):
        assert forbidden not in source


def test_frozen_chinese_policy_header_is_parsed_without_translation() -> None:
    markdown = """# evidence\n\n| 日期 | 事件 | 官方证据 | 证据位置 |\n|---|---|---|---|\n| 2019-07-04 | 推荐 | 官方报告 | report.pdf |\n| 2021-01 | 首次编码窗口 | SIGTAP | archive.zip |\n"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.md"
        path.write_text(markdown, encoding="utf-8")
        frame = module._parse_policy_markdown_rc_v1_1(
            {"path": path, "frozen_copy_of": "reports/policy_evidence_timeline.md", "sha256": _sha(path)}
        )
    assert list(frame.columns) == ["date", "event", "evidence", "location", "source_artifact", "source_sha256"]
    assert frame["date"].tolist() == ["2019-07-04", "2021-01"]
    assert frame["event"].tolist() == ["推荐", "首次编码窗口"]
    assert frame["source_artifact"].nunique() == 1


def test_boolean_event_indicators_are_not_misclassified_as_small_counts() -> None:
    indicators = pd.DataFrame({"any_cessation_event": [False, True], "any_recovery_event": [True, False]})
    module._privacy_gate_rc_v1_1(indicators, "figure_2_maintenance_source_data.csv")
    unsafe_counts = pd.DataFrame({"n_events": [0, 1, 5]})
    try:
        module._privacy_gate_rc_v1_1(unsafe_counts, "unsafe.csv")
    except module.BuilderError as exc:
        assert "unsuppressed count 1-4" in str(exc)
    else:
        raise AssertionError("numeric event counts of 1-4 must remain blocked")
