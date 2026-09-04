from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read(name: str):
    with (ROOT / "config" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_primary_window_and_thread_ceiling_are_frozen():
    project = read("project.yaml")
    assert project["primary_window"] == {"start": "2021-01", "end": "2025-12"}
    assert project["compute"]["total_thread_ceiling"] == 8


def test_unique_aih_and_double_cohort_are_frozen():
    cohorts = read("cohorts.yaml")
    assert cohorts["unique_aih_key"] == ["competence_month", "SP_CNES", "SP_NAIH"]
    assert cohorts["ercp_procedure_code"] == "0407030255"
    assert set(cohorts["cohorts"]["choledocholithiasis_strict_adult"]["diagnosis_codes"]) == {"K803", "K804", "K805"}


def test_causal_gate_fails_closed():
    gates = read("qc_gates.yaml")
    assert gates["quasi_causal"]["failure_action"].startswith("REMOVE")


def test_all_hospitals_denominator_is_forbidden():
    estimands = read("estimands.yaml")
    assert "all hospitals" in estimands["eligible_hospital_risk_sets"]["forbidden"]

