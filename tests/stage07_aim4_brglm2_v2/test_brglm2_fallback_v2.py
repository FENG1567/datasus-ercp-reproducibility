from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


PREPARE = load("stage07_prepare_aim4_brglm2_v2")
FINALIZE = load("stage07_finalize_aim4_brglm2_v2")


def synthetic_analytic(n: int = 720) -> pd.DataFrame:
    rng = np.random.default_rng(20260830)
    index = np.arange(n)
    volume = 10.0 + (index % 120)
    probability = 1.0 / (1.0 + np.exp(-(-4.0 + 0.008 * volume + 0.01 * (index % 30))))
    return pd.DataFrame({
        "analysis_row_id": index,
        "death_valid": True,
        "trailing12_complete": True,
        "in_hospital_death": rng.binomial(1, probability),
        "cnes7": [f"{1000000 + ((value // 20) % 36):07d}" for value in index],
        "state_provider": np.where((index // 20) % 2 == 0, "35", "33"),
        "calendar_month": np.array(["202201", "202202", "202203", "202204", "202205", "202206"])[index % 6],
        "age_years": 20 + (index % 70),
        "sex_category": rng.choice(["1", "2"], n),
        "race_category": rng.choice(["1", "2", "MISSING"], n),
        "emergency_admission": rng.choice(["ELECTIVE", "URGENT_OR_NON_ELECTIVE"], n),
        "diagnostic_stratum": rng.choice(["K80.3", "K80.4", "K80.5"], n),
        "hospital_type": rng.choice(["general", "specialty"], n),
        "endoscopy_capability": rng.choice(["yes", "no"], n),
        "comorbidity_burden": rng.integers(0, 4, n),
        "beds_sus": 30 + rng.integers(0, 10, n),
        "icu_beds": 2 + rng.integers(0, 5, n),
        "ivs_context": 0.1 + rng.integers(0, 7, n) / 20,
        "ans_context": 0.2 + rng.integers(0, 9, n) / 20,
        "trailing12_a_unique_aih": volume,
    })


def point_payload() -> dict:
    return {
        "schema_version": "aim4_brglm2_v2",
        "evidence": "associational/supportive",
        "input_sha256": "a" * 64,
        "design_sha256": "b" * 64,
        "detectseparation_audit": {"status": "PASS", "complete_or_quasi_complete": True},
        "primary": {"status": "valid", "risk_p10": 0.02, "risk_p90": 0.018, "rd": -0.002, "rr": 0.9},
        "sensitivity": {"status": "valid", "risk_p10": 0.021, "risk_p90": 0.019, "rd": -0.002, "rr": 0.905},
    }


def write_replicates(folder: Path, count: int = 2000) -> None:
    folder.mkdir(parents=True)
    for replicate_id in range(1, count + 1):
        rd = -0.002 + (replicate_id % 17 - 8) * 0.00001
        payload = {
            "replicate_id": replicate_id, "status": "valid", "failure_reason": None,
            "as_mean_status": "valid", "as_mean_failure_reason": None,
            "mpl_jeffreys_status": "valid", "mpl_jeffreys_failure_reason": None,
            "as_mean_risk_p10": 0.02, "as_mean_risk_p90": 0.02 + rd, "as_mean_rd": rd, "as_mean_rr": (0.02 + rd) / 0.02,
            "mpl_jeffreys_risk_p10": 0.021, "mpl_jeffreys_risk_p90": 0.021 + rd, "mpl_jeffreys_rd": rd, "mpl_jeffreys_rr": (0.021 + rd) / 0.021,
            "runtime_seconds": 0.01, "state_hospital_counts": "33:18;35:18",
            "zero_design_columns": "", "calendar_month_zero_columns": "", "hospital_type_zero_columns": "",
            "evidence": "associational/supportive",
            "input_sha256": "a" * 64, "design_sha256": "b" * 64,
        }
        (folder / f"replicate_{replicate_id:04d}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_frozen_explicit_design_and_manifest(tmp_path: Path) -> None:
    analytic = tmp_path / "analytic.parquet"
    frame = synthetic_analytic()
    frame.loc[0, "death_valid"] = False
    frame.loc[1, "trailing12_complete"] = False
    frame.loc[2, "ivs_context"] = np.nan
    frame.to_parquet(analytic)
    output = tmp_path / "prepared"
    completed = subprocess.run([sys.executable, str(ROOT / "src" / "stage07_prepare_aim4_brglm2_v2.py"), "--analytic", str(analytic), "--output-dir", str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    config = json.loads((output / "aim4_brglm2_design_v2.json").read_text(encoding="utf-8"))
    with gzip.open(output / "aim4_brglm2_input_v2.csv.gz", "rt", encoding="utf-8") as handle:
        prepared = pd.read_csv(handle)
    assert len(prepared) == len(frame) - 3
    assert {"y", "cnes7", "state_provider"}.issubset(prepared.columns)
    assert len(config["design_names"]) == len(config["design_columns"])
    assert config["bootstrap"] == {"replicates": 2000, "seed": 20260830, "stratifier": "corrected state_provider", "cluster": "cnes7"}
    assert config["p10"] < config["p90"]
    assert len(config["volume_knots"]) == 4
    manifest = json.loads((output / "aim4_brglm2_prepare_manifest_v2.json").read_text(encoding="utf-8"))
    assert all(len(value) == 64 for value in manifest["outputs"].values())


def test_finalizer_requires_exact_full_set_and_calculates_percentile_ci(tmp_path: Path) -> None:
    point = tmp_path / "point.json"
    point.write_text(json.dumps(point_payload()), encoding="utf-8")
    shards = tmp_path / "replicates"
    write_replicates(shards)
    output = tmp_path / "final"
    result = subprocess.run([sys.executable, str(ROOT / "src" / "stage07_finalize_aim4_brglm2_v2.py"), "--replicate-dir", str(shards), "--point-estimate", str(point), "--output-dir", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    qc = json.loads((output / "aim4_brglm2_final_qc_v2.json").read_text(encoding="utf-8"))
    assert qc["status"] == "PASS"
    assert qc["as_mean_valid_replicates"] == 2000
    assert qc["mpl_jeffreys_valid_replicates"] == 2000
    assert qc["bootstrap_percentile_summary"]["as_mean"]["rd"]["percentile_ci_low"] < qc["bootstrap_percentile_summary"]["as_mean"]["rd"]["percentile_ci_high"]
    assert "mpl_jeffreys" in qc["bootstrap_percentile_summary"]
    assert "Wald" not in json.dumps(qc)
    assert (output / "aim4_brglm2_bootstrap_merged_v2.csv.gz").exists()


def test_finalizer_blocks_missing_or_duplicate_ids(tmp_path: Path) -> None:
    point = tmp_path / "point.json"
    point.write_text(json.dumps(point_payload()), encoding="utf-8")
    shards = tmp_path / "replicates"
    write_replicates(shards, count=2)
    output = tmp_path / "final"
    qc, code = FINALIZE.finalize(shards, point, output)
    assert code == 2
    assert qc["status"] == "BLOCKED"
    assert 3 in qc["missing_replicates"]


def test_r_script_has_frozen_methods_and_optional_parse() -> None:
    script = ROOT / "src" / "stage07_fit_aim4_brglm2_v2.R"
    text = script.read_text(encoding="utf-8")
    for required in ["AS_mean", "MPL_Jeffreys", "L'Ecuyer-CMRG", "replicate_start", "replicate_end", "rank_deficient", "atomic_json", "detectseparation::detect_separation", "semantic_names", "Existing point artifact is not a compatible resume artifact", "colClasses"]:
        assert required in text
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed locally; server integration test is intentionally deferred")
    completed = subprocess.run([rscript, "-e", f"parse(file='{script.as_posix()}')"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_one_isolated_failure_is_not_structural_and_uses_semantic_column_names() -> None:
    failed = pd.DataFrame([{
        "failure_reason": "rank_deficient", "zero_design_columns": "calendar_month[202403]",
        "state_hospital_counts": "33:18;35:18", "calendar_month_zero_columns": "calendar_month[202403]",
        "hospital_type_zero_columns": "",
    }])
    audit = FINALIZE.structural_failure_audit(failed)
    assert not audit["systematic_failure"]
    assert ">=20 replicates" in audit["criterion"]
    assert "calendar_month[202403]" in audit["zero_design_column_counts"]
