from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_finalizer():
    path = ROOT / "src" / "stage07_finalize_aim4_brglm2_v3.py"
    spec = importlib.util.spec_from_file_location("stage07_finalize_aim4_brglm2_v3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


FINALIZER = load_finalizer()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_and_prefit(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    prefit = tmp_path / "prefit.json"
    prefit.write_text(json.dumps({"schema_version": "aim4_brglm2_v3"}), encoding="utf-8")
    hashes = {"input_sha256": "a" * 64, "design_sha256": "b" * 64, "prefit_manifest_sha256": digest(prefit)}
    point = tmp_path / "point.json"
    payload = {
        "schema_version": "aim4_brglm2_v3", "evidence": "associational/supportive",
        "formal_bootstrap_started": False, "bootstrap_eligibility": True,
        "primary": {"status": "valid", "rd": -0.001}, "sensitivity": {"status": "valid", "rd": -0.0011},
        "detectseparation_audit": {"status": "PASS"}, **hashes,
    }
    point.write_text(json.dumps(payload), encoding="utf-8")
    hashes["point_sha256"] = digest(point)
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps({
        "schema_version": "aim4_brglm2_v3", "evidence": "associational/supportive",
        "inputs": {"aim4_brglm2_input_v3_scaled.csv.gz": hashes["input_sha256"], "aim4_brglm2_design_v3_scaled.json": hashes["design_sha256"], point.name: hashes["point_sha256"], prefit.name: hashes["prefit_manifest_sha256"]},
        "environment_lock_sha256": "e" * 64,
        "bootstrap_design": {"replicates": 2000, "seed": 20260830},
        "code_sha256": {"stage07_finalize_aim4_brglm2_v3.py": digest(ROOT / "src" / "stage07_finalize_aim4_brglm2_v3.py"), "stage07_bootstrap_aim4_brglm2_v3.R": digest(ROOT / "src" / "stage07_bootstrap_aim4_brglm2_v3.R")},
    }), encoding="utf-8")
    hashes["run_manifest_sha256"] = digest(run_manifest)
    return point, prefit, run_manifest, hashes


def write_replicates(folder: Path, hashes: dict[str, str], count: int = 2000, failed: set[int] | None = None) -> None:
    folder.mkdir(parents=True)
    failed = failed or set()
    for rid in range(1, count + 1):
        valid = rid not in failed
        record = {
            "schema_version": "aim4_brglm2_v3", "evidence": "associational/supportive", "replicate_id": rid,
            "status": "valid" if valid else "failed", "failure_reason": None if valid else "nonconvergence",
            "as_mean_status": "valid" if valid else "failed", "as_mean_failure_reason": None if valid else "nonconvergence",
            "mpl_jeffreys_status": "valid" if valid else "failed", "mpl_jeffreys_failure_reason": None if valid else "nonconvergence",
            "as_mean_risk_p10": 0.02 if valid else None, "as_mean_risk_p90": 0.019 if valid else None,
            "as_mean_rd": -0.001 if valid else None, "as_mean_rr": 0.95 if valid else None,
            "mpl_jeffreys_risk_p10": 0.02 if valid else None, "mpl_jeffreys_risk_p90": 0.0189 if valid else None,
            "mpl_jeffreys_rd": -0.0011 if valid else None, "mpl_jeffreys_rr": 0.945 if valid else None,
            "provider_uf_zero_support": "", "calendar_month_zero_columns": "", "hospital_type_zero_columns": "",
            "zero_design_columns": "", "state_hospital_counts": "33:2;35:2", "runtime_seconds": 0.01, **hashes,
        }
        (folder / f"replicate_{rid:04d}.json").write_text(json.dumps(record), encoding="utf-8")


def test_external_bootstrap_has_frozen_seed_scaling_and_no_point_execution() -> None:
    text = (ROOT / "src" / "stage07_bootstrap_aim4_brglm2_v3.R").read_text(encoding="utf-8")
    for required in ["--bootstrap-only", "L'Ecuyer-CMRG", "MASTER_SEED <- 20260830L", "seed_for_replicate", "nextRNGStream", "AS_mean", "MPL_Jeffreys", "maxit = 500", "slowit = 0.5", "max_step_factor = 6", "volume_basis_p10_scaled", "volume_basis_p90_scaled", "Refusing to overwrite immutable replicate"]:
        assert required in text
    assert "stage07_fit_aim4_brglm2_v3.R" not in text
    assert "--point-only" not in text


def test_bootstrap_static_contract_covers_strata_multiplicity_and_shard_invariance() -> None:
    text = (ROOT / "src" / "stage07_bootstrap_aim4_brglm2_v3.R").read_text(encoding="utf-8")
    for required in ["for (state in sort(unique(raw$state_provider)))", "sample(hospital_ids, size = length(hospital_ids), replace = TRUE)", "multiplicity <- table(drawn)", "weights[raw$state_provider == state & raw$cnes7 == hospital]", "seed_for_replicate(master_seed, replicate_id)", "inputs$X", "inputs$p10", "inputs$p90", "provider_uf_zero_support"]:
        assert required in text
    assert "scale(" not in text and "sd(" not in text and "quantile(" not in text


def test_finalizer_exact_provenance_and_threshold_boundaries(tmp_path: Path) -> None:
    point, prefit, run_manifest, hashes = point_and_prefit(tmp_path)
    replicates = tmp_path / "replicates"
    write_replicates(replicates, hashes)
    qc, code = FINALIZER.finalize(replicates, point, prefit, run_manifest, tmp_path / "out")
    assert code == 0 and qc["status"] == "PASS"
    assert qc["as_mean_valid_replicates"] == 2000
    assert qc["bootstrap_percentile_summary"]["as_mean"]["rd"]["percentile_ci_low"] <= qc["bootstrap_percentile_summary"]["as_mean"]["rd"]["percentile_ci_high"]
    assert FINALIZER.bootstrap_quality(1900, False) == "PASS"
    assert FINALIZER.bootstrap_quality(1899, False) == "WARNING_EXPLORATORY_SUPPORTIVE"
    assert FINALIZER.bootstrap_quality(1800, False) == "WARNING_EXPLORATORY_SUPPORTIVE"
    assert FINALIZER.bootstrap_quality(1799, False) == "DOWNGRADE"
    assert FINALIZER.bootstrap_quality(2000, True) == "DOWNGRADE"


def test_finalizer_fails_closed_on_incomplete_or_duplicate_ids(tmp_path: Path) -> None:
    point, prefit, run_manifest, hashes = point_and_prefit(tmp_path)
    replicates = tmp_path / "replicates"
    write_replicates(replicates, hashes, count=2)
    qc, code = FINALIZER.finalize(replicates, point, prefit, run_manifest, tmp_path / "out")
    assert code == 2 and qc["status"] == "BLOCKED" and 3 in qc["missing_replicates"]


def test_systematic_support_failure_audit_and_runner_contract() -> None:
    failed = pd.DataFrame([{
        "as_mean_failure_reason": "nonconvergence", "provider_uf_zero_support": "35",
        "calendar_month_zero_columns": "calendar_month[202401]", "hospital_type_zero_columns": "hospital_type[specialty]",
    } for _ in range(25)])
    audit = FINALIZER.structural_failure_audit(failed)
    assert audit["systematic_failure"]
    assert audit["repeated_failure_reasons"] == ["nonconvergence"]
    runner = (ROOT / "scripts" / "run_stage07_aim4_brglm2_v3.sh").read_text(encoding="utf-8")
    for required in ["RUN_SESSION=\"stage07_aim4_v3_bootstrap\"", "COVERAGE_SESSION=\"stage07_aim2_coverage\"", "tmux has-session -t \"$COVERAGE_SESSION\"", "curl -fsS http://127.0.0.1:19999/health", "OUTPUT_DIR=\"data_analytic/stage07_rebuild/aim4_v3_stabilized/brglm2\"", "REPLICATE_DIR=\"$OUTPUT_DIR/replicates\"", "starts=(1 251 501 751 1001 1251 1501 1751)", "ends=(250 500 750 1000 1250 1500 1750 2000)", "--bootstrap-only", "aim4_brglm2_bootstrap_run_manifest_v3.json"]:
        assert required in runner
    assert "--point-only" not in runner and "--overwrite" not in runner
    assert 'MASTER_SCRIPT="$PROJECT_ROOT/$LOG_DIR/master.sh"' in runner
    assert 'MASTER_SCRIPT_TMP="$(mktemp "$PROJECT_ROOT/$LOG_DIR/.master.sh.tmp.XXXXXX")"' in runner
    assert "cat <<'MASTER_SCRIPT_BODY'" in runner
    assert 'chmod 700 "$MASTER_SCRIPT_TMP"' in runner
    assert 'mv -f -- "$MASTER_SCRIPT_TMP" "$MASTER_SCRIPT"' in runner
    assert 'tmux new-session -d -s "$RUN_SESSION" "$MASTER_SCRIPT"' in runner
    assert "read -r -d '' INNER_BOOTSTRAP_SCRIPT" not in runner
    assert 'bash -lc "$INNER_BOOTSTRAP_SCRIPT"' not in runner
    assert "if manifest.exists():" in runner and "existing run manifest differs" in runner


def test_support_and_resume_contract_corrects_centered_dummy_and_recycled_state_defects(tmp_path: Path) -> None:
    r_text = (ROOT / "src" / "stage07_bootstrap_aim4_brglm2_v3.R").read_text(encoding="utf-8")
    assert "states <- sort(unique(inputs$raw$state_provider))" in r_text
    assert "states[!state_has_positive_support]" in r_text
    assert "max(value) - min(value) <= tolerance" in r_text
    assert "all(value == 0)" not in r_text
    assert "existing_replicate_matches" in r_text and "run_manifest_sha256" in r_text
    assert "calendar_month_constant_columns" in r_text and "hospital_type_constant_columns" in r_text
    centered_absent_dummy = np.full(12, -0.4472135955)
    assert not np.all(centered_absent_dummy == 0.0)  # the old all-zero rule misses this case
    assert np.max(centered_absent_dummy) - np.min(centered_absent_dummy) <= 1e-12
    point, prefit, run_manifest, hashes = point_and_prefit(tmp_path)
    replicates = tmp_path / "replicates"
    write_replicates(replicates, hashes, count=2000)
    first = json.loads((replicates / "replicate_0001.json").read_text(encoding="utf-8"))
    assert first["run_manifest_sha256"] == hashes["run_manifest_sha256"]
    first["run_manifest_sha256"] = "f" * 64
    (replicates / "replicate_0001.json").write_text(json.dumps(first), encoding="utf-8")
    qc, code = FINALIZER.finalize(replicates, point, prefit, run_manifest, tmp_path / "out")
    assert code == 2 and qc["status"] == "BLOCKED"
    assert any("run_manifest_sha256" in error for error in qc["record_errors"])
    runner = (ROOT / "scripts" / "run_stage07_aim4_brglm2_v3.sh").read_text(encoding="utf-8")
    assert "if manifest.exists():" in runner and "existing run manifest differs" in runner
    assert "record_master_exit" in runner and "trap record_master_exit EXIT" in runner
    assert "Existing replicate provenance mismatch" in r_text and "return(invisible(\"existing_verified\"))" in r_text


def test_r_parse_if_locally_available_and_bash_syntax_if_available() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is unavailable locally; server parse is deferred")
    r = subprocess.run([rscript, "-e", f"parse(file='{(ROOT / 'src' / 'stage07_bootstrap_aim4_brglm2_v3.R').as_posix()}')"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    bash = shutil.which("bash")
    if bash is not None:
        result = subprocess.run([bash, "-n", str(ROOT / "scripts" / "run_stage07_aim4_brglm2_v3.sh")], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr + result.stdout
