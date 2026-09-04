from __future__ import annotations

import gzip
import hashlib
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
PREPARER_PATH = ROOT / "src" / "stage07_prepare_aim4_brglm2_v3.py"
FITTER_PATH = ROOT / "src" / "stage07_fit_aim4_brglm2_v3.R"
AMENDMENT = ROOT / "reports" / "amendments" / "2026-08-30_aim4_numerical_stabilization.md"


def load_preparer():
    spec = importlib.util.spec_from_file_location("stage07_prepare_aim4_brglm2_v3", PREPARER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


PREPARER = load_preparer()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_v2_contract(n: int = 120) -> tuple[pd.DataFrame, dict]:
    """A full-rank, no-effect synthetic input used only for algebraic tests."""
    rng = np.random.default_rng(20260830)
    X = np.column_stack((np.ones(n), rng.normal(size=(n, 95))))
    # Keep volume columns distinct and use deterministic non-effect values.
    X[:, 93:96] = rng.normal(size=(n, 3))
    columns = [f"x_{index:04d}" for index in range(96)]
    raw = pd.DataFrame({
        "analysis_row_id": [f"synthetic-{index:03d}" for index in range(n)],
        "y": np.asarray([index % 2 for index in range(n)], dtype=int),
        "cnes7": [f"{1000000 + index % 12:07d}" for index in range(n)],
        "state_provider": ["33" if index % 2 else "35" for index in range(n)],
    })
    for index, column in enumerate(columns):
        raw[column] = X[:, index]
    config = {
        "schema_version": "aim4_brglm2_v2",
        "design_columns": columns,
        "design_names": ["intercept", *[f"synthetic_{index:02d}" for index in range(1, 96)]],
        "volume_column_indices_zero_based": [93, 94, 95],
        "volume_knots": [12.0, 76.0, 212.0, 600.0],
        "p10": 25.0,
        "p90": 538.0,
        "volume_basis_p10": [1.1, -0.8, 2.1],
        "volume_basis_p90": [1.9, 0.8, 2.9],
        "evidence": "associational/supportive",
    }
    return raw, config


def write_v2(tmp_path: Path, n: int = 120) -> tuple[Path, Path, Path]:
    raw, config = synthetic_v2_contract(n=n)
    config["n_rows"] = n
    input_path = tmp_path / "aim4_brglm2_input_v2.csv.gz"
    with gzip.open(input_path, "wt", encoding="utf-8", newline="") as handle:
        raw.to_csv(handle, index=False, float_format="%.17g")
    design_path = tmp_path / "aim4_brglm2_design_v2.json"
    design_path.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    lock_path = tmp_path / "environment.lock"
    lock_path.write_text("synthetic test environment lock; no model run\n", encoding="utf-8")
    return input_path, design_path, lock_path


def test_scaled_preparer_binds_authorized_inputs_and_preserves_affine_contract(tmp_path: Path) -> None:
    input_path, design_path, lock_path = write_v2(tmp_path, n=30_900)
    output = tmp_path / "v3"
    completed = subprocess.run(
        [sys.executable, str(PREPARER_PATH), "--input-gz", str(input_path), "--design-json", str(design_path),
         "--amendment", str(AMENDMENT), "--environment-lock", str(lock_path), "--output-dir", str(output)],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    manifest = json.loads((output / "aim4_brglm2_prefit_manifest_v3.json").read_text(encoding="utf-8"))
    design = json.loads((output / "aim4_brglm2_design_v3_scaled.json").read_text(encoding="utf-8"))
    transform = pd.read_csv(output / "aim4_brglm2_column_transform_v3.csv")
    with gzip.open(output / "aim4_brglm2_input_v3_scaled.csv.gz", "rt", encoding="utf-8") as handle:
        scaled = pd.read_csv(handle)
    assert manifest["formal_model_run"] is False
    assert manifest["inputs"][AMENDMENT.name] == sha256(AMENDMENT)
    assert manifest["outputs"]["aim4_brglm2_column_transform_v3.csv"] == sha256(output / "aim4_brglm2_column_transform_v3.csv")
    assert design["schema_version"] == "aim4_brglm2_v3"
    assert design["optimization_controls"] == {"epsilon": 1e-8, "max_step_factor": 6, "maxit": 500, "slowit": 0.5}
    assert design["p10"] == 25.0 and design["p90"] == 538.0 and design["volume_knots"] == [12.0, 76.0, 212.0, 600.0]
    assert len(scaled) == 30_900 and len(transform) == 96
    assert np.array_equal(scaled["y"].to_numpy(), np.asarray([index % 2 for index in range(30_900)]))
    assert np.allclose(scaled["x_0000"], 1.0, rtol=0.0, atol=0.0)
    assert manifest["prefit_result_blind_audit"]["original_rank"] == 96
    assert manifest["prefit_result_blind_audit"]["scaled_rank"] == 96
    assert manifest["prefit_result_blind_audit"]["inverse_transform_max_abs_error"] <= 1e-10
    assert manifest["prefit_result_blind_audit"]["linear_predictor_and_contrast_max_abs_error"] <= 1e-10


def test_affine_reversibility_and_linear_predictor_contrast_equivalence() -> None:
    raw, config = synthetic_v2_contract()
    scaled, _, transform, audit = PREPARER.build_scaled_contract(raw, config)
    original_X = raw[[f"x_{index:04d}" for index in range(96)]].to_numpy(float)
    scaled_X = scaled[[f"x_{index:04d}" for index in range(96)]].to_numpy(float)
    means = transform.loc[1:, "frozen_sample_mean"].to_numpy(float)
    sds = transform.loc[1:, "frozen_sample_sd_ddof1"].to_numpy(float)
    rebuilt = scaled_X.copy(); rebuilt[:, 1:] = rebuilt[:, 1:] * sds + means
    beta = np.linspace(-0.15, 0.15, 96)
    beta_star = beta.copy(); beta_star[1:] *= sds; beta_star[0] = beta[0] + beta[1:] @ means
    assert np.max(np.abs(rebuilt - original_X)) <= 1e-10
    assert np.max(np.abs(original_X @ beta - scaled_X @ beta_star)) <= 1e-10
    assert audit["formal_model_run"] is False


def test_zero_or_near_zero_sd_fails_closed_before_any_model_run() -> None:
    raw, config = synthetic_v2_contract()
    raw["x_0001"] = 7.0
    with pytest.raises(ValueError, match="zero_or_near_zero_sd"):
        PREPARER.build_scaled_contract(raw, config)


def test_fitter_has_single_fixed_control_path_and_never_starts_bootstrap() -> None:
    text = FITTER_PATH.read_text(encoding="utf-8")
    for required in [
        'schema_version = "aim4_brglm2_v3"', "AS_mean", "MPL_Jeffreys", "maxit = 500",
        "epsilon = 1e-8", "slowit = 0.5", "max_step_factor = 6", "bootstrap_eligibility",
        "AS_mean_failed", "formal_bootstrap_started = FALSE", "explicit --point-only",
    ]:
        assert required in text
    assert "for (replicate_id" not in text
    assert "--bootstrap-only" in text and "never starts bootstrap" in text
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed locally; parse is deferred to the server")
    completed = subprocess.run([rscript, "-e", f"parse(file='{FITTER_PATH.as_posix()}')"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout
