#!/usr/bin/env python3
"""Create the sole, immutable scaled Aim 4 v3 numerical-stability input.

This program is deliberately data-engineering only: it never fits a model or
calculates a research outcome effect.  It performs the one affine
reparameterisation frozen in the 2026-08-30 amendment and records enough
result-blind algebraic checks for the R point-estimation gate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCHEMA = "aim4_brglm2_v3"
EXPECTED_N_PARAMETERS = 96
EXPECTED_PRODUCTION_ROWS = 30_900
SD_FLOOR = 1e-12
INVERSE_TOLERANCE = 1e-10
EXPECTED_P10 = 25.0
EXPECTED_P90 = 538.0
EXPECTED_KNOTS = (12.0, 76.0, 212.0, 600.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.17g")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.17g")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _finite_equal(observed: list[float] | np.ndarray, expected: tuple[float, ...]) -> bool:
    values = np.asarray(observed, dtype=float)
    return bool(values.shape == (len(expected),) and np.all(np.isfinite(values)) and np.allclose(values, expected, rtol=0.0, atol=1e-12))


def _require_v2_contract(raw: pd.DataFrame, config: dict) -> tuple[list[str], np.ndarray]:
    if config.get("schema_version") != "aim4_brglm2_v2":
        raise ValueError("Input design must be the frozen aim4_brglm2_v2 contract")
    design_columns = config.get("design_columns")
    design_names = config.get("design_names")
    if not isinstance(design_columns, list) or not isinstance(design_names, list):
        raise ValueError("Frozen v2 design lacks ordered design_columns/design_names")
    if len(design_columns) != EXPECTED_N_PARAMETERS or len(design_names) != EXPECTED_N_PARAMETERS:
        raise ValueError("Frozen v2 design must contain exactly 96 ordered parameters")
    if design_columns != [f"x_{index:04d}" for index in range(EXPECTED_N_PARAMETERS)]:
        raise ValueError("Frozen v2 design columns are not the expected explicit x_0000..x_0095 order")
    if str(design_names[0]).lower() != "intercept":
        raise ValueError("Frozen v2 design must retain intercept in column x_0000")
    expected_columns = ["analysis_row_id", "y", "cnes7", "state_provider", *design_columns]
    if list(raw.columns) != expected_columns:
        raise ValueError("Frozen v2 input column order/schema differs from its design contract")
    if raw.empty or raw["analysis_row_id"].isna().any() or raw["analysis_row_id"].astype(str).duplicated().any():
        raise ValueError("analysis_row_id must be nonmissing and unique")
    y = pd.to_numeric(raw["y"], errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(y)) or not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("Outcome must be finite and encoded 0/1")
    for identifier in ("cnes7", "state_provider"):
        values = raw[identifier].astype(str).str.strip()
        if raw[identifier].isna().any() or (values == "").any():
            raise ValueError(f"{identifier} must be nonmissing/nonempty")
    if not _finite_equal(config.get("volume_knots", []), EXPECTED_KNOTS):
        raise ValueError("Frozen volume knots must be 12, 76, 212, 600")
    if not np.isclose(float(config.get("p10", np.nan)), EXPECTED_P10, rtol=0.0, atol=1e-12):
        raise ValueError("Frozen P10 must equal 25")
    if not np.isclose(float(config.get("p90", np.nan)), EXPECTED_P90, rtol=0.0, atol=1e-12):
        raise ValueError("Frozen P90 must equal 538")
    positions = np.asarray(config.get("volume_column_indices_zero_based", []), dtype=int)
    if positions.shape != (3,) or np.any(positions < 1) or np.any(positions >= EXPECTED_N_PARAMETERS):
        raise ValueError("Frozen volume contrast must specify exactly three non-intercept design columns")
    if not np.array_equal(positions, np.asarray([93, 94, 95], dtype=int)):
        raise ValueError("Frozen volume design positions must be 93, 94, 95")
    for field in ("volume_basis_p10", "volume_basis_p90"):
        values = np.asarray(config.get(field, []), dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Frozen {field} must contain three finite values")
    X = raw.loc[:, design_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(X)):
        raise ValueError("Frozen v2 design matrix contains nonfinite values")
    if not np.allclose(X[:, 0], 1.0, rtol=0.0, atol=0.0):
        raise ValueError("Frozen intercept x_0000 must be exactly one for every row")
    return design_columns, X


def build_scaled_contract(raw: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """Scale 95 non-intercept columns and return output, config, table, audit.

    This function is side-effect free so tests can exercise its result-blind
    affine invariants without reading any formal analytic input.
    """
    design_columns, X = _require_v2_contract(raw, config)
    means = np.mean(X[:, 1:], axis=0)
    sds = np.std(X[:, 1:], axis=0, ddof=1)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(sds)) or np.any(sds <= SD_FLOOR):
        raise ValueError("zero_or_near_zero_sd")
    if np.linalg.matrix_rank(X) != EXPECTED_N_PARAMETERS:
        raise ValueError("Frozen v2 design matrix is rank deficient")
    scaled_X = X.copy()
    scaled_X[:, 1:] = (X[:, 1:] - means) / sds
    if not np.all(np.isfinite(scaled_X)):
        raise ValueError("Scaled design matrix contains nonfinite values")
    reconstructed = scaled_X.copy()
    reconstructed[:, 1:] = reconstructed[:, 1:] * sds + means
    inverse_error = float(np.max(np.abs(reconstructed - X)))
    if inverse_error > INVERSE_TOLERANCE:
        raise ValueError(f"inverse_transform_error_exceeds_{INVERSE_TOLERANCE:g}")
    original_rank = int(np.linalg.matrix_rank(X))
    scaled_rank = int(np.linalg.matrix_rank(scaled_X))
    if original_rank != EXPECTED_N_PARAMETERS or scaled_rank != EXPECTED_N_PARAMETERS:
        raise ValueError("rank_invariance_failed")
    original_condition = float(np.linalg.cond(X))
    scaled_condition = float(np.linalg.cond(scaled_X))
    if not np.isfinite(original_condition) or not np.isfinite(scaled_condition):
        raise ValueError("nonfinite_condition_number")
    positions = np.asarray(config["volume_column_indices_zero_based"], dtype=int)
    basis_low = np.asarray(config["volume_basis_p10"], dtype=float)
    basis_high = np.asarray(config["volume_basis_p90"], dtype=float)
    original_low, original_high = X.copy(), X.copy()
    original_low[:, positions] = basis_low
    original_high[:, positions] = basis_high
    scaled_low, scaled_high = original_low.copy(), original_high.copy()
    scaled_low[:, 1:] = (scaled_low[:, 1:] - means) / sds
    scaled_high[:, 1:] = (scaled_high[:, 1:] - means) / sds
    # A deterministic algebra-only vector proves coordinates and contrast are
    # equivalent without using y, fitting a model, or deriving an effect.
    beta_original = np.linspace(-0.25, 0.25, EXPECTED_N_PARAMETERS)
    beta_scaled = np.empty_like(beta_original)
    beta_scaled[1:] = beta_original[1:] * sds
    beta_scaled[0] = beta_original[0] + float(np.dot(beta_original[1:], means))
    linear_error = float(max(
        np.max(np.abs(X @ beta_original - scaled_X @ beta_scaled)),
        np.max(np.abs(original_low @ beta_original - scaled_low @ beta_scaled)),
        np.max(np.abs(original_high @ beta_original - scaled_high @ beta_scaled)),
    ))
    if linear_error > INVERSE_TOLERANCE:
        raise ValueError(f"contrast_parameterization_equivalence_error_exceeds_{INVERSE_TOLERANCE:g}")
    output = raw.loc[:, ["analysis_row_id", "y", "cnes7", "state_provider"]].copy()
    for index, column in enumerate(design_columns):
        output[column] = scaled_X[:, index]
    transform = pd.DataFrame({
        "design_column": design_columns,
        "design_name": [str(value) for value in config["design_names"]],
        "zero_based_index": np.arange(EXPECTED_N_PARAMETERS, dtype=int),
        "role": ["intercept_unmodified", *["centered_scaled_frozen_sample" for _ in range(EXPECTED_N_PARAMETERS - 1)]],
        "frozen_sample_mean": np.concatenate(([0.0], means)),
        "frozen_sample_sd_ddof1": np.concatenate(([1.0], sds)),
    })
    transform["is_volume_contrast_column"] = transform["zero_based_index"].isin(positions)
    scaled_config = dict(config)
    scaled_config.update({
        "schema_version": SCHEMA,
        "source_schema_version": "aim4_brglm2_v2",
        "scaling": {
            "intercept": "x_0000 unchanged",
            "non_intercept": "(x - frozen_sample_mean) / frozen_sample_sd_ddof1",
            "sd_ddof": 1,
            "sd_floor_fail_closed": SD_FLOOR,
            "inverse_tolerance": INVERSE_TOLERANCE,
            "n_scaled_non_intercept_columns": EXPECTED_N_PARAMETERS - 1,
        },
        "volume_basis_p10_scaled": [float((value - means[position - 1]) / sds[position - 1]) for value, position in zip(basis_low, positions)],
        "volume_basis_p90_scaled": [float((value - means[position - 1]) / sds[position - 1]) for value, position in zip(basis_high, positions)],
        "optimization_controls": {"maxit": 500, "epsilon": 1e-8, "slowit": 0.5, "max_step_factor": 6},
        "bootstrap_authorization": "not_started_by_preparer_or_point_fitter; eligible only if both point estimators converge",
    })
    audit = {
        "n_rows": int(len(raw)), "n_design_columns": EXPECTED_N_PARAMETERS,
        "original_rank": original_rank, "scaled_rank": scaled_rank,
        "original_condition_number": original_condition, "scaled_condition_number": scaled_condition,
        "inverse_transform_max_abs_error": inverse_error,
        "linear_predictor_and_contrast_max_abs_error": linear_error,
        "outcome_and_identifiers_preserved": True,
        "formal_model_run": False,
    }
    return output, scaled_config, transform, audit


def _existing(paths: list[Path]) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite immutable v3 evidence: {occupied}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-gz", required=True, type=Path, help="Frozen v2 explicit input CSV.GZ")
    parser.add_argument("--design-json", required=True, type=Path, help="Frozen v2 design JSON")
    parser.add_argument("--amendment", required=True, type=Path, help="Frozen 2026-08-30 numerical-stabilization amendment")
    parser.add_argument("--environment-lock", required=True, type=Path, help="Immutable environment-lock text/JSON to bind")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    input_gz, design_json, amendment, environment_lock = (value.resolve() for value in (args.input_gz, args.design_json, args.amendment, args.environment_lock))
    for required in (input_gz, design_json, amendment, environment_lock):
        if not required.is_file():
            raise FileNotFoundError(f"Required immutable input is absent: {required}")
    if sha256(amendment) != "4302844401381748d659d629c8e57a5eba73ddf97c2a6be64546da7fb472106b":
        raise ValueError("Frozen numerical-stabilization amendment SHA-256 does not match the authorized contract")
    output_dir = args.output_dir.resolve()
    output_paths = [
        output_dir / "aim4_brglm2_input_v3_scaled.csv.gz",
        output_dir / "aim4_brglm2_design_v3_scaled.json",
        output_dir / "aim4_brglm2_column_transform_v3.csv",
        output_dir / "aim4_brglm2_prefit_manifest_v3.json",
    ]
    _existing(output_paths)
    with gzip.open(input_gz, "rt", encoding="utf-8") as handle:
        raw = pd.read_csv(handle, dtype={"analysis_row_id": "string", "cnes7": "string", "state_provider": "string"})
    config = json.loads(design_json.read_text(encoding="utf-8"))
    if len(raw) != EXPECTED_PRODUCTION_ROWS or int(config.get("n_rows", -1)) != EXPECTED_PRODUCTION_ROWS:
        raise ValueError("Formal v3 preparation requires exactly the frozen 30,900-row v2 population")
    scaled, scaled_config, transform, audit = build_scaled_contract(raw, config)
    input_out, design_out, transform_out, manifest_out = output_paths
    _atomic_gzip_csv(input_out, scaled)
    _atomic_csv(transform_out, transform)
    scaled_config["source_v2"] = {"input_file": input_gz.name, "input_sha256": sha256(input_gz), "design_file": design_json.name, "design_sha256": sha256(design_json)}
    scaled_config["column_transform_file"] = transform_out.name
    scaled_config["column_transform_sha256"] = sha256(transform_out)
    scaled_config["prefit_result_blind_audit"] = audit
    _atomic_text(design_out, json.dumps(scaled_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest = {
        "schema_version": SCHEMA, "evidence": "associational/supportive", "formal_model_run": False,
        "authorized_amendment_sha256": "4302844401381748d659d629c8e57a5eba73ddf97c2a6be64546da7fb472106b",
        "inputs": {
            input_gz.name: sha256(input_gz), design_json.name: sha256(design_json),
            amendment.name: sha256(amendment), environment_lock.name: sha256(environment_lock),
        },
        "outputs": {input_out.name: sha256(input_out), design_out.name: sha256(design_out), transform_out.name: sha256(transform_out)},
        "code_sha256": {
            Path(__file__).name: sha256(Path(__file__)),
            "stage07_fit_aim4_brglm2_v3.R": sha256(ROOT / "stage07_fit_aim4_brglm2_v3.R"),
        },
        "prefit_result_blind_audit": audit,
        "frozen_contract": {"n_rows": EXPECTED_PRODUCTION_ROWS, "n_design_columns": EXPECTED_N_PARAMETERS, "p10": EXPECTED_P10, "p90": EXPECTED_P90, "volume_knots": list(EXPECTED_KNOTS), "controls": scaled_config["optimization_controls"]},
    }
    _atomic_text(manifest_out, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "formal_model_run": False, "manifest": str(manifest_out), "evidence": "associational/supportive"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
