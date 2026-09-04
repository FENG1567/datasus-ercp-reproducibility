#!/usr/bin/env python3
"""Prepare a frozen, explicit Aim 4 design for the separation fallback.

This program deliberately performs no outcome modelling.  It creates the one
input allowed to the R bias-reduced model, with all contrast values derived
once from the frozen analytic table.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fit_module():
    spec = importlib.util.spec_from_file_location("stage07_fit_aim4_outcomes_v2", ROOT / "stage07_fit_aim4_outcomes_v2.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen Aim 4 design implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the frozen primary-population rule and return an explicit matrix."""
    required = {
        "death_valid", "trailing12_complete", "in_hospital_death", "cnes7",
        "state_provider", "ivs_context", "ans_context", "trailing12_a_unique_aih",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen analytic table is missing required fields: {missing}")
    eligible = frame.loc[
        frame["death_valid"].fillna(False).astype(bool)
        & frame["trailing12_complete"].fillna(False).astype(bool)
        & frame[["ivs_context", "ans_context"]].notna().all(axis=1)
    ].copy()
    if eligible.empty:
        raise ValueError("No rows meet frozen death-valid, trailing-12, complete-context eligibility")
    y = pd.to_numeric(eligible["in_hospital_death"], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0.0, 1.0}):
        raise ValueError("Death endpoint must be observed and encoded 0/1 after frozen filtering")
    eligible["cnes7"] = eligible["cnes7"].astype(str).str.strip()
    eligible["state_provider"] = eligible["state_provider"].astype(str).str.strip()
    if (eligible["cnes7"] == "").any() or (eligible["state_provider"] == "").any():
        raise ValueError("CNES and corrected provider UF must be nonempty")
    if (eligible.groupby("cnes7", sort=False)["state_provider"].nunique() > 1).any():
        raise ValueError("A CNES maps to multiple corrected provider UFs; frozen provider geography QC failed")
    fit = _load_fit_module()
    X, design_names, audit, age_knots = fit.design_matrix(eligible, include_volume=True, allow_context=True)
    volume = pd.to_numeric(eligible["trailing12_a_unique_aih"], errors="raise").to_numpy(float)
    p10, p90 = (float(v) for v in np.quantile(volume, [0.10, 0.90]))
    volume_knots = np.asarray(audit["volume_knots"], dtype=float)
    basis_p10 = fit.rcs(np.array([p10]), volume_knots)[0]
    basis_p90 = fit.rcs(np.array([p90]), volume_knots)[0]
    volume_columns = [design_names.index(name) for name in ("volume_rcs_linear", "volume_rcs_nonlinear_1", "volume_rcs_nonlinear_2")]
    output = pd.DataFrame({
        "analysis_row_id": eligible.get("analysis_row_id", pd.Series(np.arange(len(eligible)), index=eligible.index)).astype(str).to_numpy(),
        "y": y.astype(int).to_numpy(),
        "cnes7": eligible["cnes7"].to_numpy(),
        "state_provider": eligible["state_provider"].to_numpy(),
    })
    for index in range(X.shape[1]):
        output[f"x_{index:04d}"] = X[:, index]
    config = {
        "schema_version": "aim4_brglm2_v2",
        "evidence": "associational/supportive",
        "population_rule": "death_valid AND trailing12_complete AND complete ivs_context/ans_context",
        "n_rows": int(len(output)),
        "death_events": int(y.sum()),
        "n_hospitals": int(output["cnes7"].nunique()),
        "n_provider_uf": int(output["state_provider"].nunique()),
        "design_names": design_names,
        "design_columns": [f"x_{index:04d}" for index in range(X.shape[1])],
        "volume_column_indices_zero_based": volume_columns,
        "volume_knots": [float(value) for value in volume_knots],
        "age_knots": [float(value) for value in age_knots],
        "p10": p10,
        "p90": p90,
        "volume_basis_p10": [float(value) for value in basis_p10],
        "volume_basis_p90": [float(value) for value in basis_p90],
        "design_audit": audit,
        "contrast": "fixed P90 versus P10; original frozen population standardization",
        "bootstrap": {"replicates": 2000, "seed": 20260830, "stratifier": "corrected state_provider", "cluster": "cnes7"},
        "no_wald_firth_p_values": True,
    }
    return output, config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    analytic = args.analytic.resolve()
    output_dir = args.output_dir.resolve()
    frame = pd.read_parquet(analytic)
    prepared, config = prepare(frame)
    input_path = output_dir / "aim4_brglm2_input_v2.csv.gz"
    design_path = output_dir / "aim4_brglm2_design_v2.json"
    _atomic_gzip_csv(input_path, prepared)
    _atomic_text(design_path, json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest = {
        "schema_version": "aim4_brglm2_v2",
        "evidence": "associational/supportive",
        "inputs": {analytic.name: sha256(analytic)},
        "outputs": {input_path.name: sha256(input_path), design_path.name: sha256(design_path)},
        "code_sha256": {Path(__file__).name: sha256(Path(__file__)), "stage07_fit_aim4_outcomes_v2.py": sha256(ROOT / "stage07_fit_aim4_outcomes_v2.py")},
        "frozen_population": {key: config[key] for key in ("n_rows", "death_events", "n_hospitals", "n_provider_uf", "p10", "p90", "volume_knots")},
    }
    manifest_path = output_dir / "aim4_brglm2_prepare_manifest_v2.json"
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "input": str(input_path), "manifest": str(manifest_path), "evidence": "associational/supportive"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
