#!/usr/bin/env python3
"""Independent statsmodels cross-check of the frozen Aim-4 custom IRLS fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from .stage07_fit_aim4_outcomes_v2 import design_matrix, standardized_volume_contrast
except ImportError:
    from stage07_fit_aim4_outcomes_v2 import design_matrix, standardized_volume_contrast


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analytic", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)

    qc_path = args.model_dir / "aim4_model_qc_v2.json"
    object_path = args.model_dir / "aim4_mortality_model_object_v2.npz"
    if not qc_path.exists() or not object_path.exists():
        raise FileNotFoundError("Aim-4 model QC and frozen model object are required")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.analytic)
    base = frame.loc[
        frame["death_valid"].fillna(False)
        & frame["trailing12_complete"].fillna(False)
    ].copy()
    allow_context = "ivs_context" in qc.get("design_columns", [])
    if allow_context:
        base = base.loc[base[["ivs_context", "ans_context"]].notna().all(axis=1)].copy()
    X, names, design_audit, _ = design_matrix(base, allow_context=allow_context)
    y = pd.to_numeric(base["in_hospital_death"], errors="raise").to_numpy(float)
    groups = base["cnes7"].astype(str).to_numpy()
    saved = np.load(object_path, allow_pickle=False)
    saved_names = saved["design_names"].astype(str).tolist()
    if names != saved_names:
        raise RuntimeError("reconstructed design columns do not equal frozen model columns")

    model = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
        use_t=False,
        maxiter=200,
        tol=1e-10,
    )
    beta = np.asarray(model.params, float)
    covariance = np.asarray(model.cov_params(), float)
    beta_saved = np.asarray(saved["beta"], float)
    covariance_saved = np.asarray(saved["covariance"], float)
    fit_for_contrast = {
        "beta": beta,
        "cov": covariance,
        "volume_knots": design_audit["volume_knots"],
    }
    contrast, _ = standardized_volume_contrast(base, X, names, fit_for_contrast)
    primary = qc["primary_contrast"]
    checks = {
        "analytic_hash_matches_model_qc": (
            qc.get("input_sha256") == sha256(args.analytic)
        ),
        "model_object_hash_matches_qc": (
            qc.get("model_object_sha256") == sha256(object_path)
        ),
        "design_names_identical": names == saved_names,
        "beta_allclose": bool(np.allclose(beta, beta_saved, rtol=1e-7, atol=1e-9)),
        "cluster_covariance_allclose": bool(
            np.allclose(covariance, covariance_saved, rtol=1e-5, atol=1e-8)
        ),
        "primary_rd_allclose": bool(
            np.isclose(
                contrast["marginal_rd_percentage_points"],
                primary["marginal_rd_percentage_points"],
                rtol=1e-7,
                atol=1e-9,
            )
        ),
        "primary_rr_allclose": bool(
            np.isclose(
                contrast["marginal_rr"],
                primary["marginal_rr"],
                rtol=1e-7,
                atol=1e-9,
            )
        ),
    }
    audit = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FIX",
        "purpose": "Independent statsmodels GLM reconstruction of custom IRLS coefficients, cluster sandwich covariance, and standardized primary contrast",
        "checks": checks,
        "n": len(base),
        "events": int(y.sum()),
        "clusters": int(pd.Series(groups).nunique()),
        "parameters": X.shape[1],
        "maximum_absolute_beta_difference": float(np.max(np.abs(beta - beta_saved))),
        "maximum_absolute_covariance_difference": float(
            np.max(np.abs(covariance - covariance_saved))
        ),
        "primary_contrast_statsmodels": contrast,
        "input_hashes": {
            "analytic": sha256(args.analytic),
            "model_object": sha256(object_path),
            "model_qc": sha256(qc_path),
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
