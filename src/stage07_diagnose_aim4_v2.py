#!/usr/bin/env python3
"""Pre-specified diagnostics for Aim 4 associational outcome models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def auc_rank(y: np.ndarray, score: np.ndarray) -> float | None:
    y, score = np.asarray(y, float), np.asarray(score, float)
    if len(np.unique(y)) < 2: return None
    ranks = pd.Series(score).rank(method="average").to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = frame[["outcome", "fitted_risk"]].dropna().copy()
    data["bin"] = pd.qcut(data["fitted_risk"], q=10, duplicates="drop")
    table = data.groupby("bin", observed=True).agg(n=("outcome", "size"), observed=("outcome", "mean"), fitted=("fitted_risk", "mean")).reset_index(drop=True)
    risk = np.clip(data["fitted_risk"].to_numpy(float), 1e-8, 1 - 1e-8); x = np.column_stack([np.ones(len(data)), np.log(risk / (1 - risk))]); y = data["outcome"].to_numpy(float)
    # Recalibration is only a diagnostic.  IRLS avoids a dependency on a local package.
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30))); w = np.clip(mu * (1 - mu), 1e-8, None); z = x @ beta + (y - mu) / w
        update = np.linalg.lstsq(x * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None)[0]
        if np.max(np.abs(update - beta)) < 1e-8: beta = update; break
        beta = update
    return table, {"brier": float(np.mean((y - risk) ** 2)), "auc_supportive": auc_rank(y, risk), "calibration_intercept": float(beta[0]), "calibration_slope": float(beta[1]), "n": int(len(data))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--analytic", required=True, type=Path); parser.add_argument("--model-dir", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.model_dir / "aim4_mortality_model_rows_v2.parquet"; qc_path = args.model_dir / "aim4_model_qc_v2.json"
    if not rows_path.exists() or not qc_path.exists(): raise FileNotFoundError("Model rows and model QC are required before diagnostics")
    rows, analytic, model_qc = pd.read_parquet(rows_path), pd.read_parquet(args.analytic), json.loads(qc_path.read_text(encoding="utf-8"))
    cal, cal_summary = calibration(rows)
    cal.to_parquet(args.output_dir / "aim4_calibration_bins_v2.parquet", index=False)
    event_by_hospital = rows.groupby("cnes7", as_index=False).agg(n=("outcome", "size"), death_events=("outcome", "sum")); event_by_hospital["events_lt5"] = event_by_hospital["death_events"] < 5
    event_by_state = rows.groupby("state_provider", as_index=False).agg(n=("outcome", "size"), death_events=("outcome", "sum"))
    event_by_hospital.to_parquet(args.output_dir / "aim4_events_by_hospital_v2.parquet", index=False); event_by_state.to_parquet(args.output_dir / "aim4_events_by_state_v2.parquet", index=False)
    # Observed minus fitted summaries are audit material, not a geographic effect model.
    rows["residual"] = rows["outcome"] - rows["fitted_risk"]
    temporal = rows.groupby("calendar_month", as_index=False).agg(n=("outcome", "size"), observed=("outcome", "mean"), fitted=("fitted_risk", "mean"), residual=("residual", "mean"))
    spatial = rows.groupby("state_provider", as_index=False).agg(n=("outcome", "size"), observed=("outcome", "mean"), fitted=("fitted_risk", "mean"), residual=("residual", "mean"))
    rows["volume_group"] = pd.qcut(rows["trailing12_a_unique_aih"], 10, duplicates="drop")
    volume = rows.groupby("volume_group", observed=True).agg(n=("outcome", "size"), observed=("outcome", "mean"), fitted=("fitted_risk", "mean"), residual=("residual", "mean")).reset_index()
    volume["volume_group"] = volume["volume_group"].astype(str)
    temporal.to_parquet(args.output_dir / "aim4_temporal_residuals_v2.parquet", index=False); spatial.to_parquet(args.output_dir / "aim4_spatial_residuals_v2.parquet", index=False); volume.to_parquet(args.output_dir / "aim4_volume_observed_fitted_v2.parquet", index=False)
    # No influence values are released; only the number above a conservative leverage-screen proxy.
    leverage_screen = float(np.quantile(np.abs(rows["residual"]), .99))
    influence = rows.loc[np.abs(rows["residual"]) >= leverage_screen, ["analysis_row_id", "cnes7", "outcome", "fitted_risk", "residual"]].copy()
    influence["screen_reason"] = "absolute residual at or above internal 99th percentile"
    influence.to_parquet(args.output_dir / "aim4_influence_screen_v2.parquet", index=False)
    invalid_death = int((~analytic["death_valid"].fillna(False)).sum()) if "death_valid" in analytic else None
    diagnostics = {"status": "PASS" if model_qc.get("status") == "PASS" else "DOWNGRADE", "evidence": "associational", "input_sha256": {str(args.analytic): sha256(args.analytic), str(rows_path): sha256(rows_path), str(qc_path): sha256(qc_path)}, "model_convergence": model_qc.get("status") == "PASS", "cluster_count": model_qc.get("n_clusters"), "events_per_parameter": model_qc.get("events_per_parameter"), "death_endpoint_invalid_or_missing_excluded_only": invalid_death, "calibration": cal_summary, "influence_screen_abs_residual_p99": leverage_screen, "nonfinite_fitted_risk_n": int((~np.isfinite(rows["fitted_risk"])).sum()), "extreme_fitted_risk_n": int(((rows["fitted_risk"] < 1e-6) | (rows["fitted_risk"] > 1 - 1e-6)).sum()), "hospital_event_distribution": {"clusters": int(len(event_by_hospital)), "clusters_lt5_events": int(event_by_hospital["events_lt5"].sum())}, "interpretation": "AUC and calibration are supportive fit diagnostics; this is not an individual prediction tool."}
    all_outputs = [p for p in args.output_dir.glob("aim4_*_v2.parquet")]
    diagnostics["output_sha256"] = {p.name: sha256(p) for p in all_outputs}
    qc_file = args.output_dir / "aim4_diagnostics_qc_v2.json"; qc_file.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "aim4_diagnostics_manifest_v2.json").write_text(json.dumps({"inputs": diagnostics["input_sha256"], "outputs": {p.name: sha256(p) for p in list(args.output_dir.glob("aim4_*_v2.parquet")) + [qc_file]}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False)); return 0 if diagnostics["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
