#!/usr/bin/env python3
"""Merge immutable Aim 4 cluster-bootstrap shards and apply frozen thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

N_REPLICATES = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def percentile_and_mc_error(values: np.ndarray, probability: float) -> tuple[float, float]:
    """Percentile and a local binomial-order-statistic MC-error approximation."""
    ordered = np.sort(np.asarray(values, dtype=float))
    estimate = float(np.quantile(ordered, probability, method="linear"))
    rank_sd = math.sqrt(len(ordered) * probability * (1.0 - probability))
    center = probability * (len(ordered) - 1)
    lower = int(max(0, math.floor(center - rank_sd)))
    upper = int(min(len(ordered) - 1, math.ceil(center + rank_sd)))
    mc_error = float(abs(ordered[upper] - ordered[lower]) / 2.0)
    return estimate, mc_error


def load_replicates(directory: Path) -> tuple[pd.DataFrame, list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    seen: set[int] = set()
    for path in sorted(directory.glob("replicate_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            replicate_id = int(record["replicate_id"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: malformed ({exc})")
            continue
        if replicate_id in seen:
            errors.append(f"duplicate replicate_id {replicate_id}")
        seen.add(replicate_id)
        record["file_name"] = path.name
        records.append(record)
    if not records:
        return pd.DataFrame(), errors
    return pd.DataFrame(records), errors


def structural_failure_audit(failed: pd.DataFrame) -> dict:
    if failed.empty:
        return {"systematic_failure": False, "reason": "no_failed_replicates"}
    reason_counts = failed.get("failure_reason", pd.Series(dtype=str)).fillna("unknown").astype(str).value_counts().to_dict()
    zero_counts = Counter()
    for value in failed.get("zero_design_columns", pd.Series(dtype=str)).fillna(""):
        zero_counts.update(item for item in str(value).split(";") if item)
    threshold = max(20, math.ceil(0.80 * len(failed)))
    repeated_zero = sorted(column for column, count in zero_counts.items() if count >= threshold)
    one_reason = max(reason_counts.values(), default=0) >= threshold
    return {
        "systematic_failure": bool(repeated_zero or one_reason),
        "criterion": "same failure reason or semantic zero design column in >=80% of failed replicates AND >=20 replicates (1% of 2000)",
        "failure_reason_counts": reason_counts,
        "zero_design_column_counts": dict(sorted(zero_counts.items())),
        "repeated_zero_design_columns": repeated_zero,
        "state_hospital_draws_among_failures": failed.get("state_hospital_counts", pd.Series(dtype=str)).fillna("").value_counts().to_dict(),
        "calendar_month_zero_columns_among_failures": failed.get("calendar_month_zero_columns", pd.Series(dtype=str)).fillna("").value_counts().to_dict(),
        "hospital_type_zero_columns_among_failures": failed.get("hospital_type_zero_columns", pd.Series(dtype=str)).fillna("").value_counts().to_dict(),
    }


def estimator_summary(records: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, dict]:
    """Return valid rows and fixed percentile summaries for one frozen estimator."""
    metrics = ["risk_p10", "risk_p90", "rd", "rr"]
    status_col = f"{prefix}_status"
    valid = records.get(status_col, pd.Series("failed", index=records.index)).eq("valid")
    for metric in metrics:
        valid &= pd.to_numeric(records.get(f"{prefix}_{metric}"), errors="coerce").map(np.isfinite).fillna(False)
    selected = records.loc[valid].copy()
    summary: dict = {}
    for metric in metrics:
        if selected.empty:
            break
        values = pd.to_numeric(selected[f"{prefix}_{metric}"], errors="raise").to_numpy(float)
        low, low_mc = percentile_and_mc_error(values, 0.025)
        high, high_mc = percentile_and_mc_error(values, 0.975)
        summary[metric] = {"percentile_ci_low": low, "percentile_ci_high": high, "mc_error_low": low_mc, "mc_error_high": high_mc, "method": "percentile CI; local binomial-order-statistic MC-error approximation"}
    return selected, summary


def bootstrap_quality(n_valid: int, systematic: bool) -> str:
    if n_valid >= 1900 and not systematic:
        return "PASS"
    if n_valid >= 1800 and not systematic:
        return "WARNING_EXPLORATORY_SUPPORTIVE"
    return "DOWNGRADE"


def finalize(replicate_dir: Path, point_estimate: Path, output_dir: Path) -> tuple[dict, int]:
    records, errors = load_replicates(replicate_dir)
    expected = set(range(1, N_REPLICATES + 1))
    observed = set(pd.to_numeric(records.get("replicate_id", pd.Series(dtype=int)), errors="coerce").dropna().astype(int))
    missing = sorted(expected - observed)
    out_of_range = sorted(observed - expected)
    point = json.loads(point_estimate.read_text(encoding="utf-8"))
    point_input_hash = point.get("input_sha256")
    point_design_hash = point.get("design_sha256")
    qc: dict = {
        "schema_version": "aim4_brglm2_v2", "evidence": "associational/supportive", "no_wald_firth_p_values": True,
        "point_estimate_file": point_estimate.name, "point_estimate_sha256": sha256(point_estimate),
        "expected_replicates": N_REPLICATES, "observed_replicates": int(len(records)), "missing_replicates": missing,
        "out_of_range_replicates": out_of_range, "record_errors": errors,
        "primary_estimator": "brglm2 AS_mean (mean bias reduction)",
        "sensitivity_estimator": "brglm2 MPL_Jeffreys (Jeffreys-prior/Firth-style sensitivity)",
    }
    complete = not errors and not missing and not out_of_range and len(records) == N_REPLICATES
    if complete and (not isinstance(point_input_hash, str) or len(point_input_hash) != 64 or not isinstance(point_design_hash, str) or len(point_design_hash) != 64):
        errors.append("Point estimate lacks 64-character input/design SHA-256 provenance")
        complete = False
    if complete:
        shard_hash_problem = (~records.get("input_sha256", pd.Series(index=records.index, dtype=object)).eq(point_input_hash) | ~records.get("design_sha256", pd.Series(index=records.index, dtype=object)).eq(point_design_hash)).any()
        if shard_hash_problem:
            errors.append("At least one shard does not match point-estimate input/design SHA-256")
            complete = False
    if not complete:
        qc.update({"status": "BLOCKED", "reason": "Bootstrap shards are not exactly 1..2000; percentile CI is not computed"})
        _atomic_text(output_dir / "aim4_brglm2_final_qc_v2.json", json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        return qc, 2
    records = records.sort_values("replicate_id", kind="stable").reset_index(drop=True)
    valid, as_summary = estimator_summary(records, "as_mean")
    failed = records.loc[~records.get("as_mean_status", pd.Series("failed", index=records.index)).eq("valid")].copy()
    sensitivity_valid, mpl_summary = estimator_summary(records, "mpl_jeffreys")
    audit = structural_failure_audit(failed)
    as_quality = bootstrap_quality(len(valid), audit["systematic_failure"])
    mpl_failures = records.loc[~records.get("mpl_jeffreys_status", pd.Series("failed", index=records.index)).eq("valid")].copy()
    mpl_audit = structural_failure_audit(mpl_failures)
    mpl_quality = bootstrap_quality(len(sensitivity_valid), mpl_audit["systematic_failure"])
    quality = as_quality
    qc.update({
        "as_mean_valid_replicates": int(len(valid)), "as_mean_failed_replicates": int(len(failed)), "as_mean_failure_audit": audit,
        "mpl_jeffreys_valid_replicates": int(len(sensitivity_valid)), "mpl_jeffreys_failed_replicates": int(len(mpl_failures)), "mpl_jeffreys_failure_audit": mpl_audit,
        "as_mean_bootstrap_quality": as_quality, "mpl_jeffreys_bootstrap_quality": mpl_quality,
    })
    primary = point.get("primary", {})
    sensitivity = point.get("sensitivity", {})
    primary_ok = primary.get("status") == "valid"
    sensitivity_ok = sensitivity.get("status") == "valid"
    if not primary_ok:
        quality = "DOWNGRADE"
    if primary_ok and sensitivity_ok:
        primary_rd, sensitivity_rd = float(primary["rd"]), float(sensitivity["rd"])
        sign_opposed = primary_rd * sensitivity_rd < 0
        as_ci = as_summary.get("rd", {})
        mpl_ci = mpl_summary.get("rd", {})
        both_uncertain = bool(as_ci and mpl_ci and as_ci["percentile_ci_low"] <= 0 <= as_ci["percentile_ci_high"] and mpl_ci["percentile_ci_low"] <= 0 <= mpl_ci["percentile_ci_high"])
        ci_overlap = bool(as_ci and mpl_ci and max(as_ci["percentile_ci_low"], mpl_ci["percentile_ci_low"]) <= min(as_ci["percentile_ci_high"], mpl_ci["percentile_ci_high"]))
        if sign_opposed and both_uncertain:
            quality = "DOWNGRADE"
        qc["method_sensitivity_direction_check"] = {"opposed_rd_direction": sign_opposed, "both_rd_cis_include_null": both_uncertain, "rd_percentile_ci_overlap": ci_overlap}
    if mpl_quality == "DOWNGRADE":
        quality = "DOWNGRADE"
    qc.update({"status": quality, "point_primary_status": primary.get("status"), "point_sensitivity_status": sensitivity.get("status"), "bootstrap_percentile_summary": {"as_mean": as_summary, "mpl_jeffreys": mpl_summary}, "interpretation": "associational/supportive; Aim 4 status does not alter Aim 1–3"})
    _atomic_gzip_csv(output_dir / "aim4_brglm2_bootstrap_merged_v2.csv.gz", records)
    manifest = {"schema_version": "aim4_brglm2_v2", "evidence": "associational/supportive", "inputs": {point_estimate.name: sha256(point_estimate)}, "outputs": {"aim4_brglm2_bootstrap_merged_v2.csv.gz": sha256(output_dir / "aim4_brglm2_bootstrap_merged_v2.csv.gz")}, "replicate_file_sha256": {path.name: sha256(path) for path in sorted(replicate_dir.glob("replicate_*.json"))}}
    _atomic_text(output_dir / "aim4_brglm2_final_qc_v2.json", json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest["outputs"]["aim4_brglm2_final_qc_v2.json"] = sha256(output_dir / "aim4_brglm2_final_qc_v2.json")
    _atomic_text(output_dir / "aim4_brglm2_final_manifest_v2.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return qc, 0 if quality != "DOWNGRADE" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-dir", required=True, type=Path)
    parser.add_argument("--point-estimate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc, code = finalize(args.replicate_dir, args.point_estimate, args.output_dir)
    print(json.dumps({"status": qc["status"], "evidence": "associational/supportive"}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
