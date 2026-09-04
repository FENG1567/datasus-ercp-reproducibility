#!/usr/bin/env python3
"""Finalize immutable v3 Aim 4 bootstrap replicates under frozen thresholds."""
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
SCHEMA = "aim4_brglm2_v3"
METRICS = ("risk_p10", "risk_p90", "rd", "rr")
HASH_FIELDS = ("input_sha256", "design_sha256", "prefit_manifest_sha256", "point_sha256", "run_manifest_sha256")
ALLOWED_FAILURE_REASONS = {"rank_deficient", "nonconvergence", "nonfinite_estimate", "invalid_prediction", "contrast_not_estimable", "runtime_error"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable final artifact: {path}")
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


def _atomic_gzip_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable final artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, compression="gzip", float_format="%.17g")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def percentile_and_mc_error(values: np.ndarray, probability: float) -> tuple[float, float]:
    ordered = np.sort(np.asarray(values, dtype=float))
    estimate = float(np.quantile(ordered, probability, method="linear"))
    rank_sd = math.sqrt(len(ordered) * probability * (1.0 - probability))
    center = probability * (len(ordered) - 1)
    low = int(max(0, math.floor(center - rank_sd)))
    high = int(min(len(ordered) - 1, math.ceil(center + rank_sd)))
    return estimate, float(abs(ordered[high] - ordered[low]) / 2.0)


def load_replicates(directory: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    records: list[dict] = []
    errors: list[str] = []
    hashes: dict[str, str] = {}
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
        hashes[path.name] = sha256(path)
    return (pd.DataFrame(records) if records else pd.DataFrame()), errors, hashes


def _items(value: object) -> list[str]:
    return [part for part in str(value or "").split(";") if part]


def structural_failure_audit(failed: pd.DataFrame) -> dict:
    """Pre-specified >=80% and >=20 failure-pattern audit using semantic support."""
    if failed.empty:
        return {"systematic_failure": False, "reason": "no_failed_replicates"}
    threshold = max(20, math.ceil(0.80 * len(failed)))
    reasons = failed.get("as_mean_failure_reason", failed.get("failure_reason", pd.Series(dtype=str))).fillna("unknown").astype(str).value_counts().to_dict()
    counters: dict[str, dict[str, int]] = {}
    repeated: dict[str, list[str]] = {}
    for column in ("provider_uf_zero_support", "calendar_month_constant_columns", "hospital_type_constant_columns"):
        counter: Counter[str] = Counter()
        for value in failed.get(column, pd.Series(dtype=str)).fillna(""):
            counter.update(_items(value))
        counters[column] = dict(sorted(counter.items()))
        repeated[column] = sorted(item for item, count in counter.items() if count >= threshold)
    repeated_reasons = sorted(reason for reason, count in reasons.items() if count >= threshold)
    return {
        "systematic_failure": bool(repeated_reasons or any(repeated.values())),
        "criterion": "same failure reason or provider-UF/calendar-month/hospital-type support loss in >=80% of failed replicates and >=20 replicates",
        "failure_reason_counts": reasons,
        "repeated_failure_reasons": repeated_reasons,
        "support_loss_counts": counters,
        "repeated_support_loss": repeated,
        "failed_replicate_count": int(len(failed)),
    }


def valid_replicates(records: pd.DataFrame, prefix: str) -> pd.DataFrame:
    status = records.get(f"{prefix}_status", pd.Series("failed", index=records.index)).eq("valid")
    for metric in METRICS:
        status &= pd.to_numeric(records.get(f"{prefix}_{metric}"), errors="coerce").map(np.isfinite).fillna(False)
    return records.loc[status].copy()


def summarize(values: pd.DataFrame, prefix: str) -> dict:
    result: dict = {}
    for metric in METRICS:
        series = pd.to_numeric(values[f"{prefix}_{metric}"], errors="raise").to_numpy(float)
        low, low_mc = percentile_and_mc_error(series, 0.025)
        high, high_mc = percentile_and_mc_error(series, 0.975)
        result[metric] = {
            "percentile_ci_low": low, "percentile_ci_high": high,
            "mc_error_low": low_mc, "mc_error_high": high_mc,
            "method": "percentile CI; local binomial-order-statistic MC-error approximation",
        }
    return result


def bootstrap_quality(n_valid: int, systematic: bool) -> str:
    if n_valid >= 1900 and not systematic:
        return "PASS"
    if n_valid >= 1800 and not systematic:
        return "WARNING_EXPLORATORY_SUPPORTIVE"
    return "DOWNGRADE"


def _require_point(point: dict, prefit_path: Path, point_path: Path) -> dict[str, str]:
    if point.get("schema_version") != SCHEMA or point.get("evidence") != "associational/supportive":
        raise ValueError("Point artifact is not the frozen v3 associational contract")
    if point.get("formal_bootstrap_started") is not False or point.get("bootstrap_eligibility") is not True:
        raise ValueError("Point artifact did not authorize bootstrap")
    if point.get("primary", {}).get("status") != "valid" or point.get("sensitivity", {}).get("status") != "valid" or point.get("detectseparation_audit", {}).get("status") != "PASS":
        raise ValueError("Point gate is not valid for bootstrap")
    hashes = {field: point.get(field) for field in HASH_FIELDS[:-1]}
    hashes["point_sha256"] = sha256(point_path)
    if not all(isinstance(value, str) and len(value) == 64 for value in hashes.values()):
        raise ValueError("Point artifact lacks required SHA-256 provenance")
    if hashes["prefit_manifest_sha256"] != sha256(prefit_path):
        raise ValueError("Point prefit hash does not match supplied prefit manifest")
    return hashes


def _require_run_manifest(run_manifest_path: Path, expected_hashes: dict[str, str], point_path: Path, prefit_path: Path) -> str:
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("evidence") != "associational/supportive":
        raise ValueError("Bootstrap run manifest is not the frozen v3 contract")
    inputs = manifest.get("inputs", {})
    required_inputs = {
        "aim4_brglm2_input_v3_scaled.csv.gz": expected_hashes["input_sha256"],
        "aim4_brglm2_design_v3_scaled.json": expected_hashes["design_sha256"],
        point_path.name: expected_hashes["point_sha256"], prefit_path.name: expected_hashes["prefit_manifest_sha256"],
    }
    if any(inputs.get(name) != value for name, value in required_inputs.items()):
        raise ValueError("Bootstrap run manifest does not bind supplied frozen input/design/point/prefit artifacts")
    if not isinstance(manifest.get("environment_lock_sha256"), str) or len(manifest["environment_lock_sha256"]) != 64:
        raise ValueError("Bootstrap run manifest lacks immutable environment-lock SHA-256")
    if manifest.get("bootstrap_design", {}).get("replicates") != N_REPLICATES or manifest.get("bootstrap_design", {}).get("seed") != 20260830:
        raise ValueError("Bootstrap run manifest does not retain frozen replicate/seed contract")
    source_root = Path(__file__).resolve().parent
    required_code = {
        Path(__file__).name: sha256(Path(__file__)),
        "stage07_bootstrap_aim4_brglm2_v3.R": sha256(source_root / "stage07_bootstrap_aim4_brglm2_v3.R"),
    }
    if any(manifest.get("code_sha256", {}).get(name) != value for name, value in required_code.items()):
        raise ValueError("Bootstrap run manifest does not bind the finalizer/bootstrap code identity")
    return sha256(run_manifest_path)


def finalize(replicate_dir: Path, point_path: Path, prefit_path: Path, run_manifest_path: Path, output_dir: Path) -> tuple[dict, int]:
    point = json.loads(point_path.read_text(encoding="utf-8"))
    expected_hashes = _require_point(point, prefit_path, point_path)
    expected_hashes["run_manifest_sha256"] = _require_run_manifest(run_manifest_path, expected_hashes, point_path, prefit_path)
    records, errors, file_hashes = load_replicates(replicate_dir)
    if not records.empty:
        for index, record in records.iterrows():
            replicate_id = record.get("replicate_id", "unknown")
            if record.get("schema_version") != SCHEMA or record.get("evidence") != "associational/supportive":
                errors.append(f"replicate {replicate_id}: schema/evidence contract mismatch")
            for field in ("as_mean_failure_reason", "mpl_jeffreys_failure_reason"):
                value = record.get(field)
                if pd.notna(value) and str(value) not in ALLOWED_FAILURE_REASONS:
                    errors.append(f"replicate {replicate_id}: invalid {field}={value}")
    observed = set(pd.to_numeric(records.get("replicate_id", pd.Series(dtype=int)), errors="coerce").dropna().astype(int))
    expected_ids = set(range(1, N_REPLICATES + 1))
    missing, out_of_range = sorted(expected_ids - observed), sorted(observed - expected_ids)
    qc: dict = {
        "schema_version": SCHEMA, "evidence": "associational/supportive", "no_wald_firth_p_values": True,
        "point_estimate_file": point_path.name, "point_estimate_sha256": sha256(point_path),
        "prefit_manifest_file": prefit_path.name, "prefit_manifest_sha256": sha256(prefit_path),
        "run_manifest_file": run_manifest_path.name, "run_manifest_sha256": sha256(run_manifest_path),
        "expected_replicates": N_REPLICATES, "observed_replicates": int(len(records)),
        "missing_replicates": missing, "out_of_range_replicates": out_of_range, "record_errors": errors,
        "primary_estimator": "brglm2 AS_mean (mean bias reduction)",
        "sensitivity_estimator": "brglm2 MPL_Jeffreys (Jeffreys-prior/Firth-style sensitivity)",
    }
    complete = not errors and not missing and not out_of_range and len(records) == N_REPLICATES
    if complete:
        for field, expected in expected_hashes.items():
            series = records.get(field, pd.Series(index=records.index, dtype=object))
            if not series.eq(expected).all():
                errors.append(f"Replicate provenance mismatch: {field}")
                complete = False
    if not complete:
        qc.update({"status": "BLOCKED", "reason": "Replicate files are not exact unique 1..2000 with matching provenance; no percentile CI computed"})
        _atomic_text_new(output_dir / "aim4_brglm2_final_qc_v3.json", json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        return qc, 2
    records = records.sort_values("replicate_id", kind="stable").reset_index(drop=True)
    primary = valid_replicates(records, "as_mean")
    sensitivity = valid_replicates(records, "mpl_jeffreys")
    primary_failed = records.loc[~records.index.isin(primary.index)].copy()
    sensitivity_failed = records.loc[~records.index.isin(sensitivity.index)].copy()
    primary_audit, sensitivity_audit = structural_failure_audit(primary_failed), structural_failure_audit(sensitivity_failed)
    primary_quality = bootstrap_quality(len(primary), primary_audit["systematic_failure"])
    sensitivity_quality = bootstrap_quality(len(sensitivity), sensitivity_audit["systematic_failure"])
    primary_summary = summarize(primary, "as_mean") if not primary.empty else {}
    sensitivity_summary = summarize(sensitivity, "mpl_jeffreys") if not sensitivity.empty else {}
    quality = "DOWNGRADE" if "DOWNGRADE" in (primary_quality, sensitivity_quality) else ("WARNING_EXPLORATORY_SUPPORTIVE" if "WARNING_EXPLORATORY_SUPPORTIVE" in (primary_quality, sensitivity_quality) else "PASS")
    point_primary, point_sensitivity = point["primary"], point["sensitivity"]
    method_check: dict[str, bool] = {}
    if primary_summary and sensitivity_summary:
        opposed = float(point_primary["rd"]) * float(point_sensitivity["rd"]) < 0
        as_ci, mpl_ci = primary_summary["rd"], sensitivity_summary["rd"]
        both_uncertain = as_ci["percentile_ci_low"] <= 0 <= as_ci["percentile_ci_high"] and mpl_ci["percentile_ci_low"] <= 0 <= mpl_ci["percentile_ci_high"]
        method_check = {"opposed_rd_direction": opposed, "both_rd_cis_include_null": both_uncertain}
        if opposed and both_uncertain:
            quality = "DOWNGRADE"
    qc.update({
        "status": quality, "as_mean_valid_replicates": int(len(primary)), "as_mean_failed_replicates": int(len(primary_failed)), "as_mean_failure_audit": primary_audit,
        "mpl_jeffreys_valid_replicates": int(len(sensitivity)), "mpl_jeffreys_failed_replicates": int(len(sensitivity_failed)), "mpl_jeffreys_failure_audit": sensitivity_audit,
        "as_mean_bootstrap_quality": primary_quality, "mpl_jeffreys_bootstrap_quality": sensitivity_quality,
        "bootstrap_percentile_summary": {"as_mean": primary_summary, "mpl_jeffreys": sensitivity_summary},
        "method_sensitivity_direction_check": method_check,
        "interpretation": "associational/supportive; Aim 4 status does not alter Aim 1–3",
    })
    merged_path, qc_path, manifest_path = (output_dir / "aim4_brglm2_bootstrap_merged_v3.csv.gz", output_dir / "aim4_brglm2_final_qc_v3.json", output_dir / "aim4_brglm2_final_manifest_v3.json")
    _atomic_gzip_csv_new(merged_path, records)
    _atomic_text_new(qc_path, json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest = {"schema_version": SCHEMA, "evidence": "associational/supportive", "inputs": {point_path.name: sha256(point_path), prefit_path.name: sha256(prefit_path), run_manifest_path.name: sha256(run_manifest_path)}, "outputs": {merged_path.name: sha256(merged_path), qc_path.name: sha256(qc_path)}, "replicate_file_sha256": file_hashes}
    _atomic_text_new(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return qc, 0 if quality != "DOWNGRADE" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-dir", required=True, type=Path)
    parser.add_argument("--point-estimate", required=True, type=Path)
    parser.add_argument("--prefit-manifest", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    qc, code = finalize(args.replicate_dir, args.point_estimate, args.prefit_manifest, args.run_manifest, args.output_dir)
    print(json.dumps({"status": qc["status"], "evidence": "associational/supportive"}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
