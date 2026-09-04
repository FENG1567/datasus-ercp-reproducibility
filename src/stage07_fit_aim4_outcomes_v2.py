#!/usr/bin/env python3
"""Frozen-plan Aim 4 outcome models (associational evidence only).

This implementation uses a population-averaged logistic GLM with hospital-cluster
robust covariance.  It intentionally has no data-driven variable selection and does
not create a facility league table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

Z975 = 1.959963984540054


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ModelGateError(RuntimeError):
    pass


def rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Four-knot restricted cubic spline basis with K-2 nonlinear terms."""
    x = np.asarray(x, float); k = np.asarray(knots, float)
    if len(k) != 4 or not np.all(np.diff(k) > 0):
        raise ModelGateError("Spline knots are not distinct")
    def positive(v: np.ndarray) -> np.ndarray:
        return np.maximum(v, 0.0) ** 3

    span = k[-1] - k[0]
    nonlinear = []
    for knot in k[:-2]:
        term = (
            positive(x - knot)
            - ((k[-1] - knot) / (k[-1] - k[-2])) * positive(x - k[-2])
            + ((k[-2] - knot) / (k[-1] - k[-2])) * positive(x - k[-1])
        ) / (span ** 2)
        nonlinear.append(term)
    return np.column_stack([x, *nonlinear])


def _clean_category(series: pd.Series) -> pd.Series:
    return series.fillna("MISSING").astype(str).str.strip().replace({"": "MISSING", "nan": "MISSING"})


def _add_categorical(parts: list[np.ndarray], names: list[str], series: pd.Series, prefix: str, audit: dict) -> None:
    value = _clean_category(series)
    levels = sorted(value.unique().tolist())
    reference = levels[0]
    audit[prefix] = {"reference": reference, "levels": levels}
    for level in levels[1:]:
        parts.append((value == level).to_numpy(float)[:, None]); names.append(f"{prefix}[{level}]")


def design_matrix(frame: pd.DataFrame, include_volume: bool = True, allow_context: bool = True) -> tuple[np.ndarray, list[str], dict, np.ndarray]:
    """Make an explicit, frozen covariate design.  ICU, stay and payment are excluded."""
    forbidden = {"any_icu", "length_of_stay_days", "reimbursement_brl", "reimbursement_2025_brl"}
    required = ["age_years", "sex_category", "race_category", "emergency_admission", "diagnostic_stratum", "cnes7", "state_provider", "calendar_month"]
    missing = [c for c in required if c not in frame]
    if missing: raise ModelGateError(f"Required model fields absent: {missing}")
    use = frame.copy()
    audit: dict = {"explicit_references": {}, "unavailable_covariates": [], "forbidden_not_used": sorted(forbidden)}
    age = pd.to_numeric(use["age_years"], errors="coerce")
    if age.isna().any(): raise ModelGateError("Age is missing; no imputation is allowed for this primary covariate")
    age_knots = np.quantile(age, [0.05, 0.35, 0.65, 0.95])
    parts, names = [np.ones((len(use), 1))], ["intercept"]
    age_basis = rcs(age.to_numpy(), age_knots); parts.append(age_basis); names += ["age_rcs_linear", "age_rcs_nonlinear_1", "age_rcs_nonlinear_2"]
    for col, label in [("sex_category", "sex"), ("race_category", "race"), ("emergency_admission", "emergency"), ("diagnostic_stratum", "diagnosis"), ("hospital_type", "hospital_type"), ("endoscopy_capability", "endoscopy_capability"), ("state_provider", "state"), ("calendar_month", "calendar_month")]:
        if col in use:
            _add_categorical(parts, names, use[col], label, audit["explicit_references"])
        else: audit["unavailable_covariates"].append(col)
    for col in ["comorbidity_burden", "beds_sus", "icu_beds"]:
        if col in use and pd.to_numeric(use[col], errors="coerce").notna().any():
            x = pd.to_numeric(use[col], errors="coerce"); miss = x.isna().to_numpy(float)
            x = x.fillna(x.median()); parts.append(x.to_numpy(float)[:, None]); names.append(col)
            if miss.any(): parts.append(miss[:, None]); names.append(f"{col}_missing_indicator")
        else: audit["unavailable_covariates"].append(col)
    if allow_context:
        for col in ["ivs_context", "ans_context"]:
            if col not in use: raise ModelGateError(f"Context field absent: {col}")
            if pd.to_numeric(use[col], errors="coerce").isna().any(): raise ModelGateError("Context primary analysis requires complete context rows")
            parts.append(pd.to_numeric(use[col], errors="coerce").to_numpy(float)[:, None]); names.append(col)
    if include_volume:
        volume = pd.to_numeric(use["trailing12_a_unique_aih"], errors="coerce")
        if volume.isna().any(): raise ModelGateError("Primary volume requires twelve complete prior months")
        knots = np.quantile(volume, [0.05, 0.35, 0.65, 0.95])
        vb = rcs(volume.to_numpy(float), knots); parts.append(vb); names += ["volume_rcs_linear", "volume_rcs_nonlinear_1", "volume_rcs_nonlinear_2"]
        audit["volume_knots"] = [float(x) for x in knots]
    X = np.column_stack(parts)
    if np.linalg.matrix_rank(X) < X.shape[1]: raise ModelGateError("Design matrix is rank deficient")
    if not np.isfinite(X).all(): raise ModelGateError("Non-finite design matrix")
    return X, names, audit, age_knots


def logistic_glm_clustered(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, max_iter: int = 120) -> dict:
    """IRLS binomial GLM plus cluster sandwich covariance; no external package needed."""
    y = np.asarray(y, float); X = np.asarray(X, float); clusters = np.asarray(clusters)
    if set(np.unique(y)) - {0.0, 1.0}: raise ModelGateError("Binary endpoint is not encoded 0/1")
    if y.min() == y.max(): raise ModelGateError("Outcome has no variation; separation/degeneracy")
    beta = np.zeros(X.shape[1]); converged = False
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta)); w = np.clip(mu * (1 - mu), 1e-8, None)
        z = eta + (y - mu) / w
        beta_new = np.linalg.lstsq(X * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-8: beta, converged = beta_new, True; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta)); w = np.clip(mu * (1 - mu), 1e-8, None)
    bread = np.linalg.pinv((X * w[:, None]).T @ X)
    meat = np.zeros((X.shape[1], X.shape[1])); unique = np.unique(clusters)
    for cluster in unique:
        score = X[clusters == cluster].T @ (y[clusters == cluster] - mu[clusters == cluster]); meat += np.outer(score, score)
    correction = (len(unique) / max(len(unique) - 1, 1)) * ((len(y) - 1) / max(len(y) - X.shape[1], 1))
    cov = correction * bread @ meat @ bread
    finite = bool(np.isfinite(beta).all() and np.isfinite(cov).all())
    suspected = (not converged) or (not finite) or np.max(np.abs(beta)) > 25 or np.max(np.sqrt(np.maximum(np.diag(cov), 0))) > 20
    if suspected: raise ModelGateError("Separation or non-finite/unstable maximum-likelihood estimate")
    return {"beta": beta, "cov": cov, "mu": mu, "converged": converged, "n_clusters": int(len(unique))}


def gamma_log_glm_clustered(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, max_iter: int = 120) -> dict:
    y = np.asarray(y, float)
    if (y <= 0).any() or not np.isfinite(y).all(): raise ModelGateError("Gamma-log endpoint must be finite and positive")
    beta = np.zeros(X.shape[1]); converged = False
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30); mu = np.exp(eta); z = eta + (y - mu) / mu
        beta_new = np.linalg.lstsq(X, z, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-8: beta, converged = beta_new, True; break
        beta = beta_new
    mu = np.exp(np.clip(X @ beta, -30, 30)); bread = np.linalg.pinv(X.T @ X); meat = np.zeros((X.shape[1], X.shape[1]))
    for cluster in np.unique(clusters):
        mask = clusters == cluster; score = X[mask].T @ ((y[mask] - mu[mask]) / mu[mask]); meat += np.outer(score, score)
    cov = bread @ meat @ bread
    if not converged or not np.isfinite(beta).all() or not np.isfinite(cov).all(): raise ModelGateError("Gamma-log model did not produce finite estimates")
    return {"beta": beta, "cov": cov, "mu": mu, "converged": converged, "n_clusters": int(len(np.unique(clusters)))}


def _risk_and_gradient(X: np.ndarray, beta: np.ndarray) -> tuple[float, np.ndarray]:
    p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
    return float(p.mean()), (X * (p * (1 - p))[:, None]).mean(axis=0)


def _marginal_column_contrast(
    X: np.ndarray,
    fit: dict,
    column: int,
    low: float,
    high: float,
) -> dict[str, float]:
    lo, hi = X.copy(), X.copy()
    lo[:, column] = low
    hi[:, column] = high
    p0, g0 = _risk_and_gradient(lo, fit["beta"])
    p1, g1 = _risk_and_gradient(hi, fit["beta"])
    rd = p1 - p0
    grad_rd = g1 - g0
    se_rd = math.sqrt(max(float(grad_rd @ fit["cov"] @ grad_rd), 0.0))
    rr = p1 / max(p0, 1e-12)
    grad_rr = g1 / max(p0, 1e-12) - p1 * g0 / max(p0 ** 2, 1e-12)
    se_log_rr = math.sqrt(max(float(grad_rr @ fit["cov"] @ grad_rr), 0.0)) / max(rr, 1e-12)
    p_value = math.erfc(abs(rd / se_rd) / math.sqrt(2.0)) if se_rd > 0 else float("nan")
    return {
        "risk_low": p0,
        "risk_high": p1,
        "marginal_rd_percentage_points": 100 * rd,
        "rd_ci_low_percentage_points": 100 * (rd - Z975 * se_rd),
        "rd_ci_high_percentage_points": 100 * (rd + Z975 * se_rd),
        "marginal_rr": rr,
        "rr_ci_low": math.exp(math.log(rr) - Z975 * se_log_rr),
        "rr_ci_high": math.exp(math.log(rr) + Z975 * se_log_rr),
        "raw_p_value_rd": p_value,
    }


def _bh_adjust(p_values: list[float]) -> list[float]:
    result = [float("nan")] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    if not valid:
        return result
    ordered = sorted(valid, key=lambda item: item[1])
    running = 1.0
    for reverse_rank, (index, value) in enumerate(reversed(ordered), start=1):
        rank = len(ordered) - reverse_rank + 1
        running = min(running, value * len(ordered) / rank)
        result[index] = min(1.0, running)
    return result


def fit_secondary_exposures(frame: pd.DataFrame, allow_context: bool) -> pd.DataFrame:
    """Fit the frozen secondary exposure family as explicitly exploratory models."""
    candidates: list[tuple[str, pd.DataFrame, pd.Series, float, float, str]] = []

    maturity = (
        pd.to_numeric(frame["months_since_first_observed_coded_use"], errors="coerce")
        if "months_since_first_observed_coded_use" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    prevalent = frame.get(
        "prevalent_at_window_start", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    maturity_mask = maturity.notna() & maturity.ge(0) & (~prevalent)
    if maturity_mask.any():
        current = frame.loc[maturity_mask].copy().reset_index(drop=True)
        exposure = pd.to_numeric(
            current["months_since_first_observed_coded_use"], errors="raise"
        ) / 12.0
        candidates.append(
            (
                "years_since_first_observed_coded_use",
                current,
                exposure,
                float(exposure.quantile(0.10)),
                float(exposure.quantile(0.90)),
                "P90 versus P10 years since first observed coded use; left-censored prevalent hospitals excluded",
            )
        )

    maintenance = frame.get("maintenance_status", pd.Series(pd.NA, index=frame.index))
    maintenance_mask = maintenance.notna()
    if maintenance_mask.any():
        current = frame.loc[maintenance_mask].copy().reset_index(drop=True)
        exposure = current["maintenance_status"].astype(bool).astype(float)
        candidates.append(
            (
                "maintenance_6of12",
                current,
                exposure,
                0.0,
                1.0,
                "maintained versus not maintained among evaluable hospital-months",
            )
        )

    strength = (
        pd.to_numeric(frame["network_in_strength"], errors="coerce")
        if "network_in_strength" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    strength_mask = strength.notna() & strength.ge(0)
    if strength_mask.any():
        current = frame.loc[strength_mask].copy().reset_index(drop=True)
        raw_strength = pd.to_numeric(current["network_in_strength"], errors="raise")
        exposure = np.log1p(raw_strength)
        candidates.append(
            (
                "log1p_network_in_strength",
                current,
                pd.Series(exposure, index=current.index),
                float(np.quantile(exposure, 0.10)),
                float(np.quantile(exposure, 0.90)),
                "P90 versus P10 log(1 + annual cohort-B CNES in-strength)",
            )
        )
        unique_hospital_year = current[
            ["cnes7", "year", "network_in_strength"]
        ].drop_duplicates(["cnes7", "year"])
        threshold = unique_hospital_year.groupby("year")["network_in_strength"].quantile(0.80)
        current["network_hub80"] = current["network_in_strength"].ge(
            current["year"].map(threshold)
        )
        candidates.append(
            (
                "network_hub80",
                current,
                current["network_hub80"].astype(float),
                0.0,
                1.0,
                "annual CNES in-strength at or above versus below the hospital-year 80th percentile",
            )
        )

    rows: list[dict[str, object]] = []
    for name, current, exposure, low, high, label in candidates:
        if len(current) == 0 or exposure.nunique(dropna=True) < 2 or not low < high:
            rows.append(
                {
                    "exposure": name,
                    "status": "NOT_ESTIMABLE",
                    "contrast": label,
                    "multiplicity": "secondary exposure family; BH FDR",
                }
            )
            continue
        try:
            base_X, base_names, design_audit, _ = design_matrix(
                current, include_volume=False, allow_context=allow_context
            )
            X = np.column_stack([base_X, np.asarray(exposure, dtype=float)])
            names = [*base_names, name]
            if np.linalg.matrix_rank(X) < X.shape[1]:
                raise ModelGateError("Secondary exposure design matrix is rank deficient")
            y = pd.to_numeric(current["in_hospital_death"], errors="raise").to_numpy(float)
            model = logistic_glm_clustered(X, y, current["cnes7"].astype(str).to_numpy())
            contrast = _marginal_column_contrast(X, model, len(names) - 1, low, high)
            rows.append(
                {
                    "exposure": name,
                    "status": "PASS",
                    "contrast": label,
                    "contrast_low_model_scale": low,
                    "contrast_high_model_scale": high,
                    "n": len(current),
                    "events": int(y.sum()),
                    "n_parameters": X.shape[1],
                    "n_clusters": model["n_clusters"],
                    "events_per_parameter": float(min(y.sum(), len(y) - y.sum()) / X.shape[1]),
                    "multiplicity": "secondary exposure family; BH FDR",
                    "evidence": "associational/exploratory",
                    "design_audit": json.dumps(design_audit, ensure_ascii=False, sort_keys=True),
                    **contrast,
                }
            )
        except ModelGateError as exc:
            rows.append(
                {
                    "exposure": name,
                    "status": "DOWNGRADE",
                    "contrast": label,
                    "note": str(exc),
                    "multiplicity": "secondary exposure family; BH FDR",
                }
            )
    adjusted = _bh_adjust(
        [float(row.get("raw_p_value_rd", float("nan"))) for row in rows]
    )
    for row, value in zip(rows, adjusted):
        row["bh_fdr_p_value_rd"] = value
    return pd.DataFrame(rows)


def standardized_volume_contrast(frame: pd.DataFrame, X: np.ndarray, names: list[str], fit: dict) -> tuple[dict, pd.DataFrame]:
    volume = pd.to_numeric(frame["trailing12_a_unique_aih"], errors="raise").to_numpy(float)
    q10, q90 = np.quantile(volume, [0.10, 0.90]); knots = np.asarray(fit["volume_knots"], float)
    lo, hi = X.copy(), X.copy(); columns = {n: i for i, n in enumerate(names)}
    volume_names = ["volume_rcs_linear", "volume_rcs_nonlinear_1", "volume_rcs_nonlinear_2"]
    volume_columns = [columns[name] for name in volume_names]
    lo_basis, hi_basis = rcs(np.full(len(X), q10), knots), rcs(np.full(len(X), q90), knots)
    lo[:, volume_columns] = lo_basis
    hi[:, volume_columns] = hi_basis
    p0, g0 = _risk_and_gradient(lo, fit["beta"]); p1, g1 = _risk_and_gradient(hi, fit["beta"])
    rd, grad = p1 - p0, g1 - g0; se = math.sqrt(max(float(grad @ fit["cov"] @ grad), 0.0))
    rr = p1 / max(p0, 1e-12); grad_rr = g1 / max(p0, 1e-12) - p1 * g0 / max(p0 ** 2, 1e-12); se_log_rr = math.sqrt(max(float(grad_rr @ fit["cov"] @ grad_rr), 0.0)) / max(rr, 1e-12)
    result = {"contrast": "P90 versus P10 trailing-12-month volume", "p10": float(q10), "p90": float(q90), "risk_p10": p0, "risk_p90": p1, "marginal_rd_percentage_points": 100 * rd, "rd_ci_low_percentage_points": 100 * (rd - Z975 * se), "rd_ci_high_percentage_points": 100 * (rd + Z975 * se), "marginal_rr": rr, "rr_ci_low": math.exp(math.log(rr) - Z975 * se_log_rr), "rr_ci_high": math.exp(math.log(rr) + Z975 * se_log_rr), "evidence": "associational"}
    grid = np.linspace(np.quantile(volume, .05), np.quantile(volume, .95), 30); rows = []
    for v in grid:
        xx = X.copy(); basis = rcs(np.full(len(X), v), knots); xx[:, volume_columns] = basis
        risk, grad_curve = _risk_and_gradient(xx, fit["beta"]); se_curve = math.sqrt(max(float(grad_curve @ fit["cov"] @ grad_curve), 0.0))
        rows.append({"trailing12_a_unique_aih": float(v), "standardized_risk": risk, "ci_low": max(0, risk - Z975 * se_curve), "ci_high": min(1, risk + Z975 * se_curve), "evidence": "associational"})
    return result, pd.DataFrame(rows)


def _write_json(path: Path, payload: dict) -> None: path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_secondary(frame: pd.DataFrame, X: np.ndarray, clusters: np.ndarray, volume_fit: dict, names: list[str]) -> list[dict]:
    rows: list[dict] = []
    for endpoint, label in [("any_icu", "any ICU"), ("length_of_stay_days", "length of stay"), ("reimbursement_2025_brl", "2025-BRL reimbursement")]:
        if endpoint not in frame or frame[endpoint].isna().all():
            rows.append({"endpoint": label, "status": "NOT_AVAILABLE", "n_endpoint": 0, "multiplicity": "secondary/exploratory"}); continue
        valid = frame[endpoint].notna().to_numpy()
        if endpoint == "any_icu":
            try:
                model = logistic_glm_clustered(X[valid], pd.to_numeric(frame.loc[valid, endpoint], errors="raise").to_numpy(float), clusters[valid])
                model["volume_knots"] = volume_fit["volume_knots"]; result, _ = standardized_volume_contrast(frame.loc[valid].reset_index(drop=True), X[valid], names, model)
                rows.append({"endpoint": label, "status": "PASS", "n_endpoint": int(valid.sum()), "scale": "marginal RD and RR", "multiplicity": "secondary/exploratory", **result})
            except ModelGateError as exc: rows.append({"endpoint": label, "status": "DOWNGRADE", "n_endpoint": int(valid.sum()), "note": str(exc), "multiplicity": "secondary/exploratory"})
        else:
            y = pd.to_numeric(frame[endpoint], errors="coerce").to_numpy(); valid &= y > 0
            try:
                model = gamma_log_glm_clustered(X[valid], y[valid], clusters[valid]); vol = pd.to_numeric(frame.loc[valid, "trailing12_a_unique_aih"], errors="raise").to_numpy(float); q10, q90 = np.quantile(vol, [.10, .90]); columns = {n: i for i, n in enumerate(names)}
                lo, hi = X[valid].copy(), X[valid].copy(); knots = np.asarray(volume_fit["volume_knots"]); blo, bhi = rcs(np.full(len(lo), q10), knots), rcs(np.full(len(lo), q90), knots)
                volume_columns = [columns[name] for name in ["volume_rcs_linear", "volume_rcs_nonlinear_1", "volume_rcs_nonlinear_2"]]
                lo[:, volume_columns] = blo; hi[:, volume_columns] = bhi
                mean_lo, mean_hi = float(np.exp(np.clip(lo @ model["beta"], -30, 30)).mean()), float(np.exp(np.clip(hi @ model["beta"], -30, 30)).mean())
                rows.append({"endpoint": label, "status": "PASS", "n_endpoint": int(valid.sum()), "model": "Gamma log", "p10": float(q10), "p90": float(q90), "marginal_mean_difference": mean_hi - mean_lo, "multiplicity": "secondary/exploratory", "payment_label": "reimbursement; not a resource-use valuation" if endpoint.startswith("reimbursement") else None})
            except ModelGateError as exc: rows.append({"endpoint": label, "status": "DOWNGRADE", "n_endpoint": int(valid.sum()), "note": str(exc), "multiplicity": "secondary/exploratory"})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--analytic", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--context-population-coverage", required=True, type=float); parser.add_argument("--allow-context-downgrade", action="store_true")
    args = parser.parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True); frame = pd.read_parquet(args.analytic)
    qc: dict = {"status": "PASS", "evidence": "associational", "input_sha256": sha256(args.analytic), "context_population_coverage": args.context_population_coverage, "primary_outcome": "in-hospital death", "primary_effect": "standardized marginal risk difference", "supportive_effects": ["marginal risk ratio", "conditional odds ratio"], "not_an_individual_prediction_tool": True}
    context_ok = args.context_population_coverage >= .95
    secondary_base = frame.loc[frame["trailing12_complete"].fillna(False)].copy()
    outcome_base = frame.loc[frame["death_valid"].fillna(False)].copy()
    base = outcome_base.loc[outcome_base["trailing12_complete"].fillna(False)].copy()
    complete_context = base[["ivs_context", "ans_context"]].notna().all(axis=1)
    if not context_ok and not args.allow_context_downgrade:
        qc.update({"status": "DOWNGRADE", "reason": "Context population coverage below 95%; primary complete-context analysis is not permitted"}); _write_json(args.output_dir / "aim4_model_qc_v2.json", qc); return 2
    if context_ok:
        base = base.loc[complete_context].copy(); allow_context = True
        secondary_base = secondary_base.loc[secondary_base[["ivs_context", "ans_context"]].notna().all(axis=1)].copy()
        outcome_base = outcome_base.loc[outcome_base[["ivs_context", "ans_context"]].notna().all(axis=1)].copy()
    else:
        allow_context = False; qc["context_downgrade"] = "context terms omitted; all resulting effects remain associational"
    qc["primary_model_n"] = int(len(base)); qc["secondary_candidate_n_before_secondary_endpoint_rules"] = int(len(secondary_base)); qc["death_events"] = int(pd.to_numeric(base["in_hospital_death"], errors="coerce").sum()); qc["death_endpoint_excluded_only"] = int((~frame["death_valid"].fillna(False)).sum())
    try:
        X, names, audit, age_knots = design_matrix(base, allow_context=allow_context); y = pd.to_numeric(base["in_hospital_death"], errors="raise").to_numpy(float); clusters = base["cnes7"].astype(str).to_numpy()
        fit = logistic_glm_clustered(X, y, clusters); fit["volume_knots"] = audit["volume_knots"]; contrast, curve = standardized_volume_contrast(base, X, names, fit)
    except ModelGateError as exc:
        qc.update({"status": "DOWNGRADE", "model_gate": str(exc)}); _write_json(args.output_dir / "aim4_model_qc_v2.json", qc); return 2
    cond = []
    for n, b, se in zip(names, fit["beta"], np.sqrt(np.maximum(np.diag(fit["cov"]), 0))):
        cond.append({"term": n, "conditional_odds_ratio": float(math.exp(np.clip(b, -30, 30))), "ci_low": float(math.exp(np.clip(b - Z975 * se, -30, 30))), "ci_high": float(math.exp(np.clip(b + Z975 * se, -30, 30))), "label": "conditional"})
    pred = base[["analysis_row_id", "cnes7", "state_provider", "calendar_month", "year", "trailing12_a_unique_aih", "diagnostic_stratum"]].copy(); pred["outcome"] = y; pred["fitted_risk"] = fit["mu"]
    pred.to_parquet(args.output_dir / "aim4_mortality_model_rows_v2.parquet", index=False); curve.to_parquet(args.output_dir / "aim4_mortality_volume_curve_v2.parquet", index=False); pd.DataFrame(cond).to_parquet(args.output_dir / "aim4_conditional_or_v2.parquet", index=False)
    model_object = args.output_dir / "aim4_mortality_model_object_v2.npz"
    np.savez_compressed(
        model_object,
        beta=np.asarray(fit["beta"], dtype=float),
        covariance=np.asarray(fit["cov"], dtype=float),
        design_names=np.asarray(names, dtype=str),
        age_knots=np.asarray(age_knots, dtype=float),
        volume_knots=np.asarray(audit["volume_knots"], dtype=float),
    )
    # Death validity defines only the mortality endpoint.  Secondary endpoints retain
    # records with an invalid death field when their own endpoint is observed.
    try:
        X_secondary, names_secondary, audit_secondary, _ = design_matrix(secondary_base, allow_context=allow_context)
        secondary = fit_secondary(secondary_base, X_secondary, secondary_base["cnes7"].astype(str).to_numpy(), {"volume_knots": audit_secondary["volume_knots"]}, names_secondary)
    except ModelGateError as exc:
        secondary = [{"endpoint": "all secondary endpoints", "status": "DOWNGRADE", "note": str(exc), "multiplicity": "secondary/exploratory"}]
    _write_json(args.output_dir / "aim4_secondary_outcomes_v2.json", {"multiplicity": "secondary/exploratory", "death_endpoint_only_exclusion": True, "results": secondary})
    secondary_exposures = fit_secondary_exposures(outcome_base, allow_context=allow_context)
    secondary_exposures.to_parquet(
        args.output_dir / "aim4_secondary_exposure_associations_v2.parquet", index=False
    )
    qc.update({"design_columns": names, "design_audit": audit, "design_rank": int(np.linalg.matrix_rank(X)), "design_parameters": int(X.shape[1]), "design_condition_number": float(np.linalg.cond(X)), "n_clusters": fit["n_clusters"], "events_per_parameter": float(min(y.sum(), len(y) - y.sum()) / X.shape[1]), "epv_below_10_warning": bool(min(y.sum(), len(y) - y.sum()) / X.shape[1] < 10), "primary_contrast": contrast, "secondary_exposure_status": dict(zip(secondary_exposures["exposure"], secondary_exposures["status"])) if not secondary_exposures.empty else {}, "model_object_sha256": sha256(model_object), "all_outputs_sha256": {p.name: sha256(p) for p in list(args.output_dir.glob("aim4_*_v2.parquet")) + [model_object]}})
    qc_file = args.output_dir / "aim4_model_qc_v2.json"; _write_json(qc_file, qc)
    (args.output_dir / "aim4_model_manifest_v2.json").write_text(json.dumps({"input": {args.analytic.name: sha256(args.analytic)}, "outputs": {p.name: sha256(p) for p in list(args.output_dir.glob("aim4_*_v2.parquet")) + [model_object, qc_file]}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
