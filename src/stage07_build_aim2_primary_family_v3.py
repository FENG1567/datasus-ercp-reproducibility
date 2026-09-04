from __future__ import annotations

"""Freeze and analyse the prespecified three-endpoint Aim-2 IVS family.

This v3 report-layer implementation intentionally retains the frozen v2 model
families and estimands.  Its additions are delta-method uncertainty for
rank-standardised absolute effects and explicit fit/diagnostic gates.
"""

import argparse
import hashlib
import json
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests


Z_975 = 1.959963984540054
WEIGHTED_CLUSTER_WARNING = re.compile(
    r"(?:freq(?:uency)?[_ ]weights?|weights?).*(?:cov(?:ariance|_type)?|cluster)|"
    r"(?:cov(?:ariance|_type)?|cluster).*(?:freq(?:uency)?[_ ]weights?|weights?)",
    flags=re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def implementation_payload() -> dict:
    """Return the v2 statistical implementation verbatim for freeze compatibility."""
    return {
        "schema_version": "1.0",
        "family_name": "Aim2 prespecified IVS gradient family",
        "primary_exposure": {
            "name": "IVS 2010 municipal contextual vulnerability ridit",
            "ranking": "Fixed 2022 official adult-population-weighted ridit; no contextual-value imputation",
            "interpretation": "Ecological/contextual, not an individual characteristic",
        },
        "endpoints": [
            {
                "id": "utilisation",
                "estimand": "Association of IVS ridit with cohort-B treated ERCP utilisation per adult resident",
                "analysis": "Municipality-year Poisson log-link with log(adult population) offset, year and residence-state fixed effects, municipality-clustered sandwich covariance",
                "effect": "RII rate ratio, rank-0-to-rank-1 standardised SII rate difference per 100,000, 95% CI, raw IVS-ridit p value",
            },
            {
                "id": "travel_time",
                "estimand": "Association of IVS ridit with realised treated-flow road time among positive cross-municipality cohort-B flows",
                "analysis": "Pair-year Gamma log-link with n_aih frequency weights, year and residence-state fixed effects, residence-municipality-clustered sandwich covariance; structural zero within-municipality flows described separately",
                "effect": "Rank-1/rank-0 mean-time ratio, model-standardised absolute mean-time difference, 95% CI, raw IVS-ridit p value",
            },
            {
                "id": "potential_120min_coverage",
                "estimand": "Association of IVS ridit with municipal potential 120-minute access to an observed performing municipality",
                "analysis": "Municipality-year modified Poisson log-link for binary coverage with adult-population frequency weights, year and residence-state fixed effects, municipality-clustered sandwich covariance",
                "effect": "Rank-1/rank-0 potential-coverage RR and model-standardised risk difference, 95% CI, raw IVS-ridit p value",
            },
        ],
        "multiplicity": {
            "family": ["utilisation", "travel_time", "potential_120min_coverage"],
            "procedure": "Holm step-down across exactly the three raw IVS-ridit p values",
            "verification": "manual Holm implementation must equal statsmodels.multipletests(method='holm') within 1e-12",
        },
        "evidence_language": "All endpoints are descriptive or associational; coverage is potential access and travel time is realised treated flow, not referral or causal evidence.",
    }


def write_freeze(path: Path) -> int:
    payload = implementation_payload()
    document = {
        "created_at": utc_now(),
        "freeze_payload_sha256": canonical_hash(payload),
        "implementation": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0


def validate_freeze(path: Path) -> dict:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    expected = canonical_hash(implementation_payload())
    if frozen.get("freeze_payload_sha256") != expected:
        raise RuntimeError("statistical implementation freeze payload hash does not match this implementation")
    return frozen


def finite_array(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=float)).all())
    except (TypeError, ValueError):
        return False


def warning_records(records: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [{"category": item.category.__name__, "message": str(item.message)} for item in records]


def has_unsupported_weighted_cluster_warning(records: list[dict[str, str]]) -> bool:
    """Return true for a statsmodels warning that invalidates weighted cluster SEs."""
    return any(WEIGHTED_CLUSTER_WARNING.search(item["message"]) is not None for item in records)


class ClusteredGLMResult:
    """Small result adapter carrying an explicit score-based cluster sandwich."""

    def __init__(self, base_fit, covariance: np.ndarray, covariance_audit: dict[str, Any]):
        self._base_fit = base_fit
        self._covariance = pd.DataFrame(
            covariance, index=base_fit.params.index, columns=base_fit.params.index
        )
        self.params = base_fit.params
        self.bse = pd.Series(
            np.sqrt(np.maximum(np.diag(covariance), 0.0)), index=base_fit.params.index
        )
        z_values = self.params / self.bse.replace(0, np.nan)
        self.pvalues = z_values.abs().map(lambda value: math.erfc(value / math.sqrt(2.0)))
        self.converged = bool(base_fit.converged)
        self.model = base_fit.model
        self.pearson_chi2 = float(base_fit.pearson_chi2)
        self.df_resid = float(base_fit.df_resid)
        self.covariance_audit = covariance_audit

    def cov_params(self):
        return self._covariance

    def predict(self, frame: pd.DataFrame):
        return self._base_fit.predict(frame)


def score_cluster_covariance(base_fit, groups: pd.Series, effective_n: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster sandwich from GLM score contributions and observed information."""
    group_values = np.asarray(groups)
    if len(group_values) != len(base_fit.model.endog):
        raise ValueError("cluster vector length does not match fitted rows")
    scale = float(base_fit.scale)
    score_obs = np.asarray(base_fit.model.score_obs(base_fit.params, scale=scale), dtype=float)
    hessian = np.asarray(
        base_fit.model.hessian(base_fit.params, scale=scale, observed=True), dtype=float
    )
    if score_obs.shape[0] != len(group_values) or score_obs.shape[1] != len(base_fit.params):
        raise RuntimeError("unexpected GLM score-observation dimensions")
    bread = np.linalg.pinv(-hessian)
    unique_groups = pd.unique(group_values)
    meat = np.zeros((len(base_fit.params), len(base_fit.params)), dtype=float)
    for group in unique_groups:
        cluster_score = score_obs[group_values == group].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)
    n_parameters = len(base_fit.params)
    if len(unique_groups) <= 1 or effective_n <= n_parameters:
        raise RuntimeError("cluster sandwich requires >1 cluster and effective n > parameters")
    correction = (len(unique_groups) / (len(unique_groups) - 1.0)) * (
        (effective_n - 1.0) / (effective_n - n_parameters)
    )
    covariance = correction * bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2.0
    audit = {
        "method": "explicit GLM score-cluster sandwich",
        "n_clusters": int(len(unique_groups)),
        "effective_n_for_small_sample_correction": float(effective_n),
        "small_sample_correction": float(correction),
        "scale": scale,
        "score_rows": int(score_obs.shape[0]),
        "finite": finite_array(covariance),
    }
    if not audit["finite"]:
        raise RuntimeError("explicit cluster sandwich covariance is nonfinite")
    return covariance, audit


def fit_glm_checked(*, formula: str, data: pd.DataFrame, family, groups: pd.Series, freq_weights=None, offset=None):
    """Fit the frozen GLM with an explicit score-based cluster sandwich."""
    kwargs: dict[str, Any] = {"family": family}
    if freq_weights is not None:
        kwargs["freq_weights"] = freq_weights
    if offset is not None:
        kwargs["offset"] = offset
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base_fit = smf.glm(formula, data=data, **kwargs).fit()
    records = warning_records(caught)
    effective_n = (
        float(np.asarray(freq_weights, dtype=float).sum())
        if freq_weights is not None
        else float(len(data))
    )
    covariance, covariance_audit = score_cluster_covariance(base_fit, groups, effective_n)
    fit = ClusteredGLMResult(base_fit, covariance, covariance_audit)
    return fit, records


def rank_design_matrix(fit, frame: pd.DataFrame, rank: float) -> np.ndarray:
    """Rebuild the fitted formula design matrix after setting IVS ridit to rank."""
    ranked = frame.copy()
    ranked["ivs_ridit"] = float(rank)
    design_info = fit.model.data.design_info
    matrix = build_design_matrices([design_info], ranked, return_type="dataframe")[0]
    result = np.asarray(matrix, dtype=float)
    if result.shape[1] != len(fit.params):
        raise RuntimeError("rank-standardisation design matrix does not match fitted parameter dimension")
    return result


def rank_standardised_log_delta(
    *,
    params: np.ndarray,
    covariance: np.ndarray,
    x_rank0: np.ndarray,
    x_rank1: np.ndarray,
    weights: np.ndarray,
    offset: np.ndarray,
) -> dict[str, float | bool]:
    """Standardise two log-link means and obtain the absolute-effect delta CI.

    For a log link, d mean(mu)/d beta is the standardisation-weighted mean of
    ``mu_i X_i``.  The reported absolute contrast uses g1 - g0 and the fitted
    cluster-robust covariance matrix supplied by the caller.
    """
    params = np.asarray(params, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    x_rank0 = np.asarray(x_rank0, dtype=float)
    x_rank1 = np.asarray(x_rank1, dtype=float)
    weights = np.asarray(weights, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if x_rank0.shape != x_rank1.shape or x_rank0.shape[0] != len(weights):
        raise ValueError("rank-standardisation dimensions are inconsistent")
    if x_rank0.shape[1] != len(params) or covariance.shape != (len(params), len(params)):
        raise ValueError("parameter or covariance dimensions are inconsistent")
    numeric_inputs = (x_rank0, x_rank1, weights, offset, params, covariance)
    if len(offset) != len(weights) or not all(finite_array(value) for value in numeric_inputs):
        raise ValueError("rank-standardisation inputs must be finite")
    if np.any(weights < 0) or not float(weights.sum()) > 0:
        raise ValueError("standardisation weights must be non-negative with positive total")
    eta0 = x_rank0 @ params + offset
    eta1 = x_rank1 @ params + offset
    # The frozen models all have log links.  Explicitly calculate rather than
    # relying on predict so gradients and point estimates share one code path.
    mu0 = np.exp(eta0)
    mu1 = np.exp(eta1)
    mean0 = float(np.average(mu0, weights=weights))
    mean1 = float(np.average(mu1, weights=weights))
    grad0 = np.average(mu0[:, None] * x_rank0, axis=0, weights=weights)
    grad1 = np.average(mu1[:, None] * x_rank1, axis=0, weights=weights)
    gradient = np.asarray(grad1 - grad0, dtype=float)
    variance = float(gradient @ covariance @ gradient)
    # A tiny negative value can arise from floating-point rounding of a PSD
    # covariance; a material negative value is an invalid covariance gate.
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(covariance))))
    variance_valid = bool(np.isfinite(variance) and variance >= -tolerance)
    variance_for_se = max(variance, 0.0) if variance_valid else np.nan
    se = float(np.sqrt(variance_for_se)) if np.isfinite(variance_for_se) else np.nan
    difference = float(mean1 - mean0)
    ci_low = float(difference - Z_975 * se) if np.isfinite(se) else np.nan
    ci_high = float(difference + Z_975 * se) if np.isfinite(se) else np.nan
    return {
        "rank0_mean": mean0,
        "rank1_mean": mean1,
        "absolute_rank1_minus_rank0": difference,
        "absolute_difference_delta_se": se,
        "absolute_difference_ci_low": ci_low,
        "absolute_difference_ci_high": ci_high,
        "absolute_difference_gradient_finite": finite_array(gradient),
        "absolute_difference_variance": variance,
        "absolute_difference_variance_valid": variance_valid,
        "absolute_difference_ci_finite": finite_array([ci_low, ci_high]),
    }


def rank_standardised_delta(fit, frame: pd.DataFrame, weights: np.ndarray, offset: np.ndarray) -> dict[str, float | bool]:
    return rank_standardised_log_delta(
        params=np.asarray(fit.params, dtype=float),
        covariance=np.asarray(fit.cov_params(), dtype=float),
        x_rank0=rank_design_matrix(fit, frame, 0.0),
        x_rank1=rank_design_matrix(fit, frame, 1.0),
        weights=weights,
        offset=offset,
    )


def model_summary(
    fit,
    frame: pd.DataFrame,
    endpoint: str,
    effect_kind: str,
    fit_warnings: list[dict[str, str]],
) -> dict:
    covariance = np.asarray(fit.cov_params(), dtype=float)
    beta = float(fit.params["ivs_ridit"])
    se = float(fit.bse["ivs_ridit"])
    ratio_low = float(np.exp(beta - Z_975 * se))
    ratio_high = float(np.exp(beta + Z_975 * se))
    design = np.asarray(fit.model.exog, dtype=float)
    pearson_ratio = float(fit.pearson_chi2 / fit.df_resid) if fit.df_resid > 0 else np.nan
    return {
        "endpoint": endpoint,
        "effect_kind": effect_kind,
        "n_rows": int(len(frame)),
        "n_municipalities": int(frame["res_municipio"].nunique()),
        "clusters": int(frame["res_municipio"].nunique()),
        "coefficient": beta,
        "cluster_robust_se": se,
        "ratio_rank1_vs_rank0": float(np.exp(beta)),
        "ratio_lo95": ratio_low,
        "ratio_hi95": ratio_high,
        "pvalue_raw": float(fit.pvalues["ivs_ridit"]),
        "converged": bool(fit.converged),
        "finite_estimates": finite_array([fit.params, fit.bse]),
        "finite_covariance": finite_array(covariance),
        "finite_ratio_ci": finite_array([ratio_low, ratio_high]),
        "design_columns": int(design.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_full_rank": bool(np.linalg.matrix_rank(design) == design.shape[1]),
        "pearson_chi2_over_df": pearson_ratio,
        "finite_pearson_chi2_over_df": bool(np.isfinite(pearson_ratio)),
        "fit_warnings": json.dumps(fit_warnings, ensure_ascii=False),
        "cluster_covariance_audit": json.dumps(fit.covariance_audit, ensure_ascii=False),
        "weighted_cluster_covariance_unsupported": has_unsupported_weighted_cluster_warning(fit_warnings),
    }


def add_delta_fields(result: dict, delta: dict, *, stem: str, unit: str) -> None:
    result.update({
        f"rank0_{stem}": float(delta["rank0_mean"]),
        f"rank1_{stem}": float(delta["rank1_mean"]),
        f"absolute_rank1_minus_rank0_{unit}": float(delta["absolute_rank1_minus_rank0"]),
        f"absolute_rank1_minus_rank0_{unit}_delta_se": float(delta["absolute_difference_delta_se"]),
        f"absolute_rank1_minus_rank0_{unit}_ci_low95": float(delta["absolute_difference_ci_low"]),
        f"absolute_rank1_minus_rank0_{unit}_ci_high95": float(delta["absolute_difference_ci_high"]),
        "absolute_difference_gradient_finite": bool(delta["absolute_difference_gradient_finite"]),
        "absolute_difference_variance": float(delta["absolute_difference_variance"]),
        "absolute_difference_variance_valid": bool(delta["absolute_difference_variance_valid"]),
        "absolute_difference_ci_finite": bool(delta["absolute_difference_ci_finite"]),
    })


def fit_utilisation(equity: pd.DataFrame) -> dict:
    frame = equity.dropna(subset=["ivs_ridit", "adult_population"]).copy()
    frame = frame[frame["adult_population"].gt(0)]
    fit, captured = fit_glm_checked(
        formula="n ~ ivs_ridit + C(year) + C(uf)", data=frame,
        family=sm.families.Poisson(), groups=frame["res_municipio"],
        offset=np.log(frame["adult_population"]),
    )
    result = model_summary(fit, frame, "utilisation", "rate ratio", captured)
    delta = rank_standardised_delta(
        fit, frame, frame["adult_population"].to_numpy(dtype=float), np.repeat(np.log(100000.0), len(frame))
    )
    add_delta_fields(result, delta, stem="rate_per_100k", unit="per_100k")
    result.update({"n_events": int(frame["n"].sum()), "offset": "log adult population"})
    return result


def fit_travel(travel: pd.DataFrame, equity: pd.DataFrame) -> tuple[dict, dict]:
    contextual = equity[["year", "res_municipio", "ivs_ridit"]].copy()
    flow = travel.merge(contextual, on=["year", "res_municipio"], how="left", validate="many_to_one")
    structural_zero = flow[flow["travel_minutes"].eq(0)].copy()
    frame = flow.dropna(subset=["travel_minutes", "ivs_ridit"]).copy()
    frame = frame[frame["travel_minutes"].gt(0)]
    frame["uf"] = frame["res_municipio"].str[:2]
    fit, captured = fit_glm_checked(
        formula="travel_minutes ~ ivs_ridit + C(year) + C(uf)", data=frame,
        family=sm.families.Gamma(link=sm.families.links.Log()), groups=frame["res_municipio"],
        freq_weights=frame["n_aih"],
    )
    result = model_summary(fit, frame, "travel_time", "mean time ratio", captured)
    delta = rank_standardised_delta(
        fit, frame, frame["n_aih"].to_numpy(dtype=float), np.zeros(len(frame), dtype=float)
    )
    add_delta_fields(result, delta, stem="mean_minutes", unit="minutes")
    result.update({
        "n_aih_positive_time_model": int(frame["n_aih"].sum()),
        "structural_zero_within_municipality_n_aih": int(structural_zero["n_aih"].sum()),
        "model": "Gamma(log), n_aih frequency weights; positive realised treated-flow road times only",
    })
    return result, {
        "n_pair_years_total": int(len(flow)),
        "n_pair_years_positive_time_model": int(len(frame)),
        "n_aih_total": int(flow["n_aih"].sum()),
        "n_aih_positive_time_model": int(frame["n_aih"].sum()),
        "n_aih_structural_zero": int(structural_zero["n_aih"].sum()),
        "n_aih_missing_route_or_context": int(flow.loc[flow["travel_minutes"].isna() | flow["ivs_ridit"].isna(), "n_aih"].sum()),
        "travel_interpretation": "realised treated flow, not referral or direct patient travel",
    }


def fit_coverage(coverage: pd.DataFrame, equity: pd.DataFrame) -> tuple[dict, dict]:
    contextual = equity[["year", "res_municipio", "ivs_ridit"]].copy()
    frame = coverage.rename(columns={"municipio": "res_municipio"}).merge(
        contextual, on=["year", "res_municipio"], how="left", validate="one_to_one"
    )
    model_frame = frame.dropna(subset=["has_provider_120", "ivs_ridit", "adult_population"]).copy()
    model_frame = model_frame[model_frame["adult_population"].gt(0)]
    model_frame["covered120"] = model_frame["has_provider_120"].astype(int)
    model_frame["uf"] = model_frame["res_municipio"].str[:2]
    fit, captured = fit_glm_checked(
        formula="covered120 ~ ivs_ridit + C(year) + C(uf)", data=model_frame,
        family=sm.families.Poisson(), groups=model_frame["res_municipio"],
        freq_weights=model_frame["adult_population"],
    )
    result = model_summary(fit, model_frame, "potential_120min_coverage", "risk ratio", captured)
    delta = rank_standardised_delta(
        fit, model_frame, model_frame["adult_population"].to_numpy(dtype=float), np.zeros(len(model_frame), dtype=float)
    )
    add_delta_fields(result, delta, stem="coverage_probability", unit="probability")
    result["absolute_rank1_minus_rank0_percentage_points"] = (
        100.0 * result["absolute_rank1_minus_rank0_probability"]
    )
    result["absolute_rank1_minus_rank0_percentage_points_ci_low95"] = (
        100.0 * result["absolute_rank1_minus_rank0_probability_ci_low95"]
    )
    result["absolute_rank1_minus_rank0_percentage_points_ci_high95"] = (
        100.0 * result["absolute_rank1_minus_rank0_probability_ci_high95"]
    )
    predicted = fit.predict(model_frame)
    total_adult_population = float(frame["adult_population"].sum())
    model_adult_population = float(model_frame["adult_population"].sum())
    model_coverage = model_adult_population / total_adult_population if total_adult_population > 0 else np.nan
    result.update({
        "predicted_probability_gt1_rows": int((predicted > 1).sum()),
        "adult_population_model": model_adult_population,
        "adult_population_total": total_adult_population,
        "model_population_coverage": model_coverage,
        "model_population_coverage_ge_95pct": bool(
            np.isfinite(model_coverage) and model_coverage >= 0.95
        ),
        "model": "modified Poisson with adult-population frequency weights; potential access only",
    })
    return result, {
        "n_municipality_years_total": int(len(frame)),
        "n_municipality_years_model": int(len(model_frame)),
        "unknown_anchor_municipality_years": int(frame["has_provider_120"].isna().sum()),
        "adult_population_total": total_adult_population,
        "adult_population_model": model_adult_population,
        "model_population_coverage": model_coverage,
    }


def holm_manual(pvalues: np.ndarray) -> np.ndarray:
    count = len(pvalues)
    order = np.argsort(pvalues)
    corrected = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * pvalues[index]))
        corrected[index] = running
    return corrected


def item_model_checks(item: dict) -> dict[str, bool]:
    return {
        "converged": bool(item["converged"]),
        "finite_estimates": bool(item["finite_estimates"]),
        "finite_covariance": bool(item["finite_covariance"]),
        "finite_ratio_ci": bool(item["finite_ratio_ci"]),
        "design_full_rank": bool(item["design_full_rank"]),
        "clusters_gt_one": int(item["clusters"]) > 1,
        "finite_pearson_chi2_over_df": bool(item["finite_pearson_chi2_over_df"]),
        "absolute_difference_variance_valid": bool(item["absolute_difference_variance_valid"]),
        "absolute_difference_ci_finite": bool(item["absolute_difference_ci_finite"]),
        "weighted_cluster_covariance_supported": not bool(item["weighted_cluster_covariance_unsupported"]),
        "raw_ivs_ridit_pvalue_finite": bool(np.isfinite(item["pvalue_raw"])),
    }


def analyse(args) -> int:
    frozen = validate_freeze(args.freeze)
    equity = pd.read_parquet(args.equity)
    travel = pd.read_parquet(args.travel)
    coverage = pd.read_parquet(args.coverage)
    utilisation = fit_utilisation(equity)
    travel_result, travel_qc = fit_travel(travel, equity)
    coverage_result, coverage_qc = fit_coverage(coverage, equity)
    results = [utilisation, travel_result, coverage_result]
    raw = np.asarray([item["pvalue_raw"] for item in results], dtype=float)
    manual = holm_manual(raw)
    _, statsmodels_holm, _, _ = multipletests(raw, method="holm")
    holm_equal = bool(np.allclose(manual, statsmodels_holm, rtol=0, atol=1e-12))
    for item, adjusted in zip(results, manual):
        item["pvalue_holm"] = float(adjusted)
    table = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    per_model_checks = {item["endpoint"]: item_model_checks(item) for item in results}
    checks = {
        "freeze_payload_valid": True,
        "all_model_checks_pass": all(all(values.values()) for values in per_model_checks.values()),
        "holm_manual_equals_statsmodels": holm_equal,
        "exactly_three_primary_tests": len(raw) == 3,
        "coverage_probability_not_above_one": coverage_result["predicted_probability_gt1_rows"] == 0,
        "coverage_model_population_coverage_ge_95pct": coverage_result[
            "model_population_coverage_ge_95pct"
        ],
    }
    status = "PASS" if all(checks.values()) else "DOWNGRADE"
    audit = {
        "schema_version": "1.1",
        "generated_at": utc_now(),
        "status": status,
        "freeze_file": str(args.freeze),
        "freeze_payload_sha256": frozen["freeze_payload_sha256"],
        "checks": checks,
        "per_model_checks": per_model_checks,
        "primary_family": [item["endpoint"] for item in results],
        "raw_pvalues": raw.tolist(),
        "holm_manual": manual.tolist(),
        "holm_statsmodels": statsmodels_holm.tolist(),
        "travel_qc": travel_qc,
        "coverage_qc": coverage_qc,
        "evidence_language": frozen["implementation"]["evidence_language"],
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-freeze", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--equity", type=Path)
    parser.add_argument("--travel", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.write_freeze is not None:
        return write_freeze(args.write_freeze)
    required = [args.freeze, args.equity, args.travel, args.coverage, args.output, args.audit]
    if any(item is None for item in required):
        parser.error("analysis requires --freeze --equity --travel --coverage --output --audit")
    return analyse(args)


if __name__ == "__main__":
    raise SystemExit(main())
