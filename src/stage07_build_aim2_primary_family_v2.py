from __future__ import annotations

"""Freeze and analyse the prespecified three-endpoint Aim-2 IVS family."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def implementation_payload() -> dict:
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


def finite_result(fit) -> bool:
    return bool(
        np.isfinite(np.asarray(fit.params, dtype=float)).all()
        and np.isfinite(np.asarray(fit.bse, dtype=float)).all()
    )


def rank_standardised(fit, frame: pd.DataFrame, weights: np.ndarray, offset: np.ndarray) -> tuple[float, float]:
    low = frame.copy()
    high = frame.copy()
    low["ivs_ridit"] = 0.0
    high["ivs_ridit"] = 1.0
    return (
        float(np.average(fit.predict(low, offset=offset), weights=weights)),
        float(np.average(fit.predict(high, offset=offset), weights=weights)),
    )


def model_summary(fit, frame: pd.DataFrame, endpoint: str, effect_kind: str) -> dict:
    beta = float(fit.params["ivs_ridit"])
    se = float(fit.bse["ivs_ridit"])
    return {
        "endpoint": endpoint,
        "effect_kind": effect_kind,
        "n_rows": int(len(frame)),
        "n_municipalities": int(frame["res_municipio"].nunique()),
        "clusters": int(frame["res_municipio"].nunique()),
        "coefficient": beta,
        "cluster_robust_se": se,
        "ratio_rank1_vs_rank0": float(np.exp(beta)),
        "ratio_lo95": float(np.exp(beta - 1.96 * se)),
        "ratio_hi95": float(np.exp(beta + 1.96 * se)),
        "pvalue_raw": float(fit.pvalues["ivs_ridit"]),
        "converged": bool(fit.converged),
        "finite_estimates": finite_result(fit),
        "design_columns": int(fit.model.exog.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(fit.model.exog)),
        "pearson_chi2_over_df": float(fit.pearson_chi2 / fit.df_resid),
    }


def fit_utilisation(equity: pd.DataFrame) -> dict:
    frame = equity.dropna(subset=["ivs_ridit", "adult_population"]).copy()
    frame = frame[frame["adult_population"].gt(0)]
    fit = smf.glm(
        "n ~ ivs_ridit + C(year) + C(uf)", data=frame,
        family=sm.families.Poisson(), offset=np.log(frame["adult_population"]),
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["res_municipio"]})
    result = model_summary(fit, frame, "utilisation", "rate ratio")
    offset = np.repeat(np.log(100000.0), len(frame))
    low, high = rank_standardised(fit, frame, frame["adult_population"].to_numpy(), offset)
    result.update({
        "rank0_rate_per_100k": low,
        "rank1_rate_per_100k": high,
        "absolute_rank1_minus_rank0_per_100k": high - low,
        "n_events": int(frame["n"].sum()),
        "offset": "log adult population",
    })
    return result


def fit_travel(travel: pd.DataFrame, equity: pd.DataFrame) -> tuple[dict, dict]:
    contextual = equity[["year", "res_municipio", "ivs_ridit"]].copy()
    flow = travel.merge(contextual, on=["year", "res_municipio"], how="left", validate="many_to_one")
    structural_zero = flow[flow["travel_minutes"].eq(0)].copy()
    frame = flow.dropna(subset=["travel_minutes", "ivs_ridit"]).copy()
    frame = frame[frame["travel_minutes"].gt(0)]
    frame["uf"] = frame["res_municipio"].str[:2]
    fit = smf.glm(
        "travel_minutes ~ ivs_ridit + C(year) + C(uf)", data=frame,
        family=sm.families.Gamma(link=sm.families.links.Log()), freq_weights=frame["n_aih"],
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["res_municipio"]})
    result = model_summary(fit, frame, "travel_time", "mean time ratio")
    low, high = rank_standardised(
        fit, frame, frame["n_aih"].to_numpy(dtype=float), np.zeros(len(frame))
    )
    result.update({
        "rank0_mean_minutes": low,
        "rank1_mean_minutes": high,
        "absolute_rank1_minus_rank0_minutes": high - low,
        "n_aih_positive_time_model": int(frame["n_aih"].sum()),
        "structural_zero_within_municipality_n_aih": int(structural_zero["n_aih"].sum()),
        "model": "Gamma(log), n_aih frequency weights; only positive road times",
    })
    return result, {
        "n_pair_years_total": int(len(flow)),
        "n_pair_years_positive_time_model": int(len(frame)),
        "n_aih_total": int(flow["n_aih"].sum()),
        "n_aih_positive_time_model": int(frame["n_aih"].sum()),
        "n_aih_structural_zero": int(structural_zero["n_aih"].sum()),
        "n_aih_missing_route_or_context": int(flow.loc[flow["travel_minutes"].isna() | flow["ivs_ridit"].isna(), "n_aih"].sum()),
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
    fit = smf.glm(
        "covered120 ~ ivs_ridit + C(year) + C(uf)", data=model_frame,
        family=sm.families.Poisson(), freq_weights=model_frame["adult_population"],
    ).fit(cov_type="cluster", cov_kwds={"groups": model_frame["res_municipio"]})
    result = model_summary(fit, model_frame, "potential_120min_coverage", "risk ratio")
    low, high = rank_standardised(
        fit, model_frame, model_frame["adult_population"].to_numpy(dtype=float), np.zeros(len(model_frame))
    )
    predicted = fit.predict(model_frame)
    result.update({
        "rank0_coverage_probability": low,
        "rank1_coverage_probability": high,
        "absolute_rank1_minus_rank0_probability": high - low,
        "predicted_probability_gt1_rows": int((predicted > 1).sum()),
        "adult_population_model": float(model_frame["adult_population"].sum()),
        "adult_population_total": float(frame["adult_population"].sum()),
        "model_population_coverage": float(model_frame["adult_population"].sum() / frame["adult_population"].sum()),
        "model": "modified Poisson with adult-population frequency weights",
    })
    return result, {
        "n_municipality_years_total": int(len(frame)),
        "n_municipality_years_model": int(len(model_frame)),
        "unknown_anchor_municipality_years": int(frame["has_provider_120"].isna().sum()),
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
    checks = {
        "freeze_payload_valid": True,
        "all_models_converged": all(item["converged"] for item in results),
        "all_estimates_finite": all(item["finite_estimates"] for item in results),
        "holm_manual_equals_statsmodels": holm_equal,
        "exactly_three_primary_tests": len(raw) == 3,
        "coverage_probability_not_above_one": coverage_result["predicted_probability_gt1_rows"] == 0,
    }
    status = "PASS" if all(checks.values()) else "DOWNGRADE"
    audit = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": status,
        "freeze_file": str(args.freeze),
        "freeze_payload_sha256": frozen["freeze_payload_sha256"],
        "checks": checks,
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
