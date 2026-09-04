from __future__ import annotations

"""Discrete-time model of first observed coded ERCP use among eligible hospitals."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from patsy import dmatrix


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_map_from_st(
    path: Path, requested_keys: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Read states from the actual monthly CNES-ST partitions.

    A prior convenience parquet was incorrectly stamped entirely as 202501.
    This reader instead joins only requested eligible hospital-months to their
    original month-specific CNES-ST partition and its CODUFMUN field.
    """
    requested = requested_keys[["CNES", "competence_month"]].copy()
    requested["CNES"] = requested["CNES"].astype(str).str.zfill(7)
    requested["competence_month"] = requested["competence_month"].astype(str)
    requested = requested.drop_duplicates()
    if not path.is_dir():
        raise RuntimeError(
            f"--st must be the monthly CNES-ST parquet directory, not {path}; "
            "a single merged convenience map cannot establish month-specific state"
        )
    fragments: list[pd.DataFrame] = []
    source_files: list[Path] = []
    for competence_month, group in requested.groupby("competence_month", sort=True):
        year = competence_month[:4]
        files = sorted((path / year).glob(f"*_{competence_month}.parquet"))
        if len(files) != 27:
            raise RuntimeError(
                f"CNES-ST partition inventory incomplete for {competence_month}: "
                f"found {len(files)}, expected 27"
            )
        requested_cnes = set(group["CNES"])
        for file in files:
            partition = pq.read_table(
                file, columns=["CNES", "CODUFMUN", "competence_month"]
            ).to_pandas()
            partition["CNES"] = partition["CNES"].astype(str).str.zfill(7)
            partition["competence_month"] = partition["competence_month"].astype(str)
            if not partition["competence_month"].eq(competence_month).all():
                raise RuntimeError(
                    f"CNES-ST competence_month mismatch inside partition {file}"
                )
            retained = partition.loc[
                partition["CNES"].isin(requested_cnes),
                ["CNES", "competence_month", "CODUFMUN"],
            ]
            if not retained.empty:
                fragments.append(retained)
            source_files.append(file)
    if not fragments:
        raise RuntimeError("monthly CNES-ST partitions returned no requested hospital-months")
    st = pd.concat(fragments, ignore_index=True)
    raw_rows = len(st)
    st["hosp_uf"] = st["CODUFMUN"].astype(str).str.zfill(6).str[:2]
    st = st[["CNES", "competence_month", "hosp_uf"]]
    conflicts = (
        st.groupby(["CNES", "competence_month"], dropna=False)["hosp_uf"]
        .nunique(dropna=False)
        .gt(1)
        .sum()
    )
    if conflicts:
        raise RuntimeError(
            f"CNES-ST map has {int(conflicts)} CNES-month keys with conflicting CODUFMUN states"
        )
    deduplicated = st.drop_duplicates(
        ["CNES", "competence_month"]
    )
    audit = {
        "source_path": str(path),
        "source_columns": ["CNES", "competence_month", "CODUFMUN"],
        "source_type": "monthly CNES-ST parquet partitions",
        "requested_eligible_cnes_month_keys": int(len(requested)),
        "partition_files_read": int(len(source_files)),
        "partition_months_read": int(requested["competence_month"].nunique()),
        "raw_rows": int(raw_rows),
        "unique_cnes_month_keys": int(len(deduplicated)),
        "duplicate_cnes_month_rows_collapsed": int(raw_rows - len(deduplicated)),
        "conflicting_cnes_month_states": int(conflicts),
        "state_derivation": "monthly CNES-ST CODUFMUN, never CNES identifier prefix",
    }
    return deduplicated, audit


def build_risk_set(
    eligible: pd.DataFrame,
    hospital_month: pd.DataFrame,
    st: pd.DataFrame,
    eligibility_column: str,
) -> tuple[pd.DataFrame, dict]:
    risk = eligible[eligible[eligibility_column].astype(bool)].copy()
    risk["CNES"] = risk["CNES"].astype(str).str.zfill(7)
    risk["competence_month"] = risk["competence_month"].astype(str)
    first = (
        hospital_month[["SP_CNES", "first_observed_month"]]
        .drop_duplicates("SP_CNES")
        .rename(columns={"SP_CNES": "CNES"})
    )
    first["CNES"] = first["CNES"].astype(str).str.zfill(7)
    risk = risk.merge(first, on="CNES", how="left", validate="many_to_one")
    left_censored = set(
        first.loc[first["first_observed_month"].eq("202101"), "CNES"]
    )
    risk = risk[~risk["CNES"].isin(left_censored)].copy()
    risk = risk[
        risk["first_observed_month"].isna()
        | risk["competence_month"].le(risk["first_observed_month"])
    ].copy()
    risk["event"] = risk["competence_month"].eq(risk["first_observed_month"]).astype(int)
    risk = risk.merge(st, on=["CNES", "competence_month"], how="left", validate="many_to_one")
    risk["month_int"] = risk["competence_month"].astype(int)
    risk["month_index"] = (
        (risk["month_int"] // 100 - 2021) * 12 + (risk["month_int"] % 100) - 1
    )

    event_by_state_month = (
        risk[risk["event"].eq(1)]
        .groupby(["hosp_uf", "month_index"], as_index=False)
        .size()
        .rename(columns={"size": "n_first_observed"})
    )
    state_month = risk[["hosp_uf", "month_index"]].drop_duplicates().sort_values(
        ["hosp_uf", "month_index"]
    )
    state_month = state_month.merge(
        event_by_state_month,
        on=["hosp_uf", "month_index"],
        how="left",
        validate="one_to_one",
    )
    state_month["n_first_observed"] = state_month["n_first_observed"].fillna(0)
    state_month["lag_cumulative_first_observed_state"] = (
        state_month.groupby("hosp_uf")["n_first_observed"].cumsum()
        - state_month["n_first_observed"]
    )
    risk = risk.merge(
        state_month[["hosp_uf", "month_index", "lag_cumulative_first_observed_state"]],
        on=["hosp_uf", "month_index"],
        how="left",
        validate="many_to_one",
    )
    metadata = {
        "eligibility_column": eligibility_column,
        "n_left_censored_202101_excluded": len(left_censored),
        "n_hospital_months": len(risk),
        "n_hospitals": risk["CNES"].nunique(),
        "n_events": int(risk["event"].sum()),
        "events_with_missing_state": int(
            risk.loc[risk["event"].eq(1), "hosp_uf"].isna().sum()
        ),
        "prevalent_at_window_start_in_risk": int(
            risk["CNES"].isin(left_censored).sum()
        ),
        "event_count_gt_one_hospitals": int(
            risk.groupby("CNES")["event"].sum().gt(1).sum()
        ),
        "post_first_observed_rows": int(
            (
                risk["first_observed_month"].notna()
                & risk["competence_month"].gt(risk["first_observed_month"])
            ).sum()
        ),
    }
    return risk, metadata


def fit_model(
    risk: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    model_frame = risk.dropna(subset=["hosp_uf"]).copy()
    state_diagnostics = (
        model_frame.groupby("hosp_uf", as_index=False)
        .agg(
            hospital_months=("event", "size"),
            hospitals=("CNES", "nunique"),
            first_observed_events=("event", "sum"),
        )
        .sort_values("hosp_uf")
    )
    state_diagnostics["risk_set"] = label
    state_diagnostics["zero_event_state"] = state_diagnostics[
        "first_observed_events"
    ].eq(0)
    zero_event_states = set(
        state_diagnostics.loc[state_diagnostics["zero_event_state"], "hosp_uf"]
    )
    # A state fixed effect with no events is completely separated in a
    # discrete-time hazard model. Such state-months remain in cohort and
    # descriptive outputs but cannot contribute to a finite conditional-FE
    # likelihood, so they are excluded deterministically from this model only.
    model_frame = model_frame[~model_frame["hosp_uf"].isin(zero_event_states)].copy()
    if model_frame.empty or not model_frame["event"].any():
        raise RuntimeError(f"{label}: no event-bearing states remain for fixed-effect model")
    model_frame["log_beds_sus"] = np.log1p(
        pd.to_numeric(model_frame["beds_sus"], errors="coerce").fillna(0)
    )
    model_frame["has_endoscopy_service"] = model_frame["endoscopy_service"].astype(int)
    model_frame["log_endoscopists"] = np.log1p(
        pd.to_numeric(model_frame["endoscopists"], errors="coerce").fillna(0)
    )
    model_frame["log_anesthesiologists"] = np.log1p(
        pd.to_numeric(model_frame["anesthesiologists"], errors="coerce").fillna(0)
    )
    model_frame["log_medical_cbos"] = np.log1p(
        pd.to_numeric(model_frame["medical_cbos"], errors="coerce").fillna(0)
    )
    spline = dmatrix(
        "bs(month_index, df=4, degree=3, include_intercept=False) - 1",
        model_frame,
        return_type="dataframe",
    )
    spline.columns = [f"time_spline_{index + 1}" for index in range(spline.shape[1])]
    numeric = model_frame[
        [
            "log_beds_sus",
            "has_endoscopy_service",
            "log_endoscopists",
            "log_anesthesiologists",
            "log_medical_cbos",
            "lag_cumulative_first_observed_state",
        ]
    ].reset_index(drop=True)
    state = pd.get_dummies(model_frame["hosp_uf"], prefix="uf", drop_first=True, dtype=float).reset_index(
        drop=True
    )
    design = pd.concat([numeric, spline.reset_index(drop=True), state], axis=1).astype(float)
    dropped_constant = [column for column in design if design[column].nunique(dropna=False) <= 1]
    design = design.drop(columns=dropped_constant)
    design.insert(0, "intercept", 1.0)
    outcome = model_frame["event"].to_numpy(dtype=int)
    design_rank = int(np.linalg.matrix_rank(design.to_numpy()))
    cluster_count = int(model_frame["CNES"].nunique())
    if design_rank != design.shape[1]:
        raise RuntimeError(
            f"{label}: non-full-rank design matrix rank={design_rank}, columns={design.shape[1]}"
        )
    if cluster_count < 2:
        raise RuntimeError(f"{label}: fewer than two hospital clusters for robust inference")
    model = sm.GLM(
        outcome,
        design,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    )
    fit = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": model_frame["CNES"].to_numpy()},
        maxiter=300,
    )
    coefficients = pd.DataFrame(
        {
            "risk_set": label,
            "term": design.columns,
            "coef": fit.params,
            "cluster_robust_se": fit.bse,
            "pvalue_raw": fit.pvalues,
        }
    )
    coefficients["exp_coef"] = np.exp(coefficients["coef"])
    coefficients["exp_lo95"] = np.exp(
        coefficients["coef"] - 1.96 * coefficients["cluster_robust_se"]
    )
    coefficients["exp_hi95"] = np.exp(
        coefficients["coef"] + 1.96 * coefficients["cluster_robust_se"]
    )
    finite_columns = ["coef", "cluster_robust_se", "exp_coef", "exp_lo95", "exp_hi95"]
    nonfinite_estimate_count = int(
        (~np.isfinite(coefficients[finite_columns].to_numpy(dtype=float))).sum()
    )
    ci_ratio = coefficients["exp_hi95"] / coefficients["exp_lo95"]
    extreme_ci_count = int((ci_ratio > 1e6).sum())
    try:
        influence = fit.get_influence(observed=False)
        leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
        cooks_distance = np.asarray(influence.cooks_distance[0], dtype=float)
        influence_diagnostics = {
            "max_leverage": float(np.nanmax(leverage)),
            "max_cooks_distance": float(np.nanmax(cooks_distance)),
            "n_cooks_gt_4_over_n": int((cooks_distance > 4 / len(model_frame)).sum()),
            "influence_computed": True,
        }
    except Exception as error:  # diagnostics must be visible, never silently skipped
        influence_diagnostics = {
            "influence_computed": False,
            "influence_error": f"{type(error).__name__}: {error}",
        }
    model_frame["predicted_monthly_probability"] = fit.predict(design)
    diagnostics = {
        "converged": bool(fit.converged),
        "iterations": int(fit.fit_history.get("iteration", -1)),
        "design_columns": int(design.shape[1]),
        "design_matrix_rank": design_rank,
        "design_full_rank": bool(design_rank == design.shape[1]),
        "events_per_parameter": float(outcome.sum() / design.shape[1]),
        "event_count": int(outcome.sum()),
        "model_hospital_months": int(len(model_frame)),
        "cluster_count": cluster_count,
        "covariance_type": str(fit.cov_type),
        "dropped_constant_columns": dropped_constant,
        "mean_predicted_probability": float(model_frame["predicted_monthly_probability"].mean()),
        "max_predicted_probability": float(model_frame["predicted_monthly_probability"].max()),
        "min_predicted_probability": float(model_frame["predicted_monthly_probability"].min()),
        "deviance_over_df": float(fit.deviance / fit.df_resid),
        "max_abs_coefficient": float(coefficients["coef"].abs().max()),
        "max_cluster_robust_se": float(coefficients["cluster_robust_se"].max()),
        "nonfinite_estimate_count": nonfinite_estimate_count,
        "extreme_ci_ratio_gt_1e6_count": extreme_ci_count,
        "zero_event_states_excluded_from_conditional_fe_model": sorted(zero_event_states),
        "zero_event_state_hospital_months_excluded": int(
            state_diagnostics.loc[state_diagnostics["zero_event_state"], "hospital_months"].sum()
        ),
        "influence": influence_diagnostics,
        "evidence_level": "associational discrete-time first-observed-use model",
    }
    prediction = model_frame[
        [
            "CNES", "competence_month", "hosp_uf", "event",
            "predicted_monthly_probability",
        ]
    ].copy()
    prediction["risk_set"] = label
    return coefficients, diagnostics, prediction, state_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--hospital-month", type=Path, required=True)
    parser.add_argument("--st", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    eligible = pq.read_table(
        args.eligible,
        columns=[
            "CNES", "competence_month", "broad", "primary", "strict",
            "beds_sus", "endoscopy_service", "endoscopists",
            "anesthesiologists", "medical_cbos",
        ],
    ).to_pandas()
    hospital_month = pq.read_table(
        args.hospital_month,
        columns=["SP_CNES", "first_observed_month"],
    ).to_pandas()
    st, st_audit = state_map_from_st(
        args.st,
        eligible.loc[eligible["broad"].astype(bool), ["CNES", "competence_month"]],
    )
    all_coefficients = []
    all_predictions = []
    all_state_diagnostics = []
    risk_metadata = {}
    model_diagnostics = {}
    for risk_label, eligibility_column in (
        ("primary", "primary"),
        ("broad_sensitivity", "broad"),
        ("strict_sensitivity", "strict"),
    ):
        risk, metadata = build_risk_set(
            eligible, hospital_month, st, eligibility_column
        )
        coefficients, diagnostics, predictions, state_diagnostics = fit_model(risk, risk_label)
        all_coefficients.append(coefficients)
        all_predictions.append(predictions)
        all_state_diagnostics.append(state_diagnostics)
        risk_metadata[risk_label] = metadata
        model_diagnostics[risk_label] = diagnostics

    coefficient_table = pd.concat(all_coefficients, ignore_index=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    state_diagnostic_table = pd.concat(all_state_diagnostics, ignore_index=True)
    checks = {
        "all_models_converged": all(
            item["converged"] for item in model_diagnostics.values()
        ),
        "all_event_state_complete": all(
            item["events_with_missing_state"] == 0 for item in risk_metadata.values()
        ),
        "all_prevalent_at_window_start_excluded": all(
            item["prevalent_at_window_start_in_risk"] == 0
            for item in risk_metadata.values()
        ),
        "all_event_counts_at_most_one_per_hospital": all(
            item["event_count_gt_one_hospitals"] == 0 for item in risk_metadata.values()
        ),
        "all_rows_censored_after_first_observed": all(
            item["post_first_observed_rows"] == 0 for item in risk_metadata.values()
        ),
        "all_designs_full_rank": all(
            item["design_full_rank"] for item in model_diagnostics.values()
        ),
        "all_models_epv_ge_10": all(
            item["events_per_parameter"] >= 10 for item in model_diagnostics.values()
        ),
        "primary_events_positive": risk_metadata["primary"]["n_events"] > 0,
        "left_censored_excluded": risk_metadata["primary"][
            "n_left_censored_202101_excluded"
        ] > 0,
        "primary_epv_ge_10": model_diagnostics["primary"]["events_per_parameter"] >= 10,
        "all_final_model_estimates_finite": all(
            item["nonfinite_estimate_count"] == 0 for item in model_diagnostics.values()
        ),
        "all_separated_zero_event_states_excluded_from_final_model": True,
    }
    core_check_names = [
        "all_models_converged",
        "all_event_state_complete",
        "all_prevalent_at_window_start_excluded",
        "all_event_counts_at_most_one_per_hospital",
        "all_rows_censored_after_first_observed",
        "all_designs_full_rank",
        "primary_events_positive",
        "left_censored_excluded",
        "all_final_model_estimates_finite",
    ]
    stability_warnings = []
    for risk_label, item in model_diagnostics.items():
        if item["events_per_parameter"] < 10:
            stability_warnings.append(
                {
                    "risk_set": risk_label,
                    "warning": "events-per-parameter below conventional 10 heuristic",
                    "events_per_parameter": item["events_per_parameter"],
                    "handling": (
                        "Retain as associational estimate with robust confidence intervals; "
                        "do not use for hospital ranking, causal claims, or model selection."
                    ),
                }
            )
    status = "PASS_WITH_WARNING" if all(checks[name] for name in core_check_names) else "FIX"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.coefficients.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_parquet(args.output, index=False)
    coefficient_table.to_csv(args.coefficients, index=False)
    state_diagnostic_table.to_csv(
        args.output.parent / "aim1_state_event_diagnostics.csv", index=False
    )
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "checks": checks,
        "core_check_names": core_check_names,
        "stability_warnings": stability_warnings,
        "estimand": (
            "Association between time-varying hospital characteristics and first observed "
            "coded therapeutic ERCP use among eligible hospital-months."
        ),
        "terminology": (
            "This is not a model of the true technology-adoption date. January 2021 "
            "prevalent users are left-censored and excluded from the risk model."
        ),
        "risk_sets": risk_metadata,
        "diagnostics": model_diagnostics,
        "cluster_unit": "CNES hospital",
        "time_adjustment": "4-df cubic B-spline of month index",
        "state_adjustment": "hospital-state fixed effects from monthly CNES-ST CODUFMUN",
        "cnes_st_map_qc": st_audit,
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS_WITH_WARNING" else 2


if __name__ == "__main__":
    raise SystemExit(main())
