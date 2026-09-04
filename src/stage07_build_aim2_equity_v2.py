from __future__ import annotations

"""Corrected Aim 2 treated-utilisation equity analysis.

The risk table includes every official municipality-year, including zero-event
municipalities. Adult denominators use exact ages >=18 from annual IBGE POPSBR
files. IVS and supplementary-insurance ranks are population weighted.
"""

import argparse
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dbfread import DBF


YEARS = range(2021, 2026)
AGE_GROUPS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def age_group(age: float | int) -> str | None:
    if age < 18:
        return None
    if age < 30:
        return "18-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    if age < 70:
        return "60-69"
    if age < 80:
        return "70-79"
    return "80+"


def read_population_strata(population_dir: Path) -> tuple[pd.DataFrame, dict]:
    aggregates: dict[tuple[int, str, str, str], float] = {}
    source_audit: dict[str, dict] = {}
    for year in YEARS:
        archive = population_dir / str(year) / f"POPSBR{str(year)[-2:]}.zip"
        with zipfile.ZipFile(archive) as zipped, tempfile.TemporaryDirectory(
            prefix=f"aim2_pop_{year}_"
        ) as temporary:
            members = [name for name in zipped.namelist() if name.lower().endswith(".dbf")]
            if len(members) != 1:
                raise RuntimeError(f"expected one DBF in {archive}; observed {members}")
            zipped.extract(members[0], temporary)
            table = DBF(
                str(Path(temporary) / members[0]),
                encoding="latin-1",
                char_decode_errors="ignore",
                load=False,
            )
            available = {str(field).upper(): str(field) for field in table.field_names}
            population_key = "POPULACAO" if "POPULACAO" in available else "POP"
            required = {"ANO", "IDADE", "SEXO", "COD_MUN", population_key}
            if not required.issubset(available):
                raise RuntimeError(
                    f"unexpected IBGE fields in {archive}: {table.field_names}; "
                    f"required normalized={sorted(required)}"
                )
            source_audit[str(year)] = {
                "source_fields": table.field_names,
                "normalized_mapping": {
                    "ANO": available["ANO"], "IDADE": available["IDADE"],
                    "SEXO": available["SEXO"], "COD_MUN": available["COD_MUN"],
                    "population": available[population_key],
                },
                "population_normalized_key": population_key,
                "rows_read": 0,
                "nonzero_population_rows": 0,
            }
            for row in table:
                row = {str(key).upper(): value for key, value in row.items()}
                row_year = int(str(row.get("ANO", "0")).strip() or 0)
                if row_year != year:
                    raise RuntimeError(f"population year mismatch in {archive}")
                age = int(str(row.get("IDADE", "-1")).strip() or -1)
                group = age_group(age)
                if group is None:
                    continue
                sex = str(row.get("SEXO", "")).strip()
                if sex not in {"1", "2"}:
                    continue
                municipio = str(row.get("COD_MUN", "")).strip().zfill(7)[:6]
                key = (year, municipio, sex, group)
                pop = float(row.get(population_key) or 0)
                source_audit[str(year)]["rows_read"] += 1
                source_audit[str(year)]["nonzero_population_rows"] += int(pop > 0)
                aggregates[key] = aggregates.get(key, 0.0) + pop
    result = pd.DataFrame(
        [
            {
                "year": key[0],
                "res_municipio": key[1],
                "sex_pop": key[2],
                "age_group": key[3],
                "adult_population_stratum": value,
            }
            for key, value in aggregates.items()
        ]
    )
    municipality_counts = result.groupby("year")["res_municipio"].nunique().to_dict()
    expected_counts = {year: 5570 for year in YEARS}
    expected_counts[2025] = 5571
    expected_strata = sum(expected_counts.values()) * 2 * len(AGE_GROUPS)
    if municipality_counts != expected_counts or len(result) != expected_strata:
        raise RuntimeError(
            f"unexpected adult IBGE panel: municipalities={municipality_counts}; "
            f"strata={len(result)} expected={expected_strata}"
        )
    annual = result.groupby("year", as_index=False)["adult_population_stratum"].sum()
    annual_audit = {
        str(row.year): {
            "adult_population": float(row.adult_population_stratum),
            "nonzero_denominator": bool(row.adult_population_stratum > 0),
        }
        for row in annual.itertuples(index=False)
    }
    if not all(item["nonzero_denominator"] for item in annual_audit.values()):
        raise RuntimeError("adult population denominator became zero after field normalization")
    return result, {
        "source": source_audit,
        "municipality_counts": {str(year): int(count) for year, count in municipality_counts.items()},
        "total_municipality_years": int(sum(municipality_counts.values())),
        "annual_adult_population": annual_audit,
        "administrative_change": "Official POPSBR has 5,570 municipalities in 2021–2024 and 5,571 in 2025; municipality 510183 is retained.",
    }


def weighted_ridit(values: pd.Series, weights: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": values, "weight": weights}).sort_values(
        ["value"], kind="mergesort"
    )
    total = frame["weight"].sum()
    frame["ridit"] = (frame["weight"].cumsum() - 0.5 * frame["weight"]) / total
    return frame["ridit"].reindex(values.index)


def weighted_quintile(values: pd.Series, weights: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame.sort_values("value", kind="mergesort")
    cumulative = frame["weight"].cumsum() / frame["weight"].sum()
    thresholds = [
        frame.loc[cumulative.ge(probability), "value"].iloc[0]
        for probability in (0.2, 0.4, 0.6, 0.8)
    ]
    return pd.Series(
        np.digitize(values.to_numpy(dtype=float), thresholds, right=True) + 1,
        index=values.index,
        dtype="Int64",
    )


def direct_standardised_rates(
    strata: pd.DataFrame,
    exposure: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    standard = (
        strata[strata["year"].eq(2022)]
        .groupby(["sex_pop", "age_group"], as_index=False)["adult_population_stratum"]
        .sum()
    )
    standard["weight"] = standard["adult_population_stratum"] / standard[
        "adult_population_stratum"
    ].sum()
    strata_order = [(sex, group) for sex in ("1", "2") for group in AGE_GROUPS]
    standard_weights = (
        standard.set_index(["sex_pop", "age_group"])["weight"]
        .reindex(strata_order)
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    results = []
    bootstrap: dict[int, np.ndarray] = {}
    for level in sorted(strata[exposure].dropna().astype(int).unique()):
        group = strata[strata[exposure].eq(level)].copy()
        aggregated = (
            group.groupby(["res_municipio", "sex_pop", "age_group"], as_index=False)[
                ["n", "adult_population_stratum"]
            ]
            .sum()
        )
        case_matrix = aggregated.pivot_table(
            index="res_municipio",
            columns=["sex_pop", "age_group"],
            values="n",
            aggfunc="sum",
            fill_value=0,
        ).reindex(columns=pd.MultiIndex.from_tuples(strata_order), fill_value=0).to_numpy()
        population_matrix = aggregated.pivot_table(
            index="res_municipio",
            columns=["sex_pop", "age_group"],
            values="adult_population_stratum",
            aggfunc="sum",
            fill_value=0,
        ).reindex(columns=pd.MultiIndex.from_tuples(strata_order), fill_value=0).to_numpy()

        cases = case_matrix.sum(axis=0)
        person_years = population_matrix.sum(axis=0)
        standardised = float(np.sum((cases / person_years) * standard_weights) * 100000)
        crude = float(case_matrix.sum() / population_matrix.sum() * 100000)
        draws = np.empty(n_bootstrap, dtype=float)
        for iteration in range(n_bootstrap):
            sampled = rng.integers(0, case_matrix.shape[0], size=case_matrix.shape[0])
            sampled_cases = case_matrix[sampled].sum(axis=0)
            sampled_population = population_matrix[sampled].sum(axis=0)
            draws[iteration] = float(
                np.sum((sampled_cases / sampled_population) * standard_weights) * 100000
            )
        bootstrap[int(level)] = draws
        results.append(
            {
                exposure: int(level),
                "n": int(case_matrix.sum()),
                "adult_person_years": float(population_matrix.sum()),
                "crude_rate_per_100k": crude,
                "age_sex_standardised_rate_per_100k": standardised,
                "standardised_lo95": float(np.quantile(draws, 0.025)),
                "standardised_hi95": float(np.quantile(draws, 0.975)),
                "bootstrap_municipalities": int(case_matrix.shape[0]),
            }
        )
    result = pd.DataFrame(results)
    low = int(result[exposure].min())
    high = int(result[exposure].max())
    difference_draws = bootstrap[high] - bootstrap[low]
    ratio_draws = bootstrap[high] / bootstrap[low]
    contrasts = {
        "high_minus_low": float(
            result.loc[result[exposure].eq(high), "age_sex_standardised_rate_per_100k"].iloc[0]
            - result.loc[result[exposure].eq(low), "age_sex_standardised_rate_per_100k"].iloc[0]
        ),
        "high_minus_low_lo95": float(np.quantile(difference_draws, 0.025)),
        "high_minus_low_hi95": float(np.quantile(difference_draws, 0.975)),
        "high_to_low_ratio": float(
            result.loc[result[exposure].eq(high), "age_sex_standardised_rate_per_100k"].iloc[0]
            / result.loc[result[exposure].eq(low), "age_sex_standardised_rate_per_100k"].iloc[0]
        ),
        "high_to_low_ratio_lo95": float(np.quantile(ratio_draws, 0.025)),
        "high_to_low_ratio_hi95": float(np.quantile(ratio_draws, 0.975)),
    }
    return result, contrasts


def fit_ridit_model(frame: pd.DataFrame, ridit: str) -> dict:
    model_frame = frame.dropna(subset=[ridit, "adult_population"]).copy()
    model_frame = model_frame[model_frame["adult_population"].gt(0)]
    fit = smf.glm(
        f"n ~ {ridit} + C(year) + C(uf)",
        data=model_frame,
        family=sm.families.Poisson(),
        offset=np.log(model_frame["adult_population"]),
    ).fit(cov_type="cluster", cov_kwds={"groups": model_frame["res_municipio"]})
    beta = float(fit.params[ridit])
    se = float(fit.bse[ridit])
    finite = bool(np.isfinite(np.asarray(fit.params, dtype=float)).all() and np.isfinite(np.asarray(fit.bse, dtype=float)).all())
    low = model_frame.copy()
    high = model_frame.copy()
    low[ridit] = 0.0
    high[ridit] = 1.0
    standard_offset = np.log(np.repeat(100000.0, len(model_frame)))
    rate_low_rows = fit.predict(low, offset=standard_offset)
    rate_high_rows = fit.predict(high, offset=standard_offset)
    weights = model_frame["adult_population"].to_numpy()
    rate_low = float(np.average(rate_low_rows, weights=weights))
    rate_high = float(np.average(rate_high_rows, weights=weights))
    return {
        "n_municipality_years": int(len(model_frame)),
        "n_municipalities": int(model_frame["res_municipio"].nunique()),
        "coefficient": beta,
        "cluster_robust_se": se,
        "RII_rate_ratio": float(np.exp(beta)),
        "RII_lo95": float(np.exp(beta - 1.96 * se)),
        "RII_hi95": float(np.exp(beta + 1.96 * se)),
        "SII_rate_difference_per_100k": rate_high - rate_low,
        "predicted_rate_rank0_per_100k": rate_low,
        "predicted_rate_rank1_per_100k": rate_high,
        "pvalue_raw": float(fit.pvalues[ridit]),
        "pearson_chi2_over_df": float(fit.pearson_chi2 / fit.df_resid),
        "converged": bool(fit.converged),
        "finite_estimates": finite,
        "design_columns": int(len(fit.params)),
        "inference": "Poisson mean with municipality-clustered sandwich covariance",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--ivs", type=Path, required=True)
    parser.add_argument("--ans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strata-output", type=Path, required=True)
    parser.add_argument("--quintile-output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    population_strata, population_audit = read_population_strata(args.population_dir)
    population = (
        population_strata.groupby(["year", "res_municipio"], as_index=False)[
            "adult_population_stratum"
        ]
        .sum()
        .rename(columns={"adult_population_stratum": "adult_population"})
    )
    cohort = pq.read_table(
        args.cohorts,
        columns=["cohort", "MUNIC_RES", "competence_month", "SEXO", "age_years"],
    ).to_pandas()
    cohort = cohort[cohort["cohort"].astype(str).eq("B")].copy()
    cohort["res_municipio"] = cohort["MUNIC_RES"].astype(str).str.zfill(6)
    cohort["year"] = cohort["competence_month"].astype(str).str[:4].astype(int)
    cohort["sex_pop"] = cohort["SEXO"].astype(str).map({"1": "1", "3": "2"})
    cohort["age_group"] = pd.to_numeric(cohort["age_years"], errors="coerce").map(age_group)
    invalid_strata = int(cohort[["sex_pop", "age_group"]].isna().any(axis=1).sum())
    cases = (
        cohort.dropna(subset=["sex_pop", "age_group"])
        .groupby(["year", "res_municipio", "sex_pop", "age_group"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    strata = population_strata.merge(
        cases,
        on=["year", "res_municipio", "sex_pop", "age_group"],
        how="left",
        validate="one_to_one",
    )
    strata["n"] = strata["n"].fillna(0).astype(int)

    municipality_year = population.merge(
        cohort.groupby(["year", "res_municipio"]).size().rename("n").reset_index(),
        on=["year", "res_municipio"],
        how="left",
        validate="one_to_one",
    )
    municipality_year["n"] = municipality_year["n"].fillna(0).astype(int)
    municipality_year["rate_per_100k"] = (
        municipality_year["n"] / municipality_year["adult_population"] * 100000
    )
    municipality_year["uf"] = municipality_year["res_municipio"].str[:2]

    ivs = pq.read_table(args.ivs).to_pandas()
    ivs_columns = ["ano", "label_cor", "label_sexo", "label_sit_dom"]
    for column in ivs_columns:
        ivs[column] = ivs[column].astype(str)
    ivs = ivs[
        ivs["ano"].eq("2010")
        & ivs["label_cor"].eq("Total Cor")
        & ivs["label_sexo"].eq("Total Sexo")
        & ivs["label_sit_dom"].eq("Total Situação de Domicílio")
    ].copy()
    ivs["res_municipio"] = ivs["municipio"].astype(str).str.zfill(7).str[:6]
    ivs = ivs[["res_municipio", "ivs"]].drop_duplicates("res_municipio")
    ivs["ivs"] = pd.to_numeric(ivs["ivs"], errors="coerce")
    reference_population = population[population["year"].eq(2022)][
        ["res_municipio", "adult_population"]
    ].merge(ivs, on="res_municipio", how="left", validate="one_to_one")
    complete_reference = reference_population.dropna(subset=["ivs"]).copy()
    complete_reference["ivs_ridit"] = weighted_ridit(
        complete_reference["ivs"], complete_reference["adult_population"]
    )
    complete_reference["ivs_quintile"] = weighted_quintile(
        complete_reference["ivs"], complete_reference["adult_population"]
    )
    municipality_year = municipality_year.merge(
        complete_reference[["res_municipio", "ivs", "ivs_ridit", "ivs_quintile"]],
        on="res_municipio",
        how="left",
        validate="many_to_one",
    )

    ans = pd.read_csv(args.ans, encoding="utf-8-sig", dtype=str)
    ans["res_municipio"] = ans["cd_municipio_6"].str.zfill(6)
    ans["year"] = pd.to_numeric(ans["year"], errors="coerce").astype("Int64")
    ans["supp_coverage"] = pd.to_numeric(ans["supplementary_coverage_rate"], errors="coerce")
    ans = ans[["res_municipio", "year", "supp_coverage"]].drop_duplicates(
        ["res_municipio", "year"]
    )
    municipality_year = municipality_year.merge(
        ans,
        on=["res_municipio", "year"],
        how="left",
        validate="one_to_one",
    )
    municipality_year["ans_ridit"] = np.nan
    municipality_year["ans_quintile"] = pd.Series(pd.NA, index=municipality_year.index, dtype="Int64")
    for year, group in municipality_year.groupby("year"):
        complete = group.dropna(subset=["supp_coverage"]).copy()
        municipality_year.loc[complete.index, "ans_ridit"] = weighted_ridit(
            complete["supp_coverage"], complete["adult_population"]
        )
        municipality_year.loc[complete.index, "ans_quintile"] = weighted_quintile(
            complete["supp_coverage"], complete["adult_population"]
        )
    municipality_year["ans_quintile"] = municipality_year["ans_quintile"].astype("Int64")

    strata = strata.merge(
        municipality_year[
            ["year", "res_municipio", "ivs_quintile", "ans_quintile"]
        ],
        on=["year", "res_municipio"],
        how="left",
        validate="many_to_one",
    )
    ivs_rates, ivs_contrasts = direct_standardised_rates(
        strata.dropna(subset=["ivs_quintile"]),
        "ivs_quintile",
        args.bootstrap,
        args.seed,
    )
    ans_rates, ans_contrasts = direct_standardised_rates(
        strata.dropna(subset=["ans_quintile"]),
        "ans_quintile",
        args.bootstrap,
        args.seed + 1,
    )
    ivs_rates["exposure"] = "IVS_2010_population_weighted_quintile"
    ivs_rates = ivs_rates.rename(columns={"ivs_quintile": "quintile"})
    ans_rates["exposure"] = "ANS_supplementary_coverage_population_weighted_quintile"
    ans_rates = ans_rates.rename(columns={"ans_quintile": "quintile"})
    quintile_rates = pd.concat([ivs_rates, ans_rates], ignore_index=True)

    ivs_model = fit_ridit_model(municipality_year, "ivs_ridit")
    ans_model = fit_ridit_model(municipality_year, "ans_ridit")
    ivs_population_coverage = float(
        municipality_year.loc[municipality_year["ivs"].notna(), "adult_population"].sum()
        / municipality_year["adult_population"].sum()
    )
    ans_population_coverage = float(
        municipality_year.loc[
            municipality_year["supp_coverage"].notna(), "adult_population"
        ].sum()
        / municipality_year["adult_population"].sum()
    )
    checks = {
        "municipality_year_rows_27851": len(municipality_year) == 27851,
        "zero_event_municipality_years_retained": bool(municipality_year["n"].eq(0).any()),
        "case_conservation": int(municipality_year["n"].sum()) == len(cohort),
        "adult_population_positive": bool(municipality_year["adult_population"].gt(0).all()),
        "ivs_population_coverage_ge_95pct": ivs_population_coverage >= 0.95,
        "ans_population_coverage_ge_95pct": ans_population_coverage >= 0.95,
        "ivs_model_converged": ivs_model["converged"],
        "ans_model_converged": ans_model["converged"],
        "ivs_model_finite": ivs_model["finite_estimates"],
        "ans_model_finite": ans_model["finite_estimates"],
    }
    status = "PASS" if all(checks.values()) and invalid_strata == 0 else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.strata_output.parent.mkdir(parents=True, exist_ok=True)
    args.quintile_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    municipality_year.to_parquet(args.output, index=False)
    strata.to_parquet(args.strata_output, index=False)
    quintile_rates.to_csv(args.quintile_output, index=False)
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "checks": checks,
        "estimand": "Treated cohort-B ERCP utilisation per 100,000 adult residents",
        "evidence_level": "descriptive and associational; ecological contextual exposure",
        "n_cohort_b": int(len(cohort)),
        "invalid_case_age_sex_strata": invalid_strata,
        "municipality_years": int(len(municipality_year)),
        "zero_event_municipality_years": int(municipality_year["n"].eq(0).sum()),
        "adult_person_years": float(municipality_year["adult_population"].sum()),
        "population_read_qc": population_audit,
        "ivs_population_coverage": ivs_population_coverage,
        "ans_population_coverage": ans_population_coverage,
        "ivs_quintile_contrasts": ivs_contrasts,
        "ans_quintile_contrasts": ans_contrasts,
        "ivs_ridit_model": ivs_model,
        "ans_ridit_model": ans_model,
        "multiplicity": (
            "Raw p values only at this component stage; Holm adjustment is deferred until "
            "the three prespecified Aim-2 primary endpoint tests are assembled."
        ),
        "missing_data": (
            "No contextual value was imputed in the primary analysis. Complete-context "
            "rows enter each exposure model; all municipalities remain in national counts."
        ),
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
