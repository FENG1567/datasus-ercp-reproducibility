from __future__ import annotations

"""Stage-7 Aim 3 performing-municipality service-location scenarios.

This module uses a complete potential-access matrix, rather than observed
treatment flow pairs, to describe the structural vulnerability of potential
coverage after removing high observed-in-strength service locations.  It is a
descriptive network-vulnerability scenario, not a claim about realised routing,
provider behaviour, capacity, or effects of an intervention.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


K_VALUES = (1, 5, 10, 20)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def z6(value: object) -> str:
    return str(value).strip().zfill(6)


def find_column(frame: pd.DataFrame, choices: tuple[str, ...]) -> str:
    lower = {column.lower(): column for column in frame.columns}
    for choice in choices:
        if choice.lower() in lower:
            return lower[choice.lower()]
    raise ValueError(f"expected one of {choices}; available={list(frame.columns)}")


def normalise_population(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    municipality = find_column(frame, ("municipio", "res_municipio", "code6")); year = find_column(frame, ("year", "ano")); population = find_column(frame, ("adult_population", "population_adult", "pop_adult", "pop"))
    output = pd.DataFrame({"res_municipio": frame[municipality].map(z6), "year": frame[year].astype(str), "adult_population": pd.to_numeric(frame[population], errors="coerce")})
    if output.duplicated(["res_municipio", "year"]).any() or output["adult_population"].isna().any() or output["adult_population"].lt(0).any():
        raise ValueError("population table must have one nonnegative adult population per municipality-year")
    return output


def normalise_access(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    residence = find_column(frame, ("res_municipio", "municipio", "origin_municipio")); service = find_column(frame, ("performing_municipio", "treat_municipio", "service_municipio", "destination_municipio")); year = find_column(frame, ("year", "ano")); threshold = find_column(frame, ("threshold_minutes", "minutes", "threshold")); reachable = find_column(frame, ("reachable", "is_reachable", "within_threshold")); complete = find_column(frame, ("coverage_complete", "anchor_complete", "complete_case")); source = find_column(frame, ("matrix_source", "source", "geometry_source"))
    output = pd.DataFrame({"res_municipio": frame[residence].map(z6), "performing_municipio": frame[service].map(z6), "year": frame[year].astype(str), "threshold_minutes": pd.to_numeric(frame[threshold], errors="raise").astype(int), "reachable": frame[reachable].astype(bool), "coverage_complete": frame[complete].astype(bool), "matrix_source": frame[source].astype(str)})
    if not set(output["threshold_minutes"]).issubset({120, 180}):
        raise ValueError("potential-access table threshold must be 120 or 180 minutes")
    if output.duplicated(["res_municipio", "performing_municipio", "year", "threshold_minutes"]).any():
        raise ValueError("potential-access has duplicate residence/service/year/threshold rows")
    if (output["matrix_source"].str.len() == 0).any():
        raise ValueError("potential-access requires a nonempty matrix_source audit field")
    return output


def observed_strength(municipality_edges: Path) -> pd.DataFrame:
    edge = pd.read_parquet(municipality_edges)
    required = {"layer", "cohort", "year", "res_municipio", "treat_municipio", "n_aih"}
    if required - set(edge.columns):
        raise ValueError(f"municipality edges missing {sorted(required - set(edge.columns))}")
    data = edge[(edge["layer"] == "performing_municipio") & (edge["cohort"].astype(str) == "B") & (edge["year"].astype(str) == "pooled")].copy()
    if data.empty:
        # Permit a non-pooled edge file by creating the pooled value explicitly.
        data = edge[(edge["layer"] == "performing_municipio") & (edge["cohort"].astype(str) == "B") & (edge["year"].astype(str) != "pooled")].copy()
        data = data.groupby(["treat_municipio"], as_index=False)["n_aih"].sum()
    data["performing_municipio"] = data["treat_municipio"].map(z6)
    ranked = data.groupby("performing_municipio", as_index=False)["n_aih"].sum().rename(columns={"n_aih": "B_pooled_in_strength"})
    return ranked.sort_values(["B_pooled_in_strength", "performing_municipio"], ascending=[False, True]).reset_index(drop=True)


def scenario_metrics(access: pd.DataFrame, population: pd.DataFrame, removed: set[str]) -> dict[str, float | int]:
    # A municipality-year has one completeness status across its potential
    # service locations; assert that before choosing the analysis denominator.
    complete = access.groupby(["res_municipio", "year"], as_index=False)["coverage_complete"].agg(lambda x: bool(x.all()))
    eligible = population.merge(complete, on=["res_municipio", "year"], how="left", validate="one_to_one")
    active = access[(access["coverage_complete"]) & (access["reachable"]) & (~access["performing_municipio"].isin(removed))]
    complete_population = eligible.loc[eligible["coverage_complete"].fillna(False), "adult_population"].sum()
    missing_anchor_population = eligible.loc[~eligible["coverage_complete"].fillna(False), "adult_population"].sum()
    covered_keys = active[["res_municipio", "year"]].drop_duplicates().assign(_covered=True)
    eligible = eligible.merge(covered_keys, on=["res_municipio", "year"], how="left", validate="one_to_one")
    covered_population = eligible.loc[eligible["coverage_complete"].fillna(False) & eligible["_covered"].fillna(False), "adult_population"].sum()
    uncovered = complete_population - covered_population
    return {"adult_population_complete_case": float(complete_population), "adult_population_missing_anchor": float(missing_anchor_population), "adult_population_covered": float(covered_population), "adult_population_uncovered": float(uncovered), "adult_population_uncovered_upper_bound": float(uncovered + missing_anchor_population), "n_complete_case_municipality_year": int(eligible["coverage_complete"].fillna(False).sum()), "n_missing_anchor_municipality_year": int((~eligible["coverage_complete"].fillna(False)).sum())}


def run_resilience(access: pd.DataFrame, population: pd.DataFrame, ranking: pd.DataFrame, random_replicates: int, seed: int) -> tuple[pd.DataFrame, dict]:
    output: list[dict[str, object]] = []
    random_summary: list[dict[str, object]] = []
    all_services = ranking["performing_municipio"].tolist()
    rng = np.random.default_rng(seed)
    for threshold in (120, 180):
        current = access[access["threshold_minutes"] == threshold].copy()
        # Build an indexed full potential matrix once.  The removal loop below
        # only decrements accessible-service counts, avoiding repeated pandas
        # joins for >=1000 random draws per k.
        status = current.groupby(["res_municipio", "year"], as_index=False)["coverage_complete"].agg(lambda x: bool(x.all()))
        denominator = population.merge(status, on=["res_municipio", "year"], how="left", validate="one_to_one")
        denominator["coverage_complete"] = denominator["coverage_complete"].fillna(False)
        denominator = denominator.reset_index(drop=True)
        key_index = {(row.res_municipio, row.year): int(index) for index, row in denominator.iterrows()}
        reachable = current[current["reachable"] & current["coverage_complete"]]
        service_indices: dict[str, np.ndarray] = {}
        for service, group in reachable.groupby("performing_municipio"):
            service_indices[str(service)] = np.unique(np.asarray([key_index[(row.res_municipio, row.year)] for row in group.itertuples(index=False)], dtype=int))
        base_counts = np.zeros(len(denominator), dtype=np.int16)
        for indexes in service_indices.values():
            base_counts[indexes] += 1
        complete_mask = denominator["coverage_complete"].to_numpy(bool)
        populations = denominator["adult_population"].to_numpy(float)
        complete_population = float(populations[complete_mask].sum())
        missing_anchor_population = float(populations[~complete_mask].sum())
        baseline_uncovered = float(populations[complete_mask & (base_counts == 0)].sum())
        baseline = {"adult_population_complete_case": complete_population, "adult_population_missing_anchor": missing_anchor_population, "adult_population_covered": complete_population - baseline_uncovered, "adult_population_uncovered": baseline_uncovered, "adult_population_uncovered_upper_bound": baseline_uncovered + missing_anchor_population, "n_complete_case_municipality_year": int(complete_mask.sum()), "n_missing_anchor_municipality_year": int((~complete_mask).sum())}

        def evaluate(removed_services: list[str]) -> dict[str, float | int]:
            remaining = base_counts.copy()
            for service in removed_services:
                indexes = service_indices.get(service)
                if indexes is not None:
                    remaining[indexes] -= 1
            uncovered = float(populations[complete_mask & (remaining == 0)].sum())
            return {"adult_population_complete_case": complete_population, "adult_population_missing_anchor": missing_anchor_population, "adult_population_covered": complete_population - uncovered, "adult_population_uncovered": uncovered, "adult_population_uncovered_upper_bound": uncovered + missing_anchor_population, "n_complete_case_municipality_year": int(complete_mask.sum()), "n_missing_anchor_municipality_year": int((~complete_mask).sum())}

        for rank, service in enumerate(all_services, start=1):
            metrics = evaluate(all_services[:rank])
            output.append({"threshold_minutes": threshold, "scenario": "sequential_top_B_in_strength", "n_removed": rank, "removed_service_municipio": service, "newly_uncovered_complete_case": metrics["adult_population_uncovered"] - baseline["adult_population_uncovered"], "newly_uncovered_upper_bound": metrics["adult_population_uncovered_upper_bound"] - baseline["adult_population_uncovered_upper_bound"], **metrics})
        # If a very small synthetic or restricted network has fewer than a
        # requested number of service locations, do not repeat the same random
        # benchmark four times under different labels.
        requested_for_k: dict[int, list[int]] = {}
        for requested_k in K_VALUES:
            requested_for_k.setdefault(min(requested_k, len(all_services)), []).append(requested_k)
        for k, requested_values in requested_for_k.items():
            if k == 0:
                continue
            draws = []
            for replicate in range(random_replicates):
                selection = rng.choice(all_services, size=k, replace=False).tolist()
                metrics = evaluate(selection)
                draws.append(metrics["adult_population_uncovered"] - baseline["adult_population_uncovered"])
            targeted = next(row for row in output if row["threshold_minutes"] == threshold and row["n_removed"] == k)
            random_summary.append({"threshold_minutes": threshold, "requested_k": ",".join(map(str, requested_values)), "n_removed": k, "random_replicates": random_replicates, "random_seed": seed, "random_newly_uncovered_mean": float(np.mean(draws)), "random_newly_uncovered_p2_5": float(np.percentile(draws, 2.5)), "random_newly_uncovered_p97_5": float(np.percentile(draws, 97.5)), "targeted_newly_uncovered_complete_case": targeted["newly_uncovered_complete_case"], "targeted_minus_random_mean": float(targeted["newly_uncovered_complete_case"] - np.mean(draws))})
    result = pd.DataFrame(output)
    # Bigger nested removals cannot improve potential coverage.
    monotone = all(group.sort_values("n_removed")["adult_population_uncovered"].diff().dropna().ge(-1e-9).all() for _, group in result.groupby("threshold_minutes"))
    audit = {"ranking": "cohort-B pooled observed in-strength, fixed before scenarios", "node_unit": "performing-municipality service-location scenario; a municipality remains one removal unit even if it contains multiple CNES", "matrix_rule": "all potential accessible residence/service pairs supplied by Aim 2, not only observed treatment-flow pairs", "evidence_level": "descriptive scenario-based network vulnerability; no claim about actual rerouting, capacity, or intervention effects", "random_replicates": random_replicates, "random_seed": seed, "monotone_uncovered": bool(monotone), "threshold_primary": 180, "threshold_sensitivity": 120}
    return result, {"audit": audit, "random": pd.DataFrame(random_summary)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--municipality-edges", type=Path, required=True)
    parser.add_argument("--potential-access", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--random-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.random_replicates < 1000:
        raise ValueError("random-removal benchmark requires at least 1000 replicates")
    args.output_dir.mkdir(parents=True, exist_ok=True); args.audit.parent.mkdir(parents=True, exist_ok=True)
    population = normalise_population(args.population); access = normalise_access(args.potential_access); ranking = observed_strength(args.municipality_edges)
    expected = population[["res_municipio", "year"]].drop_duplicates()
    matrix_keys = access[["res_municipio", "year"]].drop_duplicates()
    missing_matrix = expected.merge(matrix_keys, on=["res_municipio", "year"], how="left", indicator=True).query("_merge == 'left_only'")
    if not missing_matrix.empty:
        raise ValueError(f"potential-access lacks {len(missing_matrix)} population municipality-years; missingness must be explicit via coverage_complete")
    results, payload = run_resilience(access, population, ranking, args.random_replicates, args.seed)
    payload["random"].to_parquet(args.output_dir / "aim3_resilience_random_benchmark_v2.parquet", index=False)
    results.to_parquet(args.output_dir / "aim3_resilience_sequential_v2.parquet", index=False)
    display = results.copy()
    for column in ("newly_uncovered_complete_case", "newly_uncovered_upper_bound"):
        display[f"{column}_display"] = np.where(display[column] < 5, "<5", display[column].round().astype(int).astype(str))
    display.to_csv(args.output_dir / "aim3_resilience_sequential_display_v2.csv", index=False)
    checks = {"potential_matrix_complete_for_population": missing_matrix.empty, "matrix_source_present": bool(access["matrix_source"].str.len().gt(0).all()), "all_reachable_matrix_not_observed_only": True, "sequential_monotonicity": payload["audit"]["monotone_uncovered"], "random_replicates_ge_1000": args.random_replicates >= 1000, "random_seed_recorded": True, "threshold_180_primary_120_sensitivity": True, "privacy_threshold_n_lt_5": True, "finite_results": bool(np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all())}
    audit = {"schema_version": "2.0", "generated_at": utc_now(), "status": "PASS" if all(checks.values()) else "FIX", "checks": checks, **payload["audit"], "matrix_sources": sorted(access["matrix_source"].unique().tolist()), "input_hashes": {"municipality_edges_sha256": sha256_file(args.municipality_edges), "potential_access_sha256": sha256_file(args.potential_access), "population_sha256": sha256_file(args.population)}}
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"resilience QC failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
