from __future__ import annotations

"""Build the complete Aim-3 potential-access matrix from frozen Aim-2 inputs.

The output enumerates every residence municipality, every annually active
cohort-A performing municipality, and both frozen road-time thresholds.  A
missing residence road anchor is retained explicitly as ``coverage_complete =
False`` and is never recoded as not covered.  The matrix describes potential
road access only; it is not an observed referral or a rerouting counterfactual.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import make_valid

from stage07_rebuild_coverage import build_anchors


YEARS = tuple(range(2021, 2026))
THRESHOLDS = (120, 180)
OUTPUT_COLUMNS = (
    "res_municipio",
    "performing_municipio",
    "year",
    "threshold_minutes",
    "reachable",
    "coverage_complete",
    "matrix_source",
)


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


def active_provider_years(cohorts_path: Path) -> pd.DataFrame:
    cohort = pq.read_table(
        cohorts_path,
        columns=["cohort", "competence_month", "MUNIC_MOV"],
    ).to_pandas()
    cohort = cohort[cohort["cohort"].astype(str).eq("A")].copy()
    cohort["year"] = pd.to_numeric(
        cohort["competence_month"].astype(str).str[:4], errors="raise"
    ).astype(int)
    cohort["performing_municipio"] = cohort["MUNIC_MOV"].map(z6)
    active = (
        cohort[["year", "performing_municipio"]]
        .drop_duplicates()
        .sort_values(["year", "performing_municipio"])
        .reset_index(drop=True)
    )
    observed_years = tuple(sorted(active["year"].unique().tolist()))
    if observed_years != YEARS:
        raise ValueError(f"unexpected cohort-A years: {observed_years}")
    if (active.groupby("year")["performing_municipio"].nunique() == 0).any():
        raise ValueError("each year must contain at least one active service municipality")
    return active


def corrected_isochrones(path: Path) -> tuple[gpd.GeoDataFrame, int]:
    raw = gpd.read_parquet(path)
    required = {"hosp_municipio", "bucket_min", "geometry"}
    if required - set(raw.columns):
        raise ValueError(f"raw isochrones missing {sorted(required - set(raw.columns))}")
    raw["hosp_municipio"] = raw["hosp_municipio"].map(z6)
    raw["bucket_min"] = pd.to_numeric(raw["bucket_min"], errors="raise").astype(int)
    if set(raw["bucket_min"].unique()) != set(THRESHOLDS):
        raise ValueError("raw isochrones must contain exactly 120- and 180-minute contours")
    if raw.duplicated(["hosp_municipio", "bucket_min"]).any():
        raise ValueError("raw isochrones contain duplicate provider-threshold rows")

    providers = sorted(raw["hosp_municipio"].unique())
    expected = pd.MultiIndex.from_product(
        [providers, THRESHOLDS], names=["hosp_municipio", "bucket_min"]
    )
    observed = pd.MultiIndex.from_frame(raw[["hosp_municipio", "bucket_min"]])
    if len(observed) != len(expected) or not expected.isin(observed).all():
        raise ValueError("provider isochrone pairs are incomplete")

    corrected: list[dict[str, object]] = []
    corrections = 0
    for provider, group in raw.groupby("hosp_municipio", sort=True):
        geom120 = group.loc[group["bucket_min"].eq(120), "geometry"].iloc[0]
        geom180_raw = group.loc[group["bucket_min"].eq(180), "geometry"].iloc[0]
        nested = bool(geom180_raw.covers(geom120))
        geom180 = geom180_raw if nested else make_valid(geom180_raw.union(geom120))
        corrections += int(not nested)
        corrected.extend(
            [
                {"performing_municipio": provider, "threshold_minutes": 120, "geometry": geom120},
                {"performing_municipio": provider, "threshold_minutes": 180, "geometry": geom180},
            ]
        )
    result = gpd.GeoDataFrame(corrected, geometry="geometry", crs=raw.crs or "EPSG:4326")
    return result, corrections


def normalise_coverage(path: Path) -> pd.DataFrame:
    coverage = pd.read_parquet(path)
    required = {
        "year",
        "municipio",
        "adult_population",
        "anchor_available",
        "has_provider_120",
        "has_provider_180",
    }
    if required - set(coverage.columns):
        raise ValueError(f"coverage table missing {sorted(required - set(coverage.columns))}")
    output = coverage[list(required)].copy()
    output["year"] = pd.to_numeric(output["year"], errors="raise").astype(int)
    output["municipio"] = output["municipio"].map(z6)
    output["anchor_available"] = output["anchor_available"].astype(bool)
    if output.duplicated(["year", "municipio"]).any():
        raise ValueError("coverage must contain one row per municipality-year")
    counts = output.groupby("year")["municipio"].nunique().to_dict()
    expected_counts = {2021: 5570, 2022: 5570, 2023: 5570, 2024: 5570, 2025: 5571}
    if counts != expected_counts:
        raise ValueError(f"unexpected population panel counts: {counts}")
    if not ((output["year"] == 2025) & (output["municipio"] == "510183")).any():
        raise ValueError("2025 municipality 510183 was not retained")
    return output.sort_values(["year", "municipio"]).reset_index(drop=True)


def reachable_pairs(
    residence_points: gpd.GeoDataFrame,
    annual_services: list[str],
    isochrones: gpd.GeoDataFrame,
    threshold: int,
) -> pd.DataFrame:
    selected = isochrones[
        isochrones["threshold_minutes"].eq(threshold)
        & isochrones["performing_municipio"].isin(annual_services)
    ][["performing_municipio", "geometry"]]
    if len(selected) != len(annual_services):
        missing = sorted(set(annual_services) - set(selected["performing_municipio"]))
        raise ValueError(f"missing corrected service isochrones: {missing}")
    joined = residence_points.sjoin(selected, how="inner", predicate="intersects")
    return joined[["res_municipio", "performing_municipio"]].drop_duplicates()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--raw-isochrones", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--snaps", type=Path, required=True)
    parser.add_argument("--anchor-overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    active = active_provider_years(args.cohorts)
    coverage = normalise_coverage(args.coverage)
    isochrones, nesting_corrections = corrected_isochrones(args.raw_isochrones)
    pooled_services = set(active["performing_municipio"])
    if pooled_services != set(isochrones["performing_municipio"]):
        raise ValueError("pooled cohort-A services do not match complete isochrone services")

    anchors = build_anchors(args.centroids, args.snaps, args.anchor_overrides)
    anchors["municipio"] = anchors["municipio"].map(z6)
    anchor_points = gpd.GeoDataFrame(
        anchors[["municipio"]].rename(columns={"municipio": "res_municipio"}),
        geometry=gpd.points_from_xy(anchors["lon"], anchors["lat"]),
        crs="EPSG:4326",
    )
    if isochrones.crs != anchor_points.crs:
        isochrones = isochrones.to_crs(anchor_points.crs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    source_label = "GraphHopper road-time isochrone x annual observed cohort-A service municipality"
    writer: pq.ParquetWriter | None = None
    annual_audit: dict[str, dict[str, object]] = {}
    total_rows = 0
    total_reachable = 0
    monotonic_sets: dict[int, dict[int, set[tuple[str, str]]]] = {}
    try:
        for year in YEARS:
            annual_population = coverage[coverage["year"].eq(year)].copy()
            residences = annual_population["municipio"].tolist()
            complete_by_residence = annual_population.set_index("municipio")["anchor_available"]
            annual_services = active.loc[
                active["year"].eq(year), "performing_municipio"
            ].tolist()
            annual_audit[str(year)] = {
                "n_residence_municipalities": len(residences),
                "n_active_service_municipalities": len(annual_services),
                "n_anchor_complete": int(complete_by_residence.sum()),
            }
            monotonic_sets[year] = {}

            for threshold in THRESHOLDS:
                annual_points = anchor_points[
                    anchor_points["res_municipio"].isin(residences)
                ].copy()
                reachable = reachable_pairs(
                    annual_points, annual_services, isochrones, threshold
                )
                reachable_keys = set(map(tuple, reachable.to_numpy()))
                monotonic_sets[year][threshold] = reachable_keys

                res_values = np.repeat(np.asarray(residences, dtype=object), len(annual_services))
                service_values = np.tile(
                    np.asarray(annual_services, dtype=object), len(residences)
                )
                batch = pd.DataFrame(
                    {
                        "res_municipio": res_values,
                        "performing_municipio": service_values,
                    }
                )
                batch_index = pd.MultiIndex.from_frame(
                    batch[["res_municipio", "performing_municipio"]]
                )
                batch["year"] = str(year)
                batch["threshold_minutes"] = np.int16(threshold)
                batch["reachable"] = batch_index.isin(reachable_keys)
                batch["coverage_complete"] = (
                    batch["res_municipio"].map(complete_by_residence).astype(bool)
                )
                batch.loc[~batch["coverage_complete"], "reachable"] = False
                batch["matrix_source"] = source_label
                batch = batch[list(OUTPUT_COLUMNS)]

                table = pa.Table.from_pandas(batch, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                        use_dictionary=[
                            "res_municipio",
                            "performing_municipio",
                            "year",
                            "matrix_source",
                        ],
                    )
                writer.write_table(table)
                total_rows += len(batch)
                total_reachable += int(batch["reachable"].sum())
                annual_audit[str(year)][f"n_reachable_pairs_{threshold}"] = int(
                    batch["reachable"].sum()
                )
                annual_audit[str(year)][f"rows_{threshold}"] = len(batch)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("no potential-access rows were written")
    temporary.replace(args.output)

    monotonic_violations = sum(
        len(monotonic_sets[year][120] - monotonic_sets[year][180]) for year in YEARS
    )
    coverage_mismatches = 0
    unknown_semantics_violations = 0
    for year in YEARS:
        annual_population = coverage[coverage["year"].eq(year)].set_index("municipio")
        for threshold in THRESHOLDS:
            derived = {residence for residence, _ in monotonic_sets[year][threshold]}
            reference = annual_population[f"has_provider_{threshold}"]
            for residence, complete in annual_population["anchor_available"].items():
                if complete:
                    coverage_mismatches += int(bool(reference.loc[residence]) != (residence in derived))
                else:
                    unknown_semantics_violations += int(pd.notna(reference.loc[residence]))

    expected_rows = int(
        sum(
            coverage.loc[coverage["year"].eq(year), "municipio"].nunique()
            * active.loc[active["year"].eq(year), "performing_municipio"].nunique()
            * len(THRESHOLDS)
            for year in YEARS
        )
    )
    metadata_rows = pq.ParquetFile(args.output).metadata.num_rows
    checks = {
        "complete_cross_product_row_count": total_rows == expected_rows == metadata_rows,
        "all_provider_isochrone_pairs_complete": len(isochrones) == len(pooled_services) * 2,
        "coverage_equivalence_to_aim2": coverage_mismatches == 0,
        "missing_anchor_remains_unknown": unknown_semantics_violations == 0,
        "reachable_120_subset_180": monotonic_violations == 0,
        "all_population_municipality_years_represented": len(coverage) == 27851,
        "administrative_change_510183_retained": bool(
            ((coverage["year"] == 2025) & (coverage["municipio"] == "510183")).any()
        ),
        "matrix_source_nonempty": bool(source_label),
    }
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": "PASS" if all(checks.values()) else "FIX",
        "evidence_level": (
            "descriptive potential road access; not observed referral, capacity, "
            "realised access, intervention effect, or rerouting counterfactual"
        ),
        "matrix_definition": (
            "Every official population municipality-year x every annually observed "
            "cohort-A performing municipality x 120/180 minutes"
        ),
        "completeness_rule": (
            "coverage_complete is false only when the residence road anchor is missing; "
            "all service municipalities require paired isochrones before matrix creation"
        ),
        "cumulative_180_rule": (
            "Analysis 180-minute geometry is union(raw180, raw120) whenever raw180 does "
            "not cover raw120"
        ),
        "point_predicate": "intersects (boundary included)",
        "n_rows": total_rows,
        "n_reachable_rows": total_reachable,
        "n_population_municipality_years": len(coverage),
        "n_pooled_service_municipalities": len(pooled_services),
        "raw_180_geometries_requiring_union": nesting_corrections,
        "coverage_mismatches": coverage_mismatches,
        "unknown_semantics_violations": unknown_semantics_violations,
        "monotonicity_violations": monotonic_violations,
        "annual": annual_audit,
        "checks": checks,
        "input_hashes": {
            "cohorts_sha256": sha256_file(args.cohorts),
            "coverage_sha256": sha256_file(args.coverage),
            "raw_isochrones_sha256": sha256_file(args.raw_isochrones),
            "centroids_sha256": sha256_file(args.centroids),
            "snaps_sha256": sha256_file(args.snaps),
            "anchor_overrides_sha256": (
                sha256_file(args.anchor_overrides) if args.anchor_overrides else None
            ),
        },
        "artifacts": {
            "potential_access": str(args.output),
            "potential_access_sha256": sha256_file(args.output),
        },
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
