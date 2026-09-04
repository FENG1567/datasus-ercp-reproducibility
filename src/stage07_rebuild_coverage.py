from __future__ import annotations

"""Rebuild Aim 2 potential road-time coverage with auditable failure handling.

This Stage-7 repair supersedes, but does not overwrite, the original Stage-4
coverage artifact.  It uses year-specific observed provider municipalities,
official age-specific IBGE population denominators, complete paired 120/180
minute isochrones, and a cumulative 180-minute geometry that is guaranteed to
contain the corresponding 120-minute geometry.
"""

import argparse
import hashlib
import json
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from dbfread import DBF
from shapely import from_geojson, make_valid, to_geojson
from shapely.geometry import shape


YEARS = tuple(range(2021, 2026))
THRESHOLDS = (120, 180)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_geometry(geom):
    if geom is None or geom.is_empty:
        raise ValueError("empty geometry")
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"invalid isochrone geometry type: {geom.geom_type}")
    return geom


def request_isochrone(
    *,
    base_url: str,
    lat: float,
    lon: float,
    minutes: int,
    profile: str,
    timeout_seconds: int,
) -> tuple[Any | None, str | None, str]:
    url = (
        f"{base_url.rstrip('/')}/isochrone?point={lat},{lon}&profile={profile}"
        f"&buckets=1&time_limit={minutes * 60}&type=json&points_encoded=false"
        "&max_snap_distance=30000"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read())
        polygons = payload.get("polygons") or []
        if not polygons:
            return None, "response contained no polygons", url
        geometry = polygons[0].get("geometry")
        if not geometry or not geometry.get("coordinates"):
            return None, "response polygon contained no coordinates", url
        return normalise_geometry(shape(geometry)), None, url
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        return None, f"HTTPError {exc.code}: {detail}", url
    except Exception as exc:  # the exact class and message are persisted below
        return None, f"{type(exc).__name__}: {exc}", url


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def cache_path(
    cache_dir: Path,
    municipio: str,
    minutes: int,
    anchor_cache_key: str | None = None,
) -> Path:
    suffix = f"_anchor_{anchor_cache_key}" if anchor_cache_key else ""
    return cache_dir / f"{municipio}_{minutes}{suffix}.geojson"


def read_cached_geometry(path: Path):
    if not path.exists():
        return None
    return normalise_geometry(from_geojson(path.read_text(encoding="utf-8")))


def write_cached_geometry(path: Path, geom) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".geojson.tmp")
    temporary.write_text(to_geojson(geom), encoding="utf-8")
    temporary.replace(path)


def build_anchors(
    centroids: Path,
    snaps: Path,
    overrides: Path | None = None,
) -> pd.DataFrame:
    cent = pd.read_parquet(centroids)
    cent["municipio"] = cent["code7"].astype(str).str.zfill(7).str[:6]
    anchors = (
        cent[["municipio", "centroid_lat", "centroid_lon"]]
        .rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
        .drop_duplicates("municipio", keep="first")
        .set_index("municipio")
    )
    anchors["anchor_source"] = "municipality_centroid"
    anchors["anchor_cache_key"] = pd.NA
    try:
        snap = pq.read_table(snaps).to_pandas()
    except Exception:
        snap = pd.DataFrame()
    if not snap.empty:
        snap["municipio"] = snap["code6"].astype(str).str.zfill(6)
        snap = (
            snap.dropna(subset=["snap_lat", "snap_lon"])
            .rename(columns={"snap_lat": "lat", "snap_lon": "lon"})
            .drop_duplicates("municipio", keep="first")
            .set_index("municipio")
        )
        common = anchors.index.intersection(snap.index)
        anchors.loc[common, ["lat", "lon"]] = snap.loc[common, ["lat", "lon"]]
        anchors.loc[common, "anchor_source"] = "audited_road_snap"
    if overrides is not None:
        override = pq.read_table(overrides).to_pandas()
        required = {
            "municipio",
            "replacement_lat",
            "replacement_lon",
            "source_name",
            "source_url",
            "response_sha256",
        }
        missing = required - set(override.columns)
        if missing:
            raise RuntimeError(f"anchor override table lacks required columns: {sorted(missing)}")
        override["municipio"] = override["municipio"].astype(str).str.zfill(6)
        if override["municipio"].duplicated().any():
            raise RuntimeError("anchor override table contains duplicate municipalities")
        if not set(override["municipio"]).issubset(set(anchors.index)):
            unknown = sorted(set(override["municipio"]) - set(anchors.index))
            raise RuntimeError(f"anchor overrides contain unknown municipalities: {unknown}")
        numeric = override[["replacement_lat", "replacement_lon"]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = (
            numeric["replacement_lat"].between(-90, 90)
            & numeric["replacement_lon"].between(-180, 180)
        )
        if not valid.all():
            raise RuntimeError("anchor override table contains invalid coordinates")
        override = override.set_index("municipio")
        common = anchors.index.intersection(override.index)
        anchors.loc[common, "lat"] = numeric.set_axis(
            override.index, axis=0
        ).loc[common, "replacement_lat"]
        anchors.loc[common, "lon"] = numeric.set_axis(
            override.index, axis=0
        ).loc[common, "replacement_lon"]
        anchors.loc[common, "anchor_source"] = "official_ibge_city_seat_override"
        anchors.loc[common, "anchor_cache_key"] = override.loc[
            common, "response_sha256"
        ].astype(str).str[:16]
    return anchors.dropna(subset=["lat", "lon"]).reset_index()


def active_provider_years(cohorts: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pq.read_table(
        cohorts,
        columns=["cohort", "competence_month", "MUNIC_MOV"],
    ).to_pandas()
    cohort = cohort[cohort["cohort"].astype(str).eq("A")].copy()
    cohort["year"] = pd.to_numeric(
        cohort["competence_month"].astype(str).str[:4], errors="raise"
    ).astype(int)
    cohort["municipio"] = cohort["MUNIC_MOV"].astype(str).str.zfill(6)
    cohort["month"] = cohort["competence_month"].astype(str)
    by_year = cohort[["year", "municipio"]].drop_duplicates().sort_values(
        ["year", "municipio"]
    )
    sustained = (
        cohort.groupby(["year", "municipio"], as_index=False)["month"]
        .nunique()
        .rename(columns={"month": "active_months"})
    )
    return by_year, sustained


def read_population(population_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for year in YEARS:
        archive = population_dir / str(year) / f"POPSBR{str(year)[-2:]}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"missing official IBGE population archive: {archive}")
        with zipfile.ZipFile(archive) as zipped, tempfile.TemporaryDirectory(
            prefix=f"pop_{year}_"
        ) as temporary:
            members = [name for name in zipped.namelist() if name.lower().endswith(".dbf")]
            if len(members) != 1:
                raise RuntimeError(f"expected one DBF in {archive}, found {members}")
            zipped.extract(members[0], temporary)
            table = DBF(
                str(Path(temporary) / members[0]),
                encoding="latin-1",
                char_decode_errors="ignore",
                load=False,
            )
            available = {str(field).upper(): str(field) for field in table.field_names}
            population_key = "POPULACAO" if "POPULACAO" in available else "POP"
            required = {"ANO", "IDADE", "COD_MUN", population_key}
            if not required.issubset(available):
                raise RuntimeError(
                    f"unexpected IBGE fields in {archive}: {table.field_names}; "
                    f"required normalized={sorted(required)}"
                )
            source_audit[str(year)] = {
                "source_fields": table.field_names,
                "normalized_mapping": {
                    "ANO": available["ANO"], "IDADE": available["IDADE"],
                    "COD_MUN": available["COD_MUN"], "population": available[population_key],
                },
                "population_normalized_key": population_key,
                "rows_read": 0,
                "nonzero_population_rows": 0,
            }
            for row in table:
                row = {str(key).upper(): value for key, value in row.items()}
                row_year = int(str(row.get("ANO", "0")).strip() or 0)
                if row_year != year:
                    raise RuntimeError(
                        f"population year mismatch in {archive}: observed {row_year}"
                    )
                age = int(str(row.get("IDADE", "-1")).strip() or -1)
                population = float(row.get(population_key) or 0)
                source_audit[str(year)]["rows_read"] += 1
                source_audit[str(year)]["nonzero_population_rows"] += int(population > 0)
                records.append(
                    {
                        "year": year,
                        "municipio": str(row.get("COD_MUN", "")).strip().zfill(7)[:6],
                        "adult_population": population if age >= 18 else 0.0,
                        "total_population": population,
                    }
                )
    result = (
        pd.DataFrame(records)
        .groupby(["year", "municipio"], as_index=False)[
            ["adult_population", "total_population"]
        ]
        .sum()
    )
    municipality_counts = result.groupby("year")["municipio"].nunique().to_dict()
    expected_counts = {year: 5570 for year in YEARS}
    expected_counts[2025] = 5571
    if municipality_counts != expected_counts or result.duplicated(["year", "municipio"]).any():
        raise RuntimeError(
            f"unexpected IBGE population panel: observed={municipality_counts}; expected={expected_counts}"
        )
    annual = result.groupby("year", as_index=False)[["adult_population", "total_population"]].sum()
    annual_audit = {
        str(row.year): {
            "adult_population": float(row.adult_population),
            "total_population": float(row.total_population),
            "nonzero_denominators": bool(row.adult_population > 0 and row.total_population > 0),
        }
        for row in annual.itertuples(index=False)
    }
    if not all(item["nonzero_denominators"] for item in annual_audit.values()):
        raise RuntimeError("IBGE population denominator was zero after field normalization")
    return result, {
        "source": source_audit,
        "municipality_counts": {str(year): int(count) for year, count in municipality_counts.items()},
        "total_municipality_years": int(len(result)),
        "annual_denominators": annual_audit,
        "administrative_change": "Official POPSBR includes 5,570 municipalities in 2021–2024 and 5,571 in 2025; municipality 510183 is retained.",
    }


def provider_counts(
    residence: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
    active_municipalities: set[str],
) -> pd.Series:
    selected = isochrones[isochrones["hosp_municipio"].isin(active_municipalities)]
    if selected.empty:
        return pd.Series(0, index=residence["municipio"], dtype="int64")
    joined = residence.sjoin(
        selected[["hosp_municipio", "geometry"]],
        how="left",
        predicate="intersects",
    )
    counts = joined.groupby("municipio")["hosp_municipio"].nunique()
    return counts.reindex(residence["municipio"]).fillna(0).astype("int64")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--snaps", type=Path, required=True)
    parser.add_argument("--anchor-overrides", type=Path)
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isochrones", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--gh-base", default="http://127.0.0.1:19999")
    parser.add_argument("--profile", default="ercp_car")
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    args = parser.parse_args()

    anchors = build_anchors(args.centroids, args.snaps, args.anchor_overrides)
    active_year, active_months = active_provider_years(args.cohorts)
    all_providers = sorted(active_year["municipio"].unique())
    provider_anchors = anchors[anchors["municipio"].isin(all_providers)].copy()
    missing_anchor = sorted(set(all_providers) - set(provider_anchors["municipio"]))

    request_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    geometries: list[dict[str, Any]] = []
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.request_log.parent.mkdir(parents=True, exist_ok=True)
    if args.request_log.exists():
        args.request_log.unlink()

    for sequence, row in enumerate(provider_anchors.itertuples(index=False), start=1):
        for minutes in THRESHOLDS:
            cache_key = (
                None
                if pd.isna(row.anchor_cache_key)
                else str(row.anchor_cache_key)
            )
            cached = cache_path(args.cache_dir, row.municipio, minutes, cache_key)
            geom = read_cached_geometry(cached)
            source = "cache" if geom is not None else "request"
            last_error = None
            if geom is None:
                for attempt in range(1, args.max_attempts + 1):
                    started = time.monotonic()
                    geom, error, url = request_isochrone(
                        base_url=args.gh_base,
                        lat=float(row.lat),
                        lon=float(row.lon),
                        minutes=minutes,
                        profile=args.profile,
                        timeout_seconds=args.timeout_seconds,
                    )
                    record = {
                        "timestamp": utc_now(),
                        "hosp_municipio": row.municipio,
                        "bucket_min": minutes,
                        "attempt": attempt,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "status": "PASS" if geom is not None else "FAIL",
                        "error": error,
                        "url": url,
                    }
                    append_jsonl(args.request_log, record)
                    request_records.append(record)
                    last_error = error
                    if geom is not None:
                        write_cached_geometry(cached, geom)
                        break
                    if attempt < args.max_attempts:
                        time.sleep(min(args.retry_base_seconds * (2 ** (attempt - 1)), 20.0))
            if geom is None:
                failures.append(
                    {
                        "hosp_municipio": row.municipio,
                        "bucket_min": minutes,
                        "lat": float(row.lat),
                        "lon": float(row.lon),
                        "anchor_source": str(row.anchor_source),
                        "anchor_cache_key": cache_key,
                        "last_error": last_error,
                    }
                )
            else:
                geometries.append(
                    {
                        "hosp_municipio": row.municipio,
                        "bucket_min": minutes,
                        "source": source,
                        "anchor_lat": float(row.lat),
                        "anchor_lon": float(row.lon),
                        "anchor_source": str(row.anchor_source),
                        "anchor_cache_key": cache_key,
                        "geometry": geom,
                    }
                )
        if sequence % 10 == 0 or sequence == len(provider_anchors):
            print(
                f"provider municipalities complete: {sequence}/{len(provider_anchors)}; "
                f"failed pairs: {len(failures)}",
                flush=True,
            )

    failure_frame = pd.DataFrame(failures)
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    failure_frame.to_parquet(args.failures, index=False)
    pair_expected = len(provider_anchors) * len(THRESHOLDS)
    pair_complete = len(geometries) == pair_expected and not failures
    if missing_anchor or not pair_complete:
        audit = {
            "schema_version": "2.0",
            "generated_at": utc_now(),
            "status": "FIX",
            "reason": "provider anchors or paired isochrones incomplete",
            "n_provider_municipalities": len(all_providers),
            "missing_provider_anchors": missing_anchor,
            "isochrone_pairs_expected": pair_expected,
            "isochrone_pairs_observed": len(geometries),
            "failed_pairs": failures,
            "anchor_overrides": (
                int(anchors["anchor_source"].eq("official_ibge_city_seat_override").sum())
            ),
        }
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 2

    raw_iso = gpd.GeoDataFrame(geometries, geometry="geometry", crs="EPSG:4326")
    raw_iso = raw_iso.sort_values(["hosp_municipio", "bucket_min"]).reset_index(drop=True)
    args.isochrones.parent.mkdir(parents=True, exist_ok=True)
    raw_iso.to_parquet(args.isochrones, index=False)

    corrected_records: list[dict[str, Any]] = []
    nesting_corrections = 0
    for municipio, group in raw_iso.groupby("hosp_municipio"):
        geometry_120 = group.loc[group["bucket_min"].eq(120), "geometry"].iloc[0]
        geometry_180_raw = group.loc[group["bucket_min"].eq(180), "geometry"].iloc[0]
        was_nested = bool(geometry_180_raw.covers(geometry_120))
        geometry_180 = geometry_180_raw if was_nested else make_valid(
            geometry_180_raw.union(geometry_120)
        )
        nesting_corrections += int(not was_nested)
        corrected_records.extend(
            [
                {
                    "hosp_municipio": municipio,
                    "bucket_min": 120,
                    "raw_nested": was_nested,
                    "nested_union_applied": False,
                    "geometry": geometry_120,
                },
                {
                    "hosp_municipio": municipio,
                    "bucket_min": 180,
                    "raw_nested": was_nested,
                    "nested_union_applied": not was_nested,
                    "geometry": geometry_180,
                },
            ]
        )
    analysis_iso = gpd.GeoDataFrame(corrected_records, geometry="geometry", crs="EPSG:4326")
    iso_120 = analysis_iso[analysis_iso["bucket_min"].eq(120)]
    iso_180 = analysis_iso[analysis_iso["bucket_min"].eq(180)]

    residence = gpd.GeoDataFrame(
        anchors[["municipio"]].copy(),
        geometry=gpd.points_from_xy(anchors["lon"], anchors["lat"]),
        crs="EPSG:4326",
    )
    population, population_audit = read_population(args.population_dir)
    output_parts: list[pd.DataFrame] = []
    yearly_summary: dict[str, Any] = {}
    for year in YEARS:
        active = set(active_year.loc[active_year["year"].eq(year), "municipio"])
        counts_120 = provider_counts(residence, iso_120, active)
        counts_180 = provider_counts(residence, iso_180, active)
        annual = pd.DataFrame(
            {
                "year": year,
                "municipio": residence["municipio"].astype(str).values,
                "n_provider_municipalities_120": counts_120.values,
                "n_provider_municipalities_180": counts_180.values,
            }
        )
        annual["has_provider_120"] = annual["n_provider_municipalities_120"].gt(0)
        annual["has_provider_180"] = annual["n_provider_municipalities_180"].gt(0)
        annual = population[population["year"].eq(year)].merge(
            annual, on=["year", "municipio"], how="left", validate="one_to_one"
        )
        count_columns = ["n_provider_municipalities_120", "n_provider_municipalities_180"]
        annual["anchor_available"] = annual["municipio"].isin(set(anchors["municipio"]))
        # A municipality without a road anchor is unknown, not uncovered.  Keep
        # count/coverage values missing and calculate explicit all-uncovered and
        # all-covered bounds below.
        annual[count_columns] = annual[count_columns].astype("Int64")
        annual[["has_provider_120", "has_provider_180"]] = annual[
            ["has_provider_120", "has_provider_180"]
        ].astype("boolean")
        annual.loc[~annual["anchor_available"], count_columns] = pd.NA
        annual.loc[~annual["anchor_available"], ["has_provider_120", "has_provider_180"]] = pd.NA
        annual["monotonic"] = pd.Series(pd.NA, index=annual.index, dtype="boolean")
        anchored = annual["anchor_available"]
        annual.loc[anchored, "monotonic"] = annual.loc[anchored, "n_provider_municipalities_180"].ge(
            annual.loc[anchored, "n_provider_municipalities_120"]
        )
        output_parts.append(annual)
        adult_total = annual["adult_population"].sum()
        anchor_adult = annual.loc[annual["anchor_available"], "adult_population"].sum()
        missing_adult = adult_total - anchor_adult
        covered_120 = annual.loc[annual["has_provider_120"].fillna(False), "adult_population"].sum()
        covered_180 = annual.loc[annual["has_provider_180"].fillna(False), "adult_population"].sum()
        yearly_summary[str(year)] = {
            "n_active_provider_municipalities": len(active),
            "n_municipalities": len(annual),
            "n_covered_120": int(annual["has_provider_120"].sum()),
            "n_covered_180": int(annual["has_provider_180"].sum()),
            "adult_population_total": float(adult_total),
            "adult_population_with_road_anchor": float(anchor_adult),
            "adult_population_without_road_anchor": float(missing_adult),
            "adult_population_anchor_coverage": float(anchor_adult / adult_total),
            "adult_population_covered_120_anchor_denominator": float(covered_120),
            "adult_population_covered_180_anchor_denominator": float(covered_180),
            "adult_population_share_120_anchor_denominator": float(covered_120 / anchor_adult),
            "adult_population_share_180_anchor_denominator": float(covered_180 / anchor_adult),
            "adult_population_share_120_all_unanchored_uncovered": float(covered_120 / adult_total),
            "adult_population_share_120_all_unanchored_covered": float((covered_120 + missing_adult) / adult_total),
            "adult_population_share_180_all_unanchored_uncovered": float(covered_180 / adult_total),
            "adult_population_share_180_all_unanchored_covered": float((covered_180 + missing_adult) / adult_total),
            "municipality_monotonicity_violations": int((~annual.loc[anchored, "monotonic"]).sum()),
            "municipalities_without_anchor": int((~annual["anchor_available"]).sum()),
        }

    coverage = pd.concat(output_parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(args.output, index=False)
    violations = int((~coverage.loc[coverage["anchor_available"], "monotonic"]).sum())
    anchor_population_coverage = float(
        coverage.loc[coverage["anchor_available"], "adult_population"].sum() / coverage["adult_population"].sum()
    )
    status = "PASS" if violations == 0 and anchor_population_coverage >= 0.95 else "FIX"
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "estimand": (
            "Annual municipal adult population within 120/180 road-minutes of a "
            "municipality with at least one observed cohort-A ERCP AIH in that year"
        ),
        "evidence_level": "descriptive potential access; not realized access and not causal",
        "provider_unit": "performing municipality, not individual hospital",
        "n_provider_municipalities_pooled": len(all_providers),
        "provider_anchor_success": len(provider_anchors) / len(all_providers),
        "provider_anchor_sources": {
            str(key): int(value)
            for key, value in provider_anchors["anchor_source"].value_counts().items()
        },
        "isochrone_pairs_expected": pair_expected,
        "isochrone_pairs_observed": len(raw_iso),
        "isochrone_pair_success": len(raw_iso) / pair_expected,
        "raw_180_geometries_not_covering_raw_120": nesting_corrections,
        "cumulative_180_definition": (
            "For each provider municipality, analysis 180-minute geometry is the union of "
            "raw 180- and 120-minute polygons when GraphHopper contour approximation is non-nested."
        ),
        "point_predicate": "intersects (includes boundary points)",
        "population": "Official annual IBGE POPSBR age-specific DBFs; adults aged >=18 years",
        "population_read_qc": population_audit,
        "road_anchor_missingness": (
            "Municipalities without road anchors are retained with unknown potential coverage, never recoded as uncovered. "
            "Year-specific all-uncovered/all-covered bounds quantify this uncertainty."
        ),
        "adult_population_anchor_coverage_all_years": anchor_population_coverage,
        "yearly": yearly_summary,
        "monotonicity_violations": violations,
        "sustained_provider_sensitivity_available": int(
            active_months["active_months"].ge(6).sum()
        ),
        "artifacts": {
            "coverage": str(args.output),
            "raw_isochrones": str(args.isochrones),
            "failures": str(args.failures),
            "request_log": str(args.request_log),
        },
        "hashes": {
            "coverage_sha256": sha256_file(args.output),
            "raw_isochrones_sha256": sha256_file(args.isochrones),
            "cohorts_sha256": sha256_file(args.cohorts),
            "anchor_overrides_sha256": (
                sha256_file(args.anchor_overrides) if args.anchor_overrides else None
            ),
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
