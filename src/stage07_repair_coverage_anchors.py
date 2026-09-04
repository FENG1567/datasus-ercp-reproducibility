#!/usr/bin/env python3
"""Create immutable, source-grounded IBGE city-seat overrides for failed ERCP coverage anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

try:
    from .stage07_rebuild_coverage import build_anchors, request_isochrone
except ImportError:  # direct execution on the analysis server
    from stage07_rebuild_coverage import build_anchors, request_isochrone


WFS_LAYER = "CGEO:APL_Localidades_Cidade"
WFS_TITLE = "Cidades — Cadastro de Localidades Brasileiras 2010"
WFS_BASE = "https://geoservicos.ibge.gov.br/geoserver/CGEO/ows"
# This value is deliberately descriptive rather than a guessed anchor type.  A
# failure table is allowed to omit ``anchor_source`` and an older anchor table
# may omit it as well; in that case the repair must retain the missingness in its
# provenance instead of silently labelling the original coordinate.
UNKNOWN_ANCHOR_SOURCE = "unknown_original_anchor_source"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat = a2 - a1
    dlon = math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def wfs_url(municipio: str) -> str:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": WFS_LAYER,
        "outputFormat": "application/json",
        "CQL_FILTER": f"cd_geocodm LIKE '{municipio}%'",
    }
    return f"{WFS_BASE}?{urllib.parse.urlencode(params)}"


def parse_city_feature(payload: bytes, municipio: str) -> dict:
    document = json.loads(payload)
    features = document.get("features") or []
    exact = [
        feature
        for feature in features
        if str(feature.get("properties", {}).get("cd_geocodm", "")).startswith(municipio)
        and str(feature.get("properties", {}).get("nm_categor", "")).casefold() == "cidade"
    ]
    if len(exact) != 1:
        raise RuntimeError(
            f"IBGE WFS returned {len(exact)} exact city features for municipality {municipio}"
        )
    feature = exact[0]
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    if geometry.get("type") == "MultiPoint" and len(coordinates or []) == 1:
        lon, lat = coordinates[0]
    elif geometry.get("type") == "Point":
        lon, lat = coordinates
    else:
        raise RuntimeError(f"unexpected IBGE city-seat geometry for {municipio}: {geometry}")
    if not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
        raise RuntimeError(f"invalid IBGE city-seat coordinates for {municipio}")
    return {
        "feature_id": feature.get("id"),
        "replacement_lat": float(lat),
        "replacement_lon": float(lon),
        **feature.get("properties", {}),
    }


def fetch_or_reuse(url: str, raw_path: Path, timeout_seconds: int) -> tuple[bytes, str]:
    if raw_path.exists():
        return raw_path.read_bytes(), "existing_immutable_raw"
    request = urllib.request.Request(url, headers={"User-Agent": "DATASUS-ERCP-Stage07/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(payload)
    return payload, "downloaded"


def _normalise_anchor_source(value: object) -> str | None:
    """Return a usable source label, treating null-like values as missing."""
    if value is None or pd.isna(value):
        return None
    source = str(value).strip()
    if not source or source.casefold() in {"nan", "none", "<na>"}:
        return None
    return source


def resolve_original_anchor_source(
    failure: pd.DataFrame,
    original: pd.DataFrame,
    municipio: str,
) -> tuple[str, str]:
    """Resolve an original-anchor source without assuming either input schema.

    The rebuilt anchor table is the canonical source for the coordinates used by
    this repair.  If it carries ``anchor_source``, that value wins.  A source in
    the failure table is used only when the canonical table has no source, and a
    deterministic sentinel is used when neither table carries one.  The second
    return value is persisted as provenance so a schema fallback is visible in
    the audit rather than being mistaken for a real anchor type.
    """
    if "anchor_source" in original.columns and municipio in original.index:
        row = original.loc[municipio]
        # build_anchors currently guarantees a unique municipality index, but
        # selecting the first row keeps this resolver tolerant of an accidental
        # duplicate in a legacy input without using Series attribute access.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        source = _normalise_anchor_source(row.get("anchor_source"))
        if source is not None:
            return source, "original_anchor_table.anchor_source"

    if "anchor_source" in failure.columns:
        values = failure.loc[
            failure["hosp_municipio"].eq(municipio), "anchor_source"
        ].map(_normalise_anchor_source).dropna().unique()
        if len(values):
            # Multiple failure rows may represent the paired 120/180 requests.
            # Sorting makes the fallback deterministic if legacy rows disagree.
            return sorted(map(str, values))[0], "failure_table.anchor_source"

    return UNKNOWN_ANCHOR_SOURCE, "deterministic_fallback:anchor_source_missing"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--centroids", required=True, type=Path)
    parser.add_argument("--snaps", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--overrides", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--gh-base", default="http://127.0.0.1:19999")
    parser.add_argument("--profile", default="ercp_car")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)

    failure = pq.read_table(args.failures).to_pandas()
    if failure.empty:
        raise RuntimeError("failure table is empty; no anchor repair is justified")
    required_failure_columns = {"hosp_municipio", "last_error"}
    if required_failure_columns - set(failure.columns):
        raise RuntimeError(
            f"failure table lacks {sorted(required_failure_columns - set(failure.columns))}"
        )
    failure["hosp_municipio"] = failure["hosp_municipio"].astype(str).str.zfill(6)
    point_not_found = failure["last_error"].fillna("").str.contains(
        "point not found", case=False, regex=False
    )
    municipalities = sorted(failure.loc[point_not_found, "hosp_municipio"].unique())
    retry_only = sorted(failure.loc[~point_not_found, "hosp_municipio"].unique())
    if not municipalities:
        raise RuntimeError(
            "no Point not found failure is eligible for an IBGE city-seat override; "
            f"retry-only municipalities={retry_only}"
        )
    original = build_anchors(args.centroids, args.snaps).set_index("municipio")
    failure_anchor_source_present = "anchor_source" in failure.columns
    anchor_source_resolution: dict[str, dict[str, object]] = {}
    records: list[dict] = []
    for municipio in municipalities:
        if municipio not in original.index:
            raise RuntimeError(f"failed provider municipality lacks original anchor: {municipio}")
        url = wfs_url(municipio)
        raw_path = args.raw_dir / f"{municipio}_APL_Localidades_Cidade.json"
        payload, acquisition = fetch_or_reuse(url, raw_path, args.timeout_seconds)
        city = parse_city_feature(payload, municipio)
        tests = {}
        for minutes in (120, 180):
            geometry, error, gh_url = request_isochrone(
                base_url=args.gh_base,
                lat=city["replacement_lat"],
                lon=city["replacement_lon"],
                minutes=minutes,
                profile=args.profile,
                timeout_seconds=args.timeout_seconds,
            )
            tests[str(minutes)] = {
                "status": "PASS" if geometry is not None else "FAIL",
                "error": error,
                "request_url": gh_url,
            }
        original_anchor_source, source_provenance = resolve_original_anchor_source(
            failure, original, municipio
        )
        anchor_source_resolution[municipio] = {
            "value": original_anchor_source,
            "provenance": source_provenance,
            "failure_table_column_present": failure_anchor_source_present,
        }
        row = original.loc[municipio]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        records.append(
            {
                "municipio": municipio,
                "original_lat": float(row["lat"]),
                "original_lon": float(row["lon"]),
                "original_anchor_source": original_anchor_source,
                "original_anchor_source_provenance": source_provenance,
                "replacement_lat": city["replacement_lat"],
                "replacement_lon": city["replacement_lon"],
                "anchor_shift_km": haversine_km(
                    float(row["lat"]),
                    float(row["lon"]),
                    city["replacement_lat"],
                    city["replacement_lon"],
                ),
                "source_name": "IBGE Geoservicos WFS",
                "source_layer": WFS_LAYER,
                "source_title": WFS_TITLE,
                "source_url": url,
                "source_accessed_at": utc_now(),
                "source_feature_id": city.get("feature_id"),
                "source_cd_geocodm": city.get("cd_geocodm"),
                "source_nm_municip": city.get("nm_municip"),
                "source_nm_categor": city.get("nm_categor"),
                "raw_path": str(raw_path),
                "raw_acquisition": acquisition,
                "response_bytes": len(payload),
                "response_sha256": sha256_bytes(payload),
                "graphhopper_120_status": tests["120"]["status"],
                "graphhopper_120_error": tests["120"]["error"],
                "graphhopper_180_status": tests["180"]["status"],
                "graphhopper_180_error": tests["180"]["error"],
            }
        )
    overrides = pd.DataFrame(records).sort_values("municipio").reset_index(drop=True)
    fallback_sources = sorted(
        {
            (
                UNKNOWN_ANCHOR_SOURCE
                if str(item["provenance"]) == "deterministic_fallback:anchor_source_missing"
                else str(item["provenance"])
            )
            for item in anchor_source_resolution.values()
            if str(item["provenance"]) != "failure_table.anchor_source"
        }
    )
    anchor_source_fallback = (
        None
        if not fallback_sources
        else fallback_sources[0]
        if len(fallback_sources) == 1
        else "mixed:" + "|".join(fallback_sources)
    )
    success = (
        overrides["graphhopper_120_status"].eq("PASS")
        & overrides["graphhopper_180_status"].eq("PASS")
    )
    status = "PASS" if success.all() and len(overrides) == len(municipalities) else "FIX"
    args.overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_parquet(args.overrides, index=False)
    audit = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": status,
        "reason": (
            "All failed provider anchors have a unique official IBGE city feature and paired GraphHopper tests passed"
            if status == "PASS"
            else "At least one failed provider anchor lacks a validated paired replacement"
        ),
        "repair_scope": municipalities,
        "retry_without_coordinate_change": retry_only,
        "n_failed_municipalities": len(municipalities),
        "n_overrides": len(overrides),
        "anchor_source_fallback": anchor_source_fallback,
        "failure_schema": {
            "columns": sorted(map(str, failure.columns)),
            "optional_anchor_source_present": failure_anchor_source_present,
            "anchor_source_resolution": anchor_source_resolution,
            "anchor_source_fallback": anchor_source_fallback,
        },
        "source": {
            "provider": "Instituto Brasileiro de Geografia e Estatistica (IBGE)",
            "service": "Geoservicos WFS",
            "layer": WFS_LAYER,
            "title": WFS_TITLE,
            "crs": "EPSG:4674",
        },
        "selection_rule": (
            "Only municipalities whose persisted failure text contains Point not found are eligible; use the unique IBGE feature "
            "with matching six-digit municipality prefix and category Cidade; accept only when both 120- and "
            "180-minute GraphHopper test requests succeed. Timeout and other failures retain their original anchor "
            "and must be retried with an extended request timeout."
        ),
        "overrides_sha256": sha256_file(args.overrides),
        "raw_sha256": {
            path.name: sha256_file(path)
            for path in sorted(args.raw_dir.glob("*_APL_Localidades_Cidade.json"))
            if path.stem[:6] in municipalities
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
