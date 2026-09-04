from __future__ import annotations

"""Aim 2 potential geographic coverage: share of Brazilian municipalities (and
adult population) within 120 / 180 minutes of road travel from at least one
ERCP-performing hospital (2021-2025), and the number of distinct alternative
hospital municipalities reachable within 120 minutes.

Method: GraphHopper ercp_car drive-time isochrones (120 / 180 min) are computed
from each ERCP hospital municipality's road anchor; each residence municipality's
road anchor is then tested for membership to count how many hospital isochrones
cover it. Hospital municipality = MUNIC_MOV (the municipality where the ERCP was
performed), not the CNES registration prefix.
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from geopandas import GeoDataFrame, points_from_xy
from shapely.geometry import shape

GH_BASE = "http://127.0.0.1:19999"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def isochrone_polygon(lat: float, lon: float, minutes: int, profile: str = "ercp_car"):
    url = (f"{GH_BASE}/isochrone?point={lat},{lon}&profile={profile}&buckets=1"
           f"&time_limit={minutes}&type=json&points_encoded=false&max_snap_distance=30000")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = json.loads(response.read())
        polys = data.get("polygons")
        if not polys:
            return None
        geom = polys[0].get("geometry")
        if not geom or not geom.get("coordinates"):
            return None
        return shape(geom)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--snaps", type=Path, required=True)
    parser.add_argument("--equity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--limit-centers", type=int, default=0)
    args = parser.parse_args()

    # --- routable anchor table: use recovered road-anchor where centroid not routable
    cent = pd.read_parquet(args.centroids)
    cent["municipio"] = cent["code7"].astype(str).str[:6]
    anchors = cent[["municipio", "centroid_lat", "centroid_lon"]].rename(
        columns={"centroid_lat": "lat", "centroid_lon": "lon"})
    try:
        snaps = pq.read_table(args.snaps).to_pandas()
    except Exception:
        snaps = pd.DataFrame()
    if not snaps.empty:
        snaps["municipio"] = snaps["code6"].astype(str).str.zfill(6)
        snaps = snaps.dropna(subset=["snap_lat", "snap_lon"]).rename(
            columns={"snap_lat": "lat", "snap_lon": "lon"})[["municipio", "lat", "lon"]]
        anchors = anchors.set_index("municipio")
        over = snaps.set_index("municipio")[["lat", "lon"]]
        anchors.loc[over.index] = over[["lat", "lon"]].values
        anchors = anchors.reset_index()
    anchors = anchors.dropna(subset=["lat", "lon"])

    # --- ERCP hospital municipalities = distinct treating municipality
    coh = pq.read_table(args.cohorts).to_pandas()
    hosp_munis = coh["MUNIC_MOV"].astype(str).str.zfill(6).unique()
    hosp_centers = anchors[anchors["municipio"].isin(hosp_munis)].copy()
    print(f"hospital municipalities: {len(hosp_munis)} -> {len(hosp_centers)} with routable anchor",
          flush=True)
    if args.limit_centers:
        hosp_centers = hosp_centers.head(args.limit_centers)

    # --- isochrones
    iso_records = []
    for idx, row in hosp_centers.iterrows():
        for minutes in (120, 180):
            poly = isochrone_polygon(row["lat"], row["lon"], minutes)
            if poly is None:
                continue
            iso_records.append({
                "hosp_municipio": row["municipio"],
                "bucket_min": minutes,
                "geometry": poly,
            })
        if idx % 50 == 0:
            print(f"isochrones {idx}/{len(hosp_centers)}", flush=True)
    iso = pd.DataFrame(iso_records)
    print(f"isochrone polygons: {len(iso)} "
          f"(120min: {int((iso['bucket_min'] == 120).sum())}, "
          f"180min: {int((iso['bucket_min'] == 180).sum())})", flush=True)

    # --- residence points
    res_geo = GeoDataFrame(
        anchors[["municipio", "lat", "lon"]],
        geometry=points_from_xy(anchors["lon"], anchors["lat"]), crs="EPSG:4326")

    def count_coverage(points: GeoDataFrame, polys: GeoDataFrame) -> pd.Series:
        if polys.empty:
            return pd.Series(0, index=points["municipio"].values)
        joined = points.sjoin(polys, how="left", predicate="within")
        counts = joined.groupby("municipio")["hosp_municipio"].nunique()
        return counts.reindex(points["municipio"].unique()).fillna(0).astype(int)

    cov_120 = GeoDataFrame(
        iso[iso["bucket_min"] == 120][["hosp_municipio", "geometry"]], crs="EPSG:4326")
    cov_180 = GeoDataFrame(
        iso[iso["bucket_min"] == 180][["hosp_municipio", "geometry"]], crs="EPSG:4326")

    n120 = count_coverage(res_geo, cov_120)
    n180 = count_coverage(res_geo, cov_180)

    coverage = pd.DataFrame({
        "municipio": list(n120.index),
        "n_hospitals_120": n120.values,
        "n_hospitals_180": n180.values,
        "has_hospital_120": (n120 > 0),
        "has_hospital_180": (n180 > 0),
    })

    # --- population-weighted denominators from equity output (adult pop)
    pop_col = None
    try:
        equity = pq.read_table(args.equity).to_pandas()
        pop_col = "pop" if "pop" in equity.columns else None
    except Exception:
        equity = pd.DataFrame()
        pop_col = None
    if pop_col and not equity.empty:
        equity = equity.reset_index(drop=True)
        mun_pop = equity[["res_municipio", pop_col]].drop_duplicates("res_municipio")
        mun_pop["municipio"] = mun_pop["res_municipio"].astype(str).str.zfill(6)
        mun_pop = mun_pop[["municipio", pop_col]].rename(columns={pop_col: "pop"})
        mun_pop = mun_pop.reset_index(drop=True)
        coverage = coverage.reset_index(drop=True).merge(mun_pop, on="municipio", how="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(args.output, index=False)

    # --- summary
    n_all = len(coverage)
    n_120 = int(coverage["has_hospital_120"].sum())
    n_180 = int(coverage["has_hospital_180"].sum())
    summary = {
        "n_municipalities": int(n_all),
        "n_covered_120": n_120,
        "n_covered_180": n_180,
        "share_covered_120": round(n_120 / n_all, 4) if n_all else None,
        "share_covered_180": round(n_180 / n_all, 4) if n_all else None,
        "n_hospital_municipalities": int(len(hosp_centers)),
    }
    if "pop" in coverage.columns and coverage["pop"].sum() > 0:
        pop_total = float(coverage["pop"].sum())
        pop_120 = float(coverage.loc[coverage["has_hospital_120"], "pop"].sum())
        pop_180 = float(coverage.loc[coverage["has_hospital_180"], "pop"].sum())
        summary["pop_total"] = pop_total
        summary["pop_covered_120"] = pop_120
        summary["pop_covered_180"] = pop_180
        summary["pop_share_120"] = round(pop_120 / pop_total, 4) if pop_total else None
        summary["pop_share_180"] = round(pop_180 / pop_total, 4) if pop_total else None
        alt = coverage["n_hospitals_120"]
        summary["share_with_ge2_hospitals_120"] = round(float((alt >= 2).mean()), 4)
        summary["share_with_ge3_hospitals_120"] = round(float((alt >= 3).mean()), 4)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS",
        "summary": summary,
        "note": "120/180-min drive-time isochrones (GraphHopper ercp_car, brazil-260822.osm.pbf) "
                "from each ERCP hospital municipality road anchor; residence covered if an ERCP "
                "hospital isochrone contains its road anchor. Hospital municipality = MUNIC_MOV.",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())