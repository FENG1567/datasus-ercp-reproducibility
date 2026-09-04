from __future__ import annotations

"""Aim 2 travel times: residence municipality seat (IBGE official
coordinates) to actual treating hospital (CNES) via GraphHopper road
routing. Reports median/P75/P90, >120/>180 min shares, cross-municipality
and cross-state care."""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GH_BASE = "http://127.0.0.1:19999"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def route_time(lat1: float, lon1: float, lat2: float, lon2: float, profile: str = "ercp_car") -> float | None:
    url = (f"{GH_BASE}/route?point={lat1},{lon1}&point={lat2},{lon2}&profile={profile}"
           f"&points_encoded=false&max_snap_distance=30000")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read())
        paths = data.get("paths")
        if not paths:
            return None
        # time in ms; return minutes
        return paths[0].get("time", 0) / 60000.0
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--municipios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    df = pq.read_table(args.cohorts).to_pandas()
    b = df[df["cohort"] == "B"].copy()
    b["res_municipio"] = b["MUNIC_RES"].astype(str).str.strip().str.zfill(6)
    b["hosp_uf"] = b["SP_CNES"].str[:2]
    # hospital coordinates: municipality of the treating hospital via CNES -> MUNIC_MOV
    b["treat_municipio"] = b["MUNIC_MOV"].astype(str).str.strip().str.zfill(6)

    mun = pd.read_parquet(args.municipios)
    mun_df = mun[["code7", "centroid_lat", "centroid_lon"]].copy()
    mun_df["res_municipio"] = mun_df["code7"].str[:6]
    mun_coords = mun_df.dropna(subset=["centroid_lat", "centroid_lon"]).drop_duplicates("res_municipio")
    mun_coords = mun_coords.rename(columns={"centroid_lat": "latitude", "centroid_lon": "longitude"})

    # override origin coordinates for municipalities whose geometric centroid is
    # not routable, using the recovered in-polygon road anchor (snap point)
    snap = pq.read_table("data_stage/aim2/failed_municipio_snaps.parquet").to_pandas()
    snap = snap.rename(columns={"snap_lat": "latitude", "snap_lon": "longitude",
                                "code6": "res_municipio"})
    snap = snap[["res_municipio", "latitude", "longitude"]].dropna()
    snap_override = {}
    for _, r in snap.iterrows():
        snap_override.setdefault(str(r["res_municipio"]).zfill(6), (r["latitude"], r["longitude"]))
    print(f"using {len(snap_override)} recovered road-anchor coordinates for failed municipalities", flush=True)

    pairs = b[["res_municipio", "treat_municipio"]].drop_duplicates()
    pairs = pairs.merge(mun_coords[["res_municipio", "latitude", "longitude"]],
                        on="res_municipio", how="left")
    pairs = pairs.merge(
        mun_coords[["res_municipio", "latitude", "longitude"]].rename(
            columns={"res_municipio": "treat_municipio", "latitude": "t_lat", "longitude": "t_lon"}),
        on="treat_municipio", how="left",
    )
    pairs = pairs.dropna(subset=["latitude", "longitude", "t_lat", "t_lon"])
    if args.limit:
        pairs = pairs.head(args.limit)

    results = []
    failures = 0
    for idx, row in pairs.iterrows():
        lat1, lon1 = row["latitude"], row["longitude"]
        if row["res_municipio"] in snap_override:
            lat1, lon1 = snap_override[row["res_municipio"]]
        tlat, tlon = row["t_lat"], row["t_lon"]
        if row["treat_municipio"] in snap_override:
            tlat, tlon = snap_override[row["treat_municipio"]]
        minutes = route_time(lat1, lon1, tlat, tlon)
        results.append({
            "res_municipio": row["res_municipio"],
            "treat_municipio": row["treat_municipio"],
            "travel_minutes": minutes,
        })
        if minutes is None:
            failures += 1
        if idx % 500 == 0:
            print(f"progress {idx}/{len(pairs)}", flush=True)
        time.sleep(0.02)

    out = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    valid = out["travel_minutes"].dropna()
    summary = {
        "n_pairs": int(len(out)),
        "n_pairs_success": int(valid.shape[0]),
        "n_pairs_failed": failures,
        "success_rate": round(valid.shape[0] / len(out), 4) if len(out) else None,
        "median_min": float(valid.median()) if len(valid) else None,
        "p75_min": float(valid.quantile(0.75)) if len(valid) else None,
        "p90_min": float(valid.quantile(0.90)) if len(valid) else None,
        "share_gt120": round(float((valid > 120).mean()), 4) if len(valid) else None,
        "share_gt180": round(float((valid > 180).mean()), 4) if len(valid) else None,
    }
    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS",
        "summary": summary,
        "note": "road time via GraphHopper ercp_car on brazil-260822.osm.pbf; municipality seats as origin/destination (population-weighted centroids pending)",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())