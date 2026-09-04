from __future__ import annotations

"""Retryable, flow-weighted municipality-to-municipality road-time analysis."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()
    return float(sorted_values[np.searchsorted(cumulative, probability, side="left")])


def route_time(
    *,
    base_url: str,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    profile: str,
    timeout_seconds: int,
) -> tuple[float | None, str | None, str]:
    url = (
        f"{base_url.rstrip('/')}/route?point={origin_lat},{origin_lon}"
        f"&point={destination_lat},{destination_lon}&profile={profile}"
        "&points_encoded=false&max_snap_distance=30000"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read())
        paths = payload.get("paths") or []
        if not paths:
            return None, "response contained no path", url
        milliseconds = paths[0].get("time")
        if milliseconds is None:
            return None, "path contained no time", url
        return float(milliseconds) / 60000.0, None, url
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        return None, f"HTTPError {exc.code}: {detail}", url
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", url


def build_anchors(centroids: Path, snaps: Path) -> pd.DataFrame:
    centre = pd.read_parquet(centroids)
    centre["municipio"] = centre["code7"].astype(str).str.zfill(7).str[:6]
    anchors = (
        centre[["municipio", "centroid_lat", "centroid_lon"]]
        .rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
        .drop_duplicates("municipio")
        .set_index("municipio")
    )
    try:
        snap = pq.read_table(snaps).to_pandas()
    except Exception:
        snap = pd.DataFrame()
    if not snap.empty:
        snap["municipio"] = snap["code6"].astype(str).str.zfill(6)
        snap = (
            snap.dropna(subset=["snap_lat", "snap_lon"])
            .rename(columns={"snap_lat": "lat", "snap_lon": "lon"})
            .drop_duplicates("municipio")
            .set_index("municipio")
        )
        common = anchors.index.intersection(snap.index)
        anchors.loc[common, ["lat", "lon"]] = snap.loc[common, ["lat", "lon"]]
    return anchors.dropna().reset_index()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--snaps", type=Path, required=True)
    parser.add_argument("--prior-routes", type=Path)
    parser.add_argument("--prior-cache-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--gh-base", default="http://127.0.0.1:19999")
    parser.add_argument("--profile", default="ercp_car")
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    cohort = pq.read_table(
        args.cohorts,
        columns=["cohort", "competence_month", "MUNIC_RES", "MUNIC_MOV"],
    ).to_pandas()
    cohort = cohort[cohort["cohort"].astype(str).eq("B")].copy()
    cohort["year"] = cohort["competence_month"].astype(str).str[:4].astype(int)
    cohort["res_municipio"] = cohort["MUNIC_RES"].astype(str).str.zfill(6)
    cohort["treat_municipio"] = cohort["MUNIC_MOV"].astype(str).str.zfill(6)
    flows = (
        cohort.groupby(["year", "res_municipio", "treat_municipio"], as_index=False)
        .size()
        .rename(columns={"size": "n_aih"})
    )
    pairs = flows[["res_municipio", "treat_municipio"]].drop_duplicates()
    anchors = build_anchors(args.centroids, args.snaps)
    origin = anchors.rename(
        columns={"municipio": "res_municipio", "lat": "origin_lat", "lon": "origin_lon"}
    )
    destination = anchors.rename(
        columns={
            "municipio": "treat_municipio",
            "lat": "destination_lat",
            "lon": "destination_lon",
        }
    )
    pairs = pairs.merge(origin, on="res_municipio", how="left", validate="many_to_one")
    pairs = pairs.merge(destination, on="treat_municipio", how="left", validate="many_to_one")
    missing_anchor = pairs[
        pairs[["origin_lat", "origin_lon", "destination_lat", "destination_lon"]]
        .isna()
        .any(axis=1)
    ].copy()

    cache: dict[tuple[str, str], float] = {}
    cache_provenance = None
    if args.prior_routes and args.prior_routes.exists():
        prior = pd.read_parquet(args.prior_routes)
        prior = prior.dropna(subset=["travel_minutes"])
        for row in prior.itertuples(index=False):
            cache[(str(row.res_municipio).zfill(6), str(row.treat_municipio).zfill(6))] = float(
                row.travel_minutes
            )
        if args.prior_cache_audit and args.prior_cache_audit.exists():
            cache_provenance = json.loads(args.prior_cache_audit.read_text(encoding="utf-8"))
    if args.request_log.exists():
        raise FileExistsError(f"refusing to overwrite existing route request log: {args.request_log}")
    results = []
    failures = []
    valid_pairs = pairs.dropna(
        subset=["origin_lat", "origin_lon", "destination_lat", "destination_lon"]
    )
    for sequence, row in enumerate(valid_pairs.itertuples(index=False), start=1):
        key = (row.res_municipio, row.treat_municipio)
        minutes = cache.get(key)
        source = "prior_stage04_cache" if minutes is not None else "request"
        last_error = None
        if minutes is None:
            for attempt in range(1, args.max_attempts + 1):
                started = time.monotonic()
                minutes, error, url = route_time(
                    base_url=args.gh_base,
                    origin_lat=float(row.origin_lat),
                    origin_lon=float(row.origin_lon),
                    destination_lat=float(row.destination_lat),
                    destination_lon=float(row.destination_lon),
                    profile=args.profile,
                    timeout_seconds=args.timeout_seconds,
                )
                record = {
                    "timestamp": utc_now(),
                    "res_municipio": row.res_municipio,
                    "treat_municipio": row.treat_municipio,
                    "attempt": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "status": "PASS" if minutes is not None else "FAIL",
                    "error": error,
                    "url": url,
                }
                append_jsonl(args.request_log, record)
                last_error = error
                if minutes is not None:
                    break
                if attempt < args.max_attempts:
                    time.sleep(min(2 ** attempt, 20))
        results.append(
            {
                "res_municipio": row.res_municipio,
                "treat_municipio": row.treat_municipio,
                "travel_minutes": minutes,
                "route_source": source,
            }
        )
        if minutes is None:
            failures.append(
                {
                    "res_municipio": row.res_municipio,
                    "treat_municipio": row.treat_municipio,
                    "last_error": last_error,
                }
            )
        if sequence % 250 == 0 or sequence == len(valid_pairs):
            print(f"route pairs complete: {sequence}/{len(valid_pairs)}", flush=True)

    route = pd.DataFrame(results)
    pair_year = flows.merge(
        route,
        on=["res_municipio", "treat_municipio"],
        how="left",
        validate="many_to_one",
    )
    pair_year["cross_municipality"] = pair_year["res_municipio"].ne(
        pair_year["treat_municipio"]
    )
    pair_year["cross_state"] = pair_year["res_municipio"].str[:2].ne(
        pair_year["treat_municipio"].str[:2]
    )
    valid = pair_year.dropna(subset=["travel_minutes"]).copy()
    values = valid["travel_minutes"].to_numpy(dtype=float)
    weights = valid["n_aih"].to_numpy(dtype=float)
    total_aih = int(pair_year["n_aih"].sum())
    routed_aih = int(valid["n_aih"].sum())
    summary = {
        "n_unique_pairs": int(len(pairs)),
        "n_pair_years": int(len(pair_year)),
        "n_pairs_routed": int(route["travel_minutes"].notna().sum()),
        "n_pairs_failed": int(route["travel_minutes"].isna().sum() + len(missing_anchor)),
        "pair_success_rate": float(route["travel_minutes"].notna().sum() / len(pairs)),
        "n_aih_total": total_aih,
        "n_aih_with_route": routed_aih,
        "flow_weighted_success_rate": routed_aih / total_aih if total_aih else None,
        "flow_weighted_median_min": weighted_quantile(values, weights, 0.50),
        "flow_weighted_p75_min": weighted_quantile(values, weights, 0.75),
        "flow_weighted_p90_min": weighted_quantile(values, weights, 0.90),
        "flow_weighted_share_gt120": float(weights[values > 120].sum() / weights.sum()),
        "flow_weighted_share_gt180": float(weights[values > 180].sum() / weights.sum()),
        "flow_weighted_cross_municipality_share": float(
            valid.loc[valid["cross_municipality"], "n_aih"].sum() / weights.sum()
        ),
        "flow_weighted_cross_state_share": float(
            valid.loc[valid["cross_state"], "n_aih"].sum() / weights.sum()
        ),
        "unweighted_pair_median_min_supportive": float(np.median(values)),
        "cached_pair_count": int(len(cache)),
        "cached_pairs_used": int(sum(item["route_source"] == "prior_stage04_cache" for item in results)),
    }
    checks = {
        "anchor_coverage_ge_99pct": 1 - len(missing_anchor) / len(pairs) >= 0.99,
        "flow_weighted_route_success_ge_95pct": summary["flow_weighted_success_rate"] >= 0.95,
        "flow_conservation": total_aih == len(cohort),
        "travel_time_nonnegative": bool(valid["travel_minutes"].ge(0).all()),
    }
    status = "PASS" if all(checks.values()) else "DOWNGRADE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    pair_year.to_parquet(args.output, index=False)
    pd.concat(
        [pd.DataFrame(failures), missing_anchor.assign(last_error="missing road anchor")],
        ignore_index=True,
        sort=False,
    ).to_parquet(args.failures, index=False)
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "checks": checks,
        "summary": summary,
        "estimand": "Observed cohort-B flow-weighted road travel time",
        "geographic_unit": (
            "Residence-municipality road anchor to treating-municipality road anchor; "
            "not patient address and not the hospital's exact street coordinate."
        ),
        "evidence_level": "descriptive realized treatment flow; not referral and not causal",
        "prior_cache_provenance": cache_provenance,
        "input_hashes": {
            "cohorts_sha256": sha256_file(args.cohorts),
            "centroids_sha256": sha256_file(args.centroids),
            "snaps_sha256": sha256_file(args.snaps),
            "prior_routes_sha256": sha256_file(args.prior_routes) if args.prior_routes and args.prior_routes.exists() else None,
        },
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
