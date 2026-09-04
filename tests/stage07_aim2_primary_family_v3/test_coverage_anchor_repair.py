import json
from pathlib import Path

import pandas as pd

from src.stage07_rebuild_coverage import build_anchors, cache_path
from src import stage07_repair_coverage_anchors as repair
from src.stage07_repair_coverage_anchors import haversine_km, parse_city_feature


def test_ibge_city_feature_parser_and_distance():
    payload = json.dumps(
        {
            "features": [
                {
                    "id": "APL_Localidades_Cidade.13714",
                    "geometry": {"type": "MultiPoint", "coordinates": [[-46.51591568, -18.59257082]]},
                    "properties": {
                        "cd_geocodm": "3148004",
                        "nm_municip": "Patos De Minas",
                        "nm_categor": "Cidade",
                    },
                }
            ]
        }
    ).encode()
    result = parse_city_feature(payload, "314800")
    assert result["replacement_lat"] == -18.59257082
    assert result["replacement_lon"] == -46.51591568
    assert haversine_km(-18.635157497, -46.105547048, result["replacement_lat"], result["replacement_lon"]) > 40


def test_override_is_unique_and_gets_separate_cache_key(tmp_path: Path):
    centroids = tmp_path / "centroids.parquet"
    snaps = tmp_path / "snaps.parquet"
    overrides = tmp_path / "overrides.parquet"
    pd.DataFrame(
        {"code7": [3148004], "centroid_lat": [-18.635157497], "centroid_lon": [-46.105547048]}
    ).to_parquet(centroids, index=False)
    pd.DataFrame(columns=["code6", "snap_lat", "snap_lon"]).to_parquet(snaps, index=False)
    pd.DataFrame(
        {
            "municipio": ["314800"],
            "replacement_lat": [-18.59257082],
            "replacement_lon": [-46.51591568],
            "source_name": ["IBGE Geoservicos WFS"],
            "source_url": ["https://example.invalid/ibge"],
            "response_sha256": ["a" * 64],
        }
    ).to_parquet(overrides, index=False)
    anchors = build_anchors(centroids, snaps, overrides)
    assert anchors.loc[0, "anchor_source"] == "official_ibge_city_seat_override"
    assert anchors.loc[0, "anchor_cache_key"] == "a" * 16
    assert cache_path(tmp_path, "314800", 120).name == "314800_120.geojson"
    assert cache_path(tmp_path, "314800", 120, "a" * 16).name == f"314800_120_anchor_{'a' * 16}.geojson"


def test_repair_accepts_missing_failure_anchor_source_and_keeps_timeout_only_rows_unchanged(
    tmp_path: Path, monkeypatch
):
    failures = tmp_path / "failures.parquet"
    raw_dir = tmp_path / "raw"
    overrides = tmp_path / "overrides.parquet"
    audit_path = tmp_path / "audit.json"
    centroids = tmp_path / "centroids.parquet"
    snaps = tmp_path / "snaps.parquet"

    point_municipalities = ["314800", "330010", "510790"]
    timeout_only = "999999"
    failure_rows = pd.DataFrame(
        {
            "hosp_municipio": point_municipalities + [timeout_only],
            "last_error": [
                "Point not found for 120 minutes",
                "Point not found for 180 minutes",
                "Point not found for 120 minutes",
                "Read timed out",
            ],
        }
    )
    # This is the formal final-failure shape: anchor_source is optional and is
    # intentionally absent.  The legacy anchor shape below also omits it so
    # the regression reaches the old Series.anchor_source crash directly.
    failure_rows.to_parquet(failures, index=False)

    original = pd.DataFrame(
        {
            "municipio": point_municipalities,
            "lat": [-18.635157497, -22.525, -15.601],
            "lon": [-46.105547048, -44.10, -56.095],
        }
    )
    monkeypatch.setattr(repair, "build_anchors", lambda *_args: original)

    city_coordinates = {
        "314800": (-18.59257082, -46.51591568),
        "330010": (-22.520, -44.100),
        "510790": (-15.600, -56.100),
    }
    raw_dir.mkdir()
    for municipio, (lat, lon) in city_coordinates.items():
        payload = {
            "features": [
                {
                    "id": f"APL_Localidades_Cidade.{municipio}",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "cd_geocodm": f"{municipio}1",
                        "nm_municip": municipio,
                        "nm_categor": "Cidade",
                    },
                }
            ]
        }
        (raw_dir / f"{municipio}_APL_Localidades_Cidade.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    requests = []

    def fake_request_isochrone(*, base_url, lat, lon, minutes, profile, timeout_seconds):
        requests.append((lat, lon, minutes, profile, timeout_seconds))
        return object(), None, f"{base_url}/isochrone?time_limit={minutes * 60}"

    monkeypatch.setattr(repair, "request_isochrone", fake_request_isochrone)

    result = repair.main(
        [
            "--failures",
            str(failures),
            "--centroids",
            str(centroids),
            "--snaps",
            str(snaps),
            "--raw-dir",
            str(raw_dir),
            "--overrides",
            str(overrides),
            "--audit",
            str(audit_path),
        ]
    )

    assert result == 0
    repaired = pd.read_parquet(overrides)
    assert repaired["municipio"].tolist() == point_municipalities
    assert repaired["original_anchor_source"].tolist() == [
        repair.UNKNOWN_ANCHOR_SOURCE
    ] * len(point_municipalities)
    assert repaired["original_anchor_source_provenance"].tolist() == [
        "deterministic_fallback:anchor_source_missing"
    ] * len(point_municipalities)
    assert repaired["graphhopper_120_status"].eq("PASS").all()
    assert repaired["graphhopper_180_status"].eq("PASS").all()
    assert len(requests) == len(point_municipalities) * 2

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["repair_scope"] == point_municipalities
    assert audit["retry_without_coordinate_change"] == [timeout_only]
    assert audit["failure_schema"]["optional_anchor_source_present"] is False
    assert audit["anchor_source_fallback"] == repair.UNKNOWN_ANCHOR_SOURCE
    assert audit["failure_schema"]["anchor_source_fallback"] == repair.UNKNOWN_ANCHOR_SOURCE
    assert set(audit["failure_schema"]["anchor_source_resolution"]) == set(
        point_municipalities
    )
