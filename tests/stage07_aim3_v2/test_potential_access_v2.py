from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location(
    "stage07_build_aim3_potential_access_v2",
    SRC / "stage07_build_aim3_potential_access_v2.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reachable_pairs_uses_annual_services_and_boundary_intersects() -> None:
    residence = gpd.GeoDataFrame(
        {"res_municipio": ["000001", "000002", "000003"]},
        geometry=[Point(0, 0), Point(1, 0), Point(3, 0)],
        crs="EPSG:4326",
    )
    isochrones = gpd.GeoDataFrame(
        {
            "performing_municipio": ["100001", "100002"],
            "threshold_minutes": [120, 120],
        },
        geometry=[Point(0, 0).buffer(1), Point(3, 0).buffer(0.25)],
        crs="EPSG:4326",
    )
    pairs = MODULE.reachable_pairs(
        residence, ["100001"], isochrones, threshold=120
    )
    assert set(map(tuple, pairs.to_numpy())) == {
        ("000001", "100001"),
        ("000002", "100001"),
    }


def test_z6_preserves_six_digit_codes() -> None:
    assert MODULE.z6(1234) == "001234"
    assert MODULE.z6("510183") == "510183"
