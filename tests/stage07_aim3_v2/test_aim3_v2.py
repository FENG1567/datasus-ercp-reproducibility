from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


network = load("stage07_build_aim3_network_v2")
service_area = load("stage07_build_aim3_service_area_v2")
resilience = load("stage07_build_aim3_resilience_v2")


def synthetic_cohorts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cohort": "B", "competence_month": "2021-01", "MUNIC_RES": "110001", "MUNIC_MOV": "110001", "SP_CNES": "1"},
            {"cohort": "B", "competence_month": "2021-01", "MUNIC_RES": "110002", "MUNIC_MOV": "110001", "SP_CNES": "1"},
            {"cohort": "B", "competence_month": "2021-01", "MUNIC_RES": "110002", "MUNIC_MOV": "110003", "SP_CNES": "2"},
            {"cohort": "A", "competence_month": "2021-01", "MUNIC_RES": "110001", "MUNIC_MOV": "110001", "SP_CNES": "1"},
            {"cohort": "A", "competence_month": "2021-01", "MUNIC_RES": "120001", "MUNIC_MOV": "110001", "SP_CNES": "1"},
        ]
    )


def test_inverse_flow_cost_and_betweenness_direction():
    # H is connected by high-flow edges; X by low-flow edges.  Inverse flow makes
    # the H route structurally shorter, so H must carry more shortest paths.
    edges = pd.DataFrame({"u": ["a", "h", "a", "x"], "v": ["h", "b", "x", "b"], "n": [10, 10, 1, 1]})
    centrality = network.weighted_brandes_undirected(edges, "u", "v", "n")
    reference = network._weighted_brandes_python(edges, "u", "v", "n")
    assert centrality["h"] > centrality["x"]
    assert set(centrality) == set(reference)
    assert all(np.isclose(centrality[node], reference[node]) for node in reference)


def test_both_cohorts_flow_conserved_and_projection_is_same_type():
    cnes, municipality = network.build_flow_layers(synthetic_cohorts())
    assert cnes.groupby("cohort")["n_aih"].sum().to_dict() == {"A": 2, "B": 3}
    assert municipality.groupby("cohort")["n_aih"].sum().to_dict() == {"A": 2, "B": 3}
    projection = network.make_residence_projection(municipality[municipality["cohort"] == "B"])
    assert set(projection.columns) == {"residence_a", "residence_b", "projection_weight"}
    assert (projection["projection_weight"] > 0).all()


def test_service_area_keeps_missing_coverage_missing_and_zero_is_not_need():
    _, municipality = network.build_flow_layers(synthetic_cohorts())
    municipality["layer"] = "performing_municipio"
    context = pd.DataFrame(
        {
            "res_municipio": ["110001", "110002", "110003"],
            "year": ["2021", "2021", "2021"],
            "adult_population": [100.0, 100.0, 100.0],
            "potential_coverage_120": [True, np.nan, False],
            "potential_coverage_180": [True, np.nan, True],
        }
    )
    residence, _ = service_area.build_service_area(municipality, context)
    zero = residence[(residence["cohort"] == "B") & (residence["res_municipio"] == "110003")].iloc[0]
    missing = residence[(residence["cohort"] == "B") & (residence["res_municipio"] == "110002")].iloc[0]
    assert not zero.observed_treatment and zero.n_aih == 0
    assert pd.isna(missing.potential_coverage_120)


def test_service_area_reads_stage07_coverage_column_names(tmp_path: Path):
    coverage = tmp_path / "coverage.parquet"
    pd.DataFrame(
        {
            "municipio": ["110001", "110002"],
            "year": [2021, 2021],
            "adult_population": [100.0, 200.0],
            "has_provider_120": pd.Series([True, pd.NA], dtype="boolean"),
            "has_provider_180": pd.Series([True, pd.NA], dtype="boolean"),
        }
    ).to_parquet(coverage, index=False)
    context = service_area.normalise_context(coverage, None)
    assert bool(context.loc[context["res_municipio"] == "110001", "potential_coverage_120"].iloc[0])
    assert pd.isna(
        context.loc[context["res_municipio"] == "110002", "potential_coverage_120"].iloc[0]
    )


def test_service_area_ivs_uses_2010_municipal_total_and_static_context(tmp_path: Path):
    coverage = tmp_path / "coverage.parquet"
    pd.DataFrame(
        {
            "municipio": ["110001"] * 5,
            "year": [2021, 2022, 2023, 2024, 2025],
            "adult_population": [100.0] * 5,
            "has_provider_120": [True] * 5,
            "has_provider_180": [True] * 5,
        }
    ).to_parquet(coverage, index=False)
    ivs = tmp_path / "ivs.parquet"
    pd.DataFrame(
        {
            "ano": [2000, 2010, 2010],
            "municipio": ["1100011", "1100011", "1100011"],
            "municipio_6digt": ["110001"] * 3,
            "label_cor": ["Total Cor", "Total Cor", "Branca"],
            "label_sexo": ["Total Sexo"] * 3,
            "label_sit_dom": ["Total Situação de Domicílio"] * 3,
            "ivs": [0.8, 0.2, 0.9],
        }
    ).to_parquet(ivs, index=False)

    context = service_area.normalise_context(coverage, ivs)

    assert context["res_municipio"].tolist() == ["110001"] * 5
    assert context["year"].tolist() == ["2021", "2022", "2023", "2024", "2025"]
    assert context["ivs"].tolist() == [0.2] * 5
    assert not context.duplicated(["res_municipio", "year"]).any()
    assert context.attrs["ivs_audit"] == {
        "selection_rule": {
            "year": 2010,
            "label_cor": "Total Cor",
            "label_sexo": "Total Sexo",
            "label_sit_dom": "Total Situação de Domicílio",
            "municipality_code_column": "municipio_6digt",
            "municipality_code_source": "municipio_6digt",
            "context_semantics": "static 2010 municipal-total contextual exposure carried across coverage years",
        },
        "input_rows": 3,
        "selected_rows": 1,
        "output_rows": 5,
        "unique_municipalities": 1,
        "context_semantics": "IVS 2010 is a municipality-level contextual exposure, not an individual attribute; the same value is carried across every coverage year",
    }


def test_service_area_ivs_fails_closed_on_missing_strata_or_duplicate_city(tmp_path: Path):
    coverage = tmp_path / "coverage.parquet"
    pd.DataFrame(
        {
            "municipio": ["110001"],
            "year": [2021],
            "adult_population": [100.0],
        }
    ).to_parquet(coverage, index=False)

    missing_stratum = tmp_path / "ivs_missing_stratum.parquet"
    pd.DataFrame(
        {
            "ano": [2010],
            "municipio_6digt": ["110001"],
            "label_cor": ["Total Cor"],
            "label_sexo": ["Total Sexo"],
            "ivs": [0.2],
        }
    ).to_parquet(missing_stratum, index=False)
    with pytest.raises(ValueError, match="label_sit_dom"):
        service_area.normalise_context(coverage, missing_stratum)

    duplicate_city = tmp_path / "ivs_duplicate_city.parquet"
    pd.DataFrame(
        {
            "ano": [2010, 2010],
            "municipio_6digt": ["110001", "110001"],
            "label_cor": ["Total Cor", "Total Cor"],
            "label_sexo": ["Total Sexo", "Total Sexo"],
            "label_sit_dom": ["Total Situação de Domicílio", "Total Situação de Domicílio"],
            "ivs": [0.2, 0.3],
        }
    ).to_parquet(duplicate_city, index=False)
    with pytest.raises(ValueError, match="unique by municipio_6digt"):
        service_area.normalise_context(coverage, duplicate_city)


def test_resilience_monotone_random_seed_reproducible_and_suppression_rule():
    edges = pd.DataFrame(
        {"layer": ["performing_municipio", "performing_municipio"], "cohort": ["B", "B"], "year": ["pooled", "pooled"], "res_municipio": ["110001", "110002"], "treat_municipio": ["110010", "110020"], "n_aih": [20, 10]}
    )
    ranking = resilience.observed_strength_from_frame(edges) if hasattr(resilience, "observed_strength_from_frame") else pd.DataFrame({"performing_municipio": ["110010", "110020"], "B_pooled_in_strength": [20, 10]})
    population = pd.DataFrame({"res_municipio": ["110001", "110002"], "year": ["2021", "2021"], "adult_population": [100.0, 100.0]})
    rows = []
    for threshold in (120, 180):
        for residence in ("110001", "110002"):
            for service in ("110010", "110020"):
                rows.append({"res_municipio": residence, "performing_municipio": service, "year": "2021", "threshold_minutes": threshold, "reachable": (residence, service) in {("110001", "110010"), ("110002", "110010"), ("110002", "110020")}, "coverage_complete": True, "matrix_source": "synthetic_complete_isochrone_matrix"})
    access = pd.DataFrame(rows)
    first, extra_first = resilience.run_resilience(access, population, ranking, random_replicates=1000, seed=77)
    second, extra_second = resilience.run_resilience(access, population, ranking, random_replicates=1000, seed=77)
    assert first.groupby("threshold_minutes")["adult_population_uncovered"].apply(lambda x: x.diff().dropna().ge(0).all()).all()
    pd.testing.assert_frame_equal(extra_first["random"], extra_second["random"])
    analysis = pd.DataFrame({"n_aih": [4, 5]})
    public = network.display_suppress(analysis, ["n_aih"], 5)
    assert public["n_aih_display"].tolist() == ["<5", "5"]
    assert public["n_aih"].tolist() == [4, 5]
