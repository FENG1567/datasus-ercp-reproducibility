from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# The source is deployed in the analysis environment, where statsmodels is a
# required dependency.  This local authoring environment intentionally has no
# statsmodels installation; lightweight import stubs allow the pure numerical
# and freeze-contract tests below to run without pretending that a model fit
# was executed locally.  When statsmodels is available, the Holm check below
# uses its actual implementation.
try:
    from statsmodels.stats.multitest import multipletests
    HAVE_STATSMODELS = True
except ModuleNotFoundError:
    HAVE_STATSMODELS = False
    statsmodels = types.ModuleType("statsmodels")
    statsmodels_api = types.ModuleType("statsmodels.api")
    statsmodels_formula = types.ModuleType("statsmodels.formula")
    statsmodels_formula_api = types.ModuleType("statsmodels.formula.api")
    statsmodels_stats = types.ModuleType("statsmodels.stats")
    statsmodels_multitest = types.ModuleType("statsmodels.stats.multitest")

    def unavailable_multipletests(*args, **kwargs):
        raise RuntimeError("statsmodels is required to run the primary-family analysis")

    statsmodels_multitest.multipletests = unavailable_multipletests
    sys.modules.update({
        "statsmodels": statsmodels,
        "statsmodels.api": statsmodels_api,
        "statsmodels.formula": statsmodels_formula,
        "statsmodels.formula.api": statsmodels_formula_api,
        "statsmodels.stats": statsmodels_stats,
        "statsmodels.stats.multitest": statsmodels_multitest,
    })
    patsy = types.ModuleType("patsy")

    def unavailable_design_matrix(*args, **kwargs):
        raise RuntimeError("patsy is required to run the primary-family analysis")

    patsy.build_design_matrices = unavailable_design_matrix
    sys.modules["patsy"] = patsy


ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load("stage07_build_aim2_primary_family_v2")
target = load("stage07_build_aim2_primary_family_v3")


def finite_difference_difference(params, x0, x1, weights, offset, index, epsilon=1e-6):
    plus = params.copy()
    minus = params.copy()
    plus[index] += epsilon
    minus[index] -= epsilon

    def difference(coefficients):
        low = np.average(np.exp(x0 @ coefficients + offset), weights=weights)
        high = np.average(np.exp(x1 @ coefficients + offset), weights=weights)
        return high - low

    return (difference(plus) - difference(minus)) / (2 * epsilon)


@pytest.mark.parametrize(
    "params,x0,x1,weights,offset,covariance",
    [
        # utilisation-style Poisson log-link, including a per-100k offset
        (
            np.array([-7.0, 0.35, 0.10]),
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
            np.array([100.0, 250.0, 300.0]),
            np.repeat(np.log(100000.0), 3),
            np.array([[0.010, 0.001, 0.000], [0.001, 0.020, 0.001], [0.000, 0.001, 0.010]]),
        ),
        # Gamma/log realised treated-flow time with n_aih standardisation
        (
            np.array([4.2, 0.15, -0.08]),
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
            np.array([3.0, 7.0, 10.0]),
            np.zeros(3),
            np.array([[0.010, 0.000, 0.000], [0.000, 0.004, 0.000], [0.000, 0.000, 0.010]]),
        ),
        # modified-Poisson potential coverage (probability scale)
        (
            np.array([-1.3, 0.12, 0.04]),
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
            np.array([1000.0, 3000.0, 2000.0]),
            np.zeros(3),
            np.array([[0.002, 0.000, 0.000], [0.000, 0.003, 0.000], [0.000, 0.000, 0.002]]),
        ),
    ],
)
def test_log_link_absolute_difference_gradient_and_ci_match_numerical_delta(
    params, x0, x1, weights, offset, covariance
):
    result = target.rank_standardised_log_delta(
        params=params, covariance=covariance, x_rank0=x0, x_rank1=x1, weights=weights, offset=offset
    )
    analytic_gradient = (
        np.average(np.exp(x1 @ params + offset)[:, None] * x1, axis=0, weights=weights)
        - np.average(np.exp(x0 @ params + offset)[:, None] * x0, axis=0, weights=weights)
    )
    numerical_gradient = np.array(
        [finite_difference_difference(params, x0, x1, weights, offset, i) for i in range(len(params))]
    )
    assert np.allclose(analytic_gradient, numerical_gradient, rtol=1e-6, atol=1e-7)
    expected_variance = float(analytic_gradient @ covariance @ analytic_gradient)
    expected_se = np.sqrt(expected_variance)
    expected_difference = result["rank1_mean"] - result["rank0_mean"]
    assert result["absolute_difference_delta_se"] == pytest.approx(expected_se)
    assert result["absolute_difference_ci_low"] == pytest.approx(expected_difference - target.Z_975 * expected_se)
    assert result["absolute_difference_ci_high"] == pytest.approx(expected_difference + target.Z_975 * expected_se)
    assert result["absolute_difference_ci_finite"]


def test_holm_has_exactly_three_raw_tests_and_matches_statsmodels():
    raw = np.array([0.03, 0.001, 0.04])
    manual = target.holm_manual(raw)
    if HAVE_STATSMODELS:
        _, expected, _, _ = multipletests(raw, method="holm")
    else:
        # Independent known-answer reference for the local dependency-free run.
        expected = np.array([0.06, 0.003, 0.06])
    assert len(raw) == 3
    assert np.allclose(manual, expected, rtol=0, atol=1e-12)
    assert manual.tolist() == pytest.approx([0.06, 0.003, 0.06])


def test_v3_freeze_hash_is_identical_to_v2_and_tamper_is_rejected(tmp_path):
    assert target.canonical_hash(target.implementation_payload()) == v2.canonical_hash(v2.implementation_payload())
    freeze = tmp_path / "freeze.json"
    assert target.write_freeze(freeze) == 0
    assert target.validate_freeze(freeze)["freeze_payload_sha256"] == target.canonical_hash(target.implementation_payload())
    document = json.loads(freeze.read_text(encoding="utf-8"))
    document["freeze_payload_sha256"] = "0" * 64
    freeze.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze payload"):
        target.validate_freeze(freeze)


def test_rank_deficiency_nonfinite_ci_and_weighted_cluster_warning_are_gated():
    item = {
        "converged": True,
        "finite_estimates": True,
        "finite_covariance": True,
        "finite_ratio_ci": True,
        "design_full_rank": False,
        "clusters": 3,
        "finite_pearson_chi2_over_df": True,
        "absolute_difference_variance_valid": True,
        "absolute_difference_ci_finite": False,
        "weighted_cluster_covariance_unsupported": True,
        "pvalue_raw": 0.01,
    }
    checks = target.item_model_checks(item)
    assert not checks["design_full_rank"]
    assert not checks["absolute_difference_ci_finite"]
    assert not checks["weighted_cluster_covariance_supported"]
    warnings_found = [{"category": "SpecificationWarning", "message": "cov_type not fully supported with freq_weights"}]
    assert target.has_unsupported_weighted_cluster_warning(warnings_found)
    bad_covariance = np.array([[np.nan, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        target.rank_standardised_log_delta(
            params=np.array([0.0, 0.1]), covariance=bad_covariance,
            x_rank0=np.array([[1.0, 0.0]]), x_rank1=np.array([[1.0, 1.0]]),
            weights=np.array([1.0]), offset=np.array([0.0]),
        )


@pytest.mark.skipif(not HAVE_STATSMODELS, reason="requires statsmodels")
def test_utilisation_offset_and_explicit_cluster_covariance_are_used():
    data = pd.DataFrame(
        {
            "n": [1, 2, 0, 3, 1, 4, 2, 1],
            "ivs_ridit": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "cluster": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )
    offset = np.log(np.repeat(100.0, len(data)))
    fit, warning_records = target.fit_glm_checked(
        formula="n ~ ivs_ridit",
        data=data,
        family=target.sm.families.Poisson(),
        groups=data["cluster"],
        offset=offset,
    )
    assert warning_records == []
    assert np.array_equal(fit.model.offset, offset)
    assert fit.covariance_audit["method"] == "explicit GLM score-cluster sandwich"
    assert fit.covariance_audit["n_clusters"] == 4
    assert np.isfinite(fit.cov_params()).all().all()
    reference = target.smf.glm(
        "n ~ ivs_ridit",
        data=data,
        family=target.sm.families.Poisson(),
        offset=offset,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["cluster"]})
    assert np.allclose(
        np.asarray(fit.cov_params()), np.asarray(reference.cov_params()), rtol=1e-8, atol=1e-10
    )


def test_coverage_population_gate_is_95_percent_not_complete_case_perfection():
    source = (ROOT / "src" / "stage07_build_aim2_primary_family_v3.py").read_text(encoding="utf-8")
    assert "model_population_coverage_ge_95pct" in source
    assert "model_coverage >= 0.95" in source
    assert "model_population_complete" not in source
