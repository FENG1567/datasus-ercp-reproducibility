from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module); return module

BUILD, FIT = load("stage07_build_aim4_analytic_v2"), load("stage07_fit_aim4_outcomes_v2")


def synthetic_analytic(n: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(22); i = np.arange(n); volume = (i % 48).astype(float)
    probability = 1 / (1 + np.exp(-(-3.0 + .055 * volume + .02 * ((i % 50) - 25))))
    death = rng.binomial(1, probability)
    return pd.DataFrame({"analysis_row_id": i, "cnes7": [f"{1000000 + j % 18:07d}" for j in i], "state_provider": rng.choice(["35", "33"], n), "calendar_month": rng.choice(["202201", "202202", "202203", "202204", "202205", "202206"], n), "year": "2022", "death_valid": True, "in_hospital_death": death, "trailing12_complete": True, "trailing12_a_unique_aih": volume, "months_since_first_observed_coded_use": 1 + i % 36, "prevalent_at_window_start": False, "maintenance_status": (i % 3 != 0), "maintenance_evaluable": True, "network_in_strength": 1 + (i % 18) * 5, "age_years": 25 + i % 65, "sex_category": rng.choice(["1", "2"], n), "race_category": rng.choice(["1", "2", "MISSING"], n), "emergency_admission": rng.choice(["1", "2"], n), "diagnostic_stratum": rng.choice(["K80.3", "K80.4", "K80.5"], n), "comorbidity_burden": rng.integers(0, 4, n), "hospital_type": rng.choice(["general", "specialty"], n), "beds_sus": 20 + rng.integers(0, 30, n), "icu_beds": 2 + rng.integers(0, 5, n), "endoscopy_capability": rng.choice(["yes", "no"], n), "ivs_context": .1 + rng.integers(0, 11, n) / 20, "ans_context": .2 + rng.integers(0, 9, n) / 20, "any_icu": rng.binomial(1, .2, n), "length_of_stay_days": 1 + i % 8, "reimbursement_2025_brl": 1000 + 30 * volume, "reimbursement_brl": 900 + 20 * volume})


class Aim4V2Tests(unittest.TestCase):
    def test_four_knot_rcs_has_linear_plus_two_nonlinear_terms(self):
        basis = FIT.rcs(np.linspace(0, 10, 21), np.array([0.5, 3.5, 6.5, 9.5]))
        self.assertEqual(basis.shape, (21, 3))
        self.assertEqual(np.linalg.matrix_rank(basis), 3)

    def test_trailing12_excludes_index_and_is_calendar_complete(self):
        rows = []
        for month in pd.period_range("2021-01", "2022-02", freq="M"):
            m = month.strftime("%Y%m")
            for k in range(month.month): rows.append({"cohort": "A", "SP_CNES": "1", "competence_month": m, "NAIH": f"{m}-{k}"})
        target = pd.DataFrame({"SP_CNES": ["1"] * 14, "competence_month": [p.strftime("%Y%m") for p in pd.period_range("2021-01", "2022-02", freq="M")]})
        tr = BUILD.make_trailing_volume(pd.DataFrame(rows), target)
        feb22 = tr.loc[tr["competence_month"].eq("202202")].iloc[0]
        self.assertTrue(feb22.trailing12_complete)
        self.assertEqual(float(feb22.trailing12_a_unique_aih), float(sum(range(1, 13))))
        self.assertTrue(pd.isna(tr.loc[tr["competence_month"].eq("202112"), "trailing12_a_unique_aih"].iloc[0]))

    def test_analytic_cli_unique_aih_and_invalid_death(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); cohorts = pd.DataFrame({"cohort": ["A", "B", "B"], "adult": [True, True, True], "SP_CNES": ["1", "1", "2"], "competence_month": ["202201"] * 3, "SP_NAIH": ["a", "b", "c"], "MORTE": [0, 1, 7], "IDADE": [40, 50, 60], "DIAG_PRINC": ["K803", "K804", "K805"], "MUNIC_RES": ["1", "1", "2"], "MUNIC_MOV": ["355030", "355030", "330455"], "CAR_INT": ["06", "01", "02"], "MARCA_UTI": ["00", "01", ""], "DIAS_PERM": [1, 2, 3]})
            cohorts.to_parquet(d / "cohorts.parquet"); cohorts.iloc[1:].to_parquet(d / "linked.parquet")
            hm = pd.DataFrame({"SP_CNES": ["1", "2"], "competence_month": ["202201", "202201"], "first_observed_month": ["202101", "202101"], "left_censored_prevalent_202101": [False, True], "maintenance_6of12_evaluable": [True, False], "maintained_6of12": [True, False], "state": ["SP", "RJ"]}); hm.to_parquet(d / "hm.parquet")
            ehm = pd.DataFrame({"CNES": ["1", "2"], "competence_month": ["202201", "202201"], "beds_sus": [10.0, 20.0], "endoscopy_service": [True, True], "TP_UNID": ["05", "07"]}); ehm.to_parquet(d / "ehm.parquet")
            pd.DataFrame({"res_municipio": ["000001", "000002"], "year": ["2022", "2022"], "ivs": [.2, .3], "supp_coverage": [.4, .5]}).to_parquet(d / "context.parquet")
            result = subprocess.run([sys.executable, str(ROOT / "src" / "stage07_build_aim4_analytic_v2.py"), "--cohorts", str(d / "cohorts.parquet"), "--linked", str(d / "linked.parquet"), "--hospital-month", str(d / "hm.parquet"), "--eligible-hospital-month", str(d / "ehm.parquet"), "--context", str(d / "context.parquet"), "--output-dir", str(d / "out")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            out = pd.read_parquet(d / "out" / "aim4_analytic_v2.parquet")
            self.assertEqual(len(out), 2); self.assertEqual(int(out.death_valid.sum()), 1); self.assertTrue(pd.isna(out.loc[out.death_valid.eq(False), "in_hospital_death"].iloc[0]))
            self.assertEqual(set(out.emergency_admission), {"ELECTIVE", "URGENT_OR_NON_ELECTIVE"}); self.assertEqual(set(out.beds_sus), {10.0, 20.0})
            self.assertEqual(set(out.state_provider), {"35", "33"})
            self.assertEqual(set(out.provider_municipality), {"355030", "330455"})
            self.assertEqual(out.set_index("provider_municipality")["state_provider"].to_dict(), {"355030": "35", "330455": "33"})
            self.assertEqual(out.set_index("provider_municipality")["emergency_admission"].to_dict(), {"355030": "ELECTIVE", "330455": "URGENT_OR_NON_ELECTIVE"})
            self.assertEqual(int(out.maintenance_evaluable.fillna(False).sum()), 1)

    def test_primary_design_excludes_secondary_endpoints_and_has_fixed_effects(self):
        frame = synthetic_analytic(); X, names, audit, _ = FIT.design_matrix(frame)
        self.assertGreater(X.shape[1], 10); self.assertTrue(any(n.startswith("state[") for n in names)); self.assertTrue(any(n.startswith("calendar_month[") for n in names))
        self.assertTrue(set(["any_icu", "length_of_stay_days", "reimbursement_brl", "reimbursement_2025_brl"]).isdisjoint(names)); self.assertIn("race", audit["explicit_references"])

    def test_marginal_rd_direction_cluster_covariance_and_diagnostics_cli(self):
        frame = synthetic_analytic(); X, names, audit, _ = FIT.design_matrix(frame); y = frame.in_hospital_death.to_numpy(float)
        model = FIT.logistic_glm_clustered(X, y, frame.cnes7.to_numpy()); model["volume_knots"] = audit["volume_knots"]
        contrast, _ = FIT.standardized_volume_contrast(frame, X, names, model)
        self.assertGreater(contrast["marginal_rd_percentage_points"], 0); self.assertEqual(model["cov"].shape[0], X.shape[1]); self.assertGreater(model["n_clusters"], 1)
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); frame.loc[0, ["death_valid", "in_hospital_death"]] = [False, np.nan]; frame.to_parquet(d / "analytic.parquet")
            run = subprocess.run([sys.executable, str(ROOT / "src" / "stage07_fit_aim4_outcomes_v2.py"), "--analytic", str(d / "analytic.parquet"), "--output-dir", str(d / "models"), "--context-population-coverage", "0.99"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            model_qc = json.loads((d / "models" / "aim4_model_qc_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(model_qc["secondary_candidate_n_before_secondary_endpoint_rules"], model_qc["primary_model_n"] + 1)
            secondary_exposures = pd.read_parquet(d / "models" / "aim4_secondary_exposure_associations_v2.parquet")
            self.assertEqual(set(secondary_exposures.exposure), {"years_since_first_observed_coded_use", "maintenance_6of12", "log1p_network_in_strength", "network_hub80"})
            self.assertIn("bh_fdr_p_value_rd", secondary_exposures)
            diag = subprocess.run([sys.executable, str(ROOT / "src" / "stage07_diagnose_aim4_v2.py"), "--analytic", str(d / "analytic.parquet"), "--model-dir", str(d / "models"), "--output-dir", str(d / "diag")], capture_output=True, text=True)
            self.assertEqual(diag.returncode, 0, diag.stderr + diag.stdout)
            self.assertTrue((d / "diag" / "aim4_diagnostics_qc_v2.json").exists())

    def test_rank_deficiency_and_separation_gate(self):
        frame = synthetic_analytic(120); frame["state_provider"] = "35"; frame["calendar_month"] = "202201"; frame["comorbidity_burden"] = frame["beds_sus"]
        with self.assertRaises(FIT.ModelGateError): FIT.design_matrix(frame)
        frame = synthetic_analytic(180); frame["in_hospital_death"] = (frame["trailing12_a_unique_aih"] > 20).astype(float)
        X, _, _, _ = FIT.design_matrix(frame)
        with self.assertRaises(FIT.ModelGateError): FIT.logistic_glm_clustered(X, frame.in_hospital_death.to_numpy(), frame.cnes7.to_numpy())

    def test_source_language_and_payment_terminology(self):
        text = "\n".join((ROOT / "src" / f).read_text(encoding="utf-8").lower() for f in ["stage07_build_aim4_analytic_v2.py", "stage07_fit_aim4_outcomes_v2.py", "stage07_diagnose_aim4_v2.py"])
        for forbidden in ["causal", "predictive", "ranking", "cost"]: self.assertNotIn(forbidden, text)
        self.assertIn("reimbursement", text)


if __name__ == "__main__": unittest.main(verbosity=2)
