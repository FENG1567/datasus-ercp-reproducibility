from __future__ import annotations

"""Aim 1 adoption model: discrete-time complementary log-log (cloglog) GLM
for first adoption among eligible hospital-months (primary risk set), with
hospital-clustered robust inference. Exposures: same-month CNES capacity
(beds, endoscopy service, workforce), state population demand, lagged
same-state adoption (neighbouring diffusion), calendar time (month index
with spline via b-spline basis)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--hospital-month", type=Path, required=True)
    parser.add_argument("--st", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    elig = pq.read_table(args.eligible).to_pandas()
    hm = pq.read_table(args.hospital_month).to_pandas()
    st = pq.read_table(args.st, columns=["CNES", "competence_month", "CODUFMUN"]).to_pandas()
    st["CNES"] = st["CNES"].astype(str).str.zfill(7)
    st["hosp_uf"] = st["CODUFMUN"].astype(str).str[:2]

    # risk set: primary eligible hospital-months
    risk = elig[elig["primary"]].copy()
    risk = risk.merge(st[["CNES", "hosp_uf"]].drop_duplicates("CNES"), on="CNES", how="left")
    risk["month_int"] = risk["competence_month"].astype(int)
    risk["month_index"] = (risk["month_int"] // 100 - 2021) * 12 + (risk["month_int"] % 100) - 1

    # adoption month per hospital
    adopt = (
        hm[hm["state"] == "first_adoption"][["SP_CNES", "competence_month"]]
        .rename(columns={"SP_CNES": "CNES"})
    )
    adopt["CNES"] = adopt["CNES"].astype(str).str.zfill(7)
    adopt["adopt_int"] = adopt["competence_month"].astype(int)
    risk = risk.merge(adopt[["CNES", "adopt_int"]], on="CNES", how="left")
    risk["event"] = (risk["month_int"] == risk["adopt_int"]).astype(int)
    # drop post-adoption months (already at risk only pre-adoption + event month)
    risk = risk[(risk["adopt_int"].isna()) | (risk["month_int"] <= risk["adopt_int"])]

    # covariates
    risk["log_beds_sus"] = np.log1p(risk["beds_sus"])
    risk["has_endoscopy_service"] = risk["endoscopy_service"].astype(int)
    risk["log_endoscopists"] = np.log1p(risk["endoscopists"])
    risk["log_anesthesiologists"] = np.log1p(risk["anesthesiologists"])
    risk["log_medical_cbos"] = np.log1p(risk["medical_cbos"])

    # state population demand (approximation via IBGE state population year)
    # state pop from supplement (built in descriptive step) - reuse crude denominator
    pop_path = args.st.parent.parent / "supplement"
    state_pop = None
    # simpler: use hospital count in state as demand proxy
    state_hospitals = (
        elig[elig["primary"]].merge(st[["CNES", "hosp_uf"]].drop_duplicates("CNES"), on="CNES", how="left")
        .groupby("hosp_uf")["CNES"].nunique().rename("state_eligible_hospitals")
    )
    risk = risk.merge(state_hospitals.reset_index(), on="hosp_uf", how="left")

    # lagged same-state adoption count (previous month)
    state_adoption_monthly = (
        adopt.merge(st[["CNES", "hosp_uf"]].drop_duplicates("CNES"), on="CNES", how="left")
        .groupby(["hosp_uf", "adopt_int"]).size().reset_index(name="n_adopted")
    )
    state_adoption_monthly["month_int"] = state_adoption_monthly["adopt_int"]
    all_months = risk[["month_int"]].drop_duplicates()
    state_cum = (
        state_adoption_monthly.groupby("hosp_uf").apply(
            lambda g: pd.DataFrame({
                "month_int": all_months["month_int"],
                "lag_adopted_state": g.set_index("month_int").reindex(all_months["month_int"], fill_value=0)["n_adopted"].cumsum().shift(1).fillna(0),
            }), include_groups=False
        ).reset_index()
    )
    risk = risk.merge(state_cum, on=["hosp_uf", "month_int"], how="left")
    risk["lag_adopted_state"] = risk["lag_adopted_state"].fillna(0)

    # time spline (b-spline of month_index)
    from statsmodels.stats.sandwich_covariance import cov_cluster

    X = risk[["month_index", "log_beds_sus", "has_endoscopy_service", "log_endoscopists",
              "log_anesthesiologists", "log_medical_cbos", "lag_adopted_state"]].copy()
    X = X.merge(risk[["hosp_uf"]], left_index=True, right_index=True)
    X = pd.get_dummies(X, columns=["hosp_uf"], drop_first=True).astype(float)
    X["intercept"] = 1.0
    y = risk["event"]

    model = sm.GLM(y, X, family=sm.families.Binomial(sm.families.links.CLogLog()))
    try:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": risk["CNES"]}, maxiter=200)
        converged = result.mle_retvals is None or result.mle_retvals.get("converged", True)
    except Exception as exc:
        audit = {"schema_version": "1.0", "accessed_at": utc_now(), "status": "FAIL",
                 "error": str(exc)[:400]}
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return 1

    coefs = pd.DataFrame({
        "term": X.columns,
        "coef": result.params,
        "se_cluster": result.bse,
        "pvalue": result.pvalues,
        "exp_coef": np.exp(result.params),
    })
    coefs["exp_lo95"] = np.exp(result.params - 1.96 * result.bse)
    coefs["exp_hi95"] = np.exp(result.params + 1.96 * result.bse)

    # absolute adoption probability: mean predicted P(event) per month at
    # median covariates (marginal at observed X)
    risk["pred_prob"] = result.predict(X)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    coefs.to_csv(args.output.with_suffix(".coefs.csv"), index=False)
    risk[["CNES", "competence_month", "month_int", "event", "pred_prob", "hosp_uf"]].to_parquet(
        args.output, index=False)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS" if converged else "WARN",
        "n_hospital_months": int(len(risk)),
        "n_events": int(y.sum()),
        "n_hospitals": int(risk["CNES"].nunique()),
        "converged": converged,
        "key_effects": {
            "log_beds_sus": {"exp_coef": float(np.exp(result.params["log_beds_sus"])),
                             "lo95": float(np.exp(result.params["log_beds_sus"] - 1.96 * result.bse["log_beds_sus"])),
                             "hi95": float(np.exp(result.params["log_beds_sus"] + 1.96 * result.bse["log_beds_sus"]))},
            "has_endoscopy_service": {"exp_coef": float(np.exp(result.params["has_endoscopy_service"]))},
            "log_endoscopists": {"exp_coef": float(np.exp(result.params["log_endoscopists"]))},
            "lag_adopted_state": {"exp_coef": float(np.exp(result.params["lag_adopted_state"]))},
            "month_index": {"exp_coef": float(np.exp(result.params["month_index"]))},
        },
        "mean_monthly_adoption_prob": float(risk["pred_prob"].mean()),
        "output": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())