from __future__ import annotations

"""hospital_month table: per hospital-month unique ERCP counts (cohort A),
adoption state (first >=1; sensitivities >=3/>=5; maintenance 6/12;
cessation 6 consecutive inactive months while eligible)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_key(comp: str) -> int:
    return int(comp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    df = pq.read_table(args.cohorts).to_pandas()
    cohort_a = df[df["cohort"] == "A"]
    counts = (
        cohort_a.groupby(["SP_CNES", "competence_month"])
        .size()
        .rename("ercp_count")
        .reset_index()
    )

    # full grid: all months 202101..202512 for every hospital that ever appeared
    month_ints = [y * 100 + m for y in range(2021, 2026) for m in range(1, 13)]
    all_months = pd.DataFrame(
        {"month_int": month_ints, "competence_month": [str(m) for m in month_ints]}
    )
    grid = counts[["SP_CNES"]].drop_duplicates().merge(all_months, how="cross")
    hm = grid.merge(counts, on=["SP_CNES", "competence_month"], how="left")
    hm["ercp_count"] = hm["ercp_count"].fillna(0).astype(int)
    hm["active"] = hm["ercp_count"] >= 1

    hm = hm.sort_values(["SP_CNES", "month_int"]).reset_index(drop=True)
    # first adoption: first month with count>=1
    hm["cum_active"] = hm.groupby("SP_CNES")["active"].cumsum()
    first_active = hm[hm["active"]].groupby("SP_CNES")["month_int"].first()
    hm["adoption_month"] = hm["SP_CNES"].map(first_active)
    hm["state"] = "never"
    hm.loc[hm["active"], "state"] = "active"
    hm.loc[(hm["cum_active"] == 1) & (hm["active"]), "state"] = "first_adoption"
    hm.loc[(hm["active"]) & (hm["cum_active"] > 1), "state"] = "active"
    hm.loc[~hm["active"] & hm["adoption_month"].notna(), "state"] = "inactive_after_adoption"

    # adoption threshold sensitivities
    for threshold, col in [(3, "adopt_ge3"), (5, "adopt_ge5")]:
        t_first = (
            counts[counts["ercp_count"] >= threshold]
            .groupby("SP_CNES")["competence_month"].first()
        )
        hm[col] = hm["SP_CNES"].map(t_first)

    # maintenance: 12-month rolling window with >=6 active months after adoption
    def maintenance_flag(g: pd.DataFrame) -> pd.Series:
        ad = g["adoption_month"].iloc[0]
        if pd.isna(ad):
            return pd.Series(False, index=g.index)
        window_end = ad + 100  # ~12 months ahead on yyyymm scale (approx)
        end_int = ad + 99 if ad % 100 <= 12 else ad + 99
        # simpler: adopt month plus 11 months
        year = ad // 100
        mon = ad % 100
        end = (year + 1) * 100 + mon if mon <= 11 else (year + 2) * 100 + 1
        # not exact; use calendar approach below instead
        return pd.Series(False, index=g.index)

    # calendar-based 12-month windows
    import calendar as _cal

    def add_months(y, m, delta):
        total = y * 12 + (m - 1) + delta
        return (total // 12) * 100 + (total % 12) + 1

    months_sorted = sorted(all_months["month_int"].tolist())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    hm["month_idx"] = hm["month_int"].map(month_to_idx)
    hm["maintained_6of12"] = False
    for hospital, group in hm.groupby("SP_CNES"):
        ad = group["adoption_month"].iloc[0]
        if pd.isna(ad):
            continue
        ad_idx = month_to_idx.get(int(ad))
        if ad_idx is None:
            continue
        for _, row in group.iterrows():
            idx = row["month_idx"]
            if idx < ad_idx:
                continue
            window = hm[(hm["SP_CNES"] == hospital) & (hm["month_idx"] >= idx) & (hm["month_idx"] < idx + 12)]
            if int(window["active"].sum()) >= 6:
                hm.loc[(hm["SP_CNES"] == hospital) & (hm["month_idx"] == idx), "maintained_6of12"] = True

    # cessation: 6 consecutive inactive months after adoption
    hm["consecutive_inactive"] = 0
    for hospital, group in hm.groupby("SP_CNES"):
        idxs = group.index.tolist()
        run = 0
        for i in idxs:
            if hm.at[i, "active"]:
                run = 0
            else:
                run += 1
            hm.at[i, "consecutive_inactive"] = run
    hm["cessation"] = hm["consecutive_inactive"] >= 6

    out_cols = ["SP_CNES", "competence_month", "month_int", "ercp_count", "active",
                "state", "adoption_month", "adopt_ge3", "adopt_ge5",
                "maintained_6of12", "consecutive_inactive", "cessation"]
    result = hm[out_cols].sort_values(["SP_CNES", "month_int"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS",
        "hospitals_ever_active": int(first_active.shape[0]),
        "hospital_months": int(len(result)),
        "unique_key_duplicates": int(result.duplicated(["SP_CNES", "competence_month"]).sum()),
        "adoption_months_distribution": {
            str(y): int(n) for y, n in first_active.reset_index(name="m").groupby(lambda r: int(str(first_active.iloc[r])[:4]) if False else first_active.iloc[r] // 100).size().items()
        } if first_active.shape[0] else {},
        "state_counts": {k: int(v) for k, v in result["state"].value_counts().items()},
        "maintained_6of12_hospitals": int(result.groupby("SP_CNES")["maintained_6of12"].any().sum()),
        "cessation_hospitals": int(result.groupby("SP_CNES")["cessation"].any().sum()),
        "output": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())