from __future__ import annotations

"""Corrected observed-uptake, maintenance, cessation, and recovery state machine."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


MONTHS = [f"{year}{month:02d}" for year in range(2021, 2026) for month in range(1, 13)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    cohort = pq.read_table(
        args.cohorts,
        columns=["cohort", "SP_CNES", "competence_month"],
    ).to_pandas()
    cohort = cohort[cohort["cohort"].astype(str).eq("A")].copy()
    cohort["SP_CNES"] = cohort["SP_CNES"].astype(str).str.zfill(7)
    cohort["competence_month"] = cohort["competence_month"].astype(str)
    counts = (
        cohort.groupby(["SP_CNES", "competence_month"], as_index=False)
        .size()
        .rename(columns={"size": "ercp_count"})
    )
    hospitals = counts[["SP_CNES"]].drop_duplicates()
    month_table = pd.DataFrame(
        {"competence_month": MONTHS, "month_index": range(len(MONTHS))}
    )
    result = hospitals.merge(month_table, how="cross").merge(
        counts,
        on=["SP_CNES", "competence_month"],
        how="left",
        validate="one_to_one",
    )
    result["ercp_count"] = result["ercp_count"].fillna(0).astype(int)
    result["active"] = result["ercp_count"].ge(1)

    eligible = pq.read_table(
        args.eligible,
        columns=["CNES", "competence_month", "broad", "primary", "strict"],
    ).to_pandas()
    eligible = eligible.rename(
        columns={
            "CNES": "SP_CNES",
            "broad": "eligible_broad",
            "primary": "eligible_primary",
            "strict": "eligible_strict",
        }
    )
    eligible["SP_CNES"] = eligible["SP_CNES"].astype(str).str.zfill(7)
    eligible["competence_month"] = eligible["competence_month"].astype(str)
    result = result.merge(
        eligible,
        on=["SP_CNES", "competence_month"],
        how="left",
        validate="one_to_one",
    )
    eligibility_columns = ["eligible_broad", "eligible_primary", "eligible_strict"]
    result[eligibility_columns] = result[eligibility_columns].fillna(False).astype(bool)
    result = result.sort_values(["SP_CNES", "month_index"]).reset_index(drop=True)

    first_observed = (
        result[result["active"]].groupby("SP_CNES")["competence_month"].first()
    )
    first_ge3 = (
        result[result["ercp_count"].ge(3)]
        .groupby("SP_CNES")["competence_month"]
        .first()
    )
    first_ge5 = (
        result[result["ercp_count"].ge(5)]
        .groupby("SP_CNES")["competence_month"]
        .first()
    )
    result["first_observed_month"] = result["SP_CNES"].map(first_observed)
    result["first_observed_ge3_month"] = result["SP_CNES"].map(first_ge3)
    result["first_observed_ge5_month"] = result["SP_CNES"].map(first_ge5)
    result["left_censored_prevalent_202101"] = result["first_observed_month"].eq("202101")

    result["state"] = "not_yet_observed"
    after_first = result["competence_month"].ge(result["first_observed_month"])
    result.loc[
        after_first & ~result["active"], "state"
    ] = "inactive_after_first_observed"
    result.loc[
        after_first & result["active"], "state"
    ] = "active_after_first_observed"
    result.loc[
        result["competence_month"].eq(result["first_observed_month"]), "state"
    ] = "first_observed_coded_use"

    result["rolling12_active_months"] = pd.NA
    result["maintenance_6of12_evaluable"] = False
    result["maintained_6of12"] = False
    result["eligible_12of12_primary"] = False
    result["first_3_consecutive_active_months"] = False
    result["consecutive_eligible_inactive"] = 0
    result["cessation_event"] = False
    result["post_cessation"] = False
    result["recovery_event"] = False

    for hospital, group in result.groupby("SP_CNES", sort=False):
        indices = group.index.to_list()
        active = group["active"].to_numpy(dtype=bool)
        eligible_primary = group["eligible_primary"].to_numpy(dtype=bool)
        first_positions = [position for position, value in enumerate(active) if value]
        if not first_positions:
            continue
        first_position = first_positions[0]

        for position in range(first_position, len(indices)):
            if position >= first_position + 2 and bool(active[position - 2 : position + 1].all()):
                result.at[indices[position - 2], "first_3_consecutive_active_months"] = True
                break

        for position in range(first_position + 11, len(indices)):
            window_active = active[position - 11 : position + 1]
            window_eligible = eligible_primary[position - 11 : position + 1]
            index = indices[position]
            result.at[index, "rolling12_active_months"] = int(window_active.sum())
            result.at[index, "maintenance_6of12_evaluable"] = True
            result.at[index, "maintained_6of12"] = bool(window_active.sum() >= 6)
            result.at[index, "eligible_12of12_primary"] = bool(window_eligible.all())

        inactive_run = 0
        cessation_seen = False
        for position in range(first_position + 1, len(indices)):
            index = indices[position]
            if active[position]:
                if cessation_seen:
                    result.at[index, "recovery_event"] = True
                    cessation_seen = False
                inactive_run = 0
            elif eligible_primary[position]:
                inactive_run += 1
                if inactive_run == 6:
                    result.at[index, "cessation_event"] = True
                    cessation_seen = True
            else:
                inactive_run = 0
            result.at[index, "consecutive_eligible_inactive"] = inactive_run
            result.at[index, "post_cessation"] = cessation_seen

    # Cessation/recovery must never occur before first observed coded use.
    pre_first = result["state"].eq("not_yet_observed")
    impossible_pre_first_events = int(
        result.loc[pre_first, ["cessation_event", "recovery_event"]].any(axis=1).sum()
    )
    result["month_int"] = result["competence_month"].astype(int)
    key_duplicates = int(result.duplicated(["SP_CNES", "competence_month"]).sum())
    count_conservation = int(result["ercp_count"].sum()) == len(cohort)
    status = "PASS" if key_duplicates == 0 and count_conservation and impossible_pre_first_events == 0 else "FAIL"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "terminology": (
            "First observed coded use is not the true technology-adoption date; "
            "January 2021 performers are left-censored prevalent users."
        ),
        "n_hospitals_ever_observed": int(len(hospitals)),
        "n_hospital_months": int(len(result)),
        "cohort_a_unique_aih": int(len(cohort)),
        "hospital_month_count_sum": int(result["ercp_count"].sum()),
        "key_duplicates": key_duplicates,
        "first_observed_by_year": {
            str(year): int(value)
            for year, value in first_observed.str[:4].value_counts().sort_index().items()
        },
        "left_censored_prevalent_202101_hospitals": int(first_observed.eq("202101").sum()),
        "hospitals_with_evaluable_12_month_window": int(
            result.groupby("SP_CNES")["maintenance_6of12_evaluable"].any().sum()
        ),
        "hospitals_ever_maintained_6of12": int(
            result.groupby("SP_CNES")["maintained_6of12"].any().sum()
        ),
        "hospitals_with_cessation_event": int(
            result.groupby("SP_CNES")["cessation_event"].any().sum()
        ),
        "hospitals_with_recovery_event": int(
            result.groupby("SP_CNES")["recovery_event"].any().sum()
        ),
        "observed_active_months_not_primary_eligible": int(
            (result["active"] & ~result["eligible_primary"]).sum()
        ),
        "impossible_pre_first_events": impossible_pre_first_events,
        "definitions": {
            "maintenance": (
                "At least 6 active months in a completed trailing 12-month window "
                "whose first month is no earlier than first observed coded use."
            ),
            "cessation": (
                "First month reaching 6 consecutive inactive months after first observed "
                "coded use while primary eligibility is present in each counted month; "
                "ineligibility breaks the run."
            ),
            "recovery": "First active month after a cessation event.",
        },
        "output": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
