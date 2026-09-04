from __future__ import annotations

"""Corrected Aim 1 descriptive observed-uptake and continuity summaries."""

import argparse
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from dbfread import DBF


UF_REGION = {
    "11": "North", "12": "North", "13": "North", "14": "North", "15": "North", "16": "North", "17": "North",
    "21": "Northeast", "22": "Northeast", "23": "Northeast", "24": "Northeast", "25": "Northeast", "26": "Northeast", "27": "Northeast", "28": "Northeast", "29": "Northeast",
    "31": "Southeast", "32": "Southeast", "33": "Southeast", "35": "Southeast",
    "41": "South", "42": "South", "43": "South",
    "50": "Central-West", "51": "Central-West", "52": "Central-West", "53": "Central-West",
}

YEARS = range(2021, 2026)


def read_population_case_insensitive(population_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read official IBGE POPSBR DBFs without assuming DBF field casing.

    The DBF reader exposes fields as lower-case on the current server although
    the source documentation prints them in upper-case.  Normalising every row
    key prevents a silent zero-year/zero-denominator failure while retaining the
    required year and municipal-key checks.
    """
    records: list[dict[str, Any]] = []
    source_audit: dict[str, dict[str, Any]] = {}
    for year in YEARS:
        archive = population_dir / str(year) / f"POPSBR{str(year)[-2:]}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"missing official IBGE population archive: {archive}")
        with zipfile.ZipFile(archive) as zipped, tempfile.TemporaryDirectory(
            prefix=f"aim1_pop_{year}_"
        ) as temporary:
            members = [name for name in zipped.namelist() if name.lower().endswith(".dbf")]
            if len(members) != 1:
                raise RuntimeError(f"expected one DBF in {archive}, found {members}")
            zipped.extract(members[0], temporary)
            table = DBF(
                str(Path(temporary) / members[0]),
                encoding="latin-1",
                char_decode_errors="ignore",
                load=False,
            )
            available = {str(field).upper(): str(field) for field in table.field_names}
            population_key = "POPULACAO" if "POPULACAO" in available else "POP"
            required = {"COD_MUN", "ANO", "IDADE", population_key}
            if not required.issubset(available):
                raise RuntimeError(
                    f"unexpected IBGE fields in {archive}: {table.field_names}; "
                    f"required normalized fields={sorted(required)}"
                )
            source_audit[str(year)] = {
                "source_fields": table.field_names,
                "normalized_mapping": {
                    "COD_MUN": available["COD_MUN"],
                    "ANO": available["ANO"],
                    "IDADE": available["IDADE"],
                    "population": available[population_key],
                },
                "population_normalized_key": population_key,
                "rows_read": 0,
                "nonzero_population_rows": 0,
            }
            for raw_row in table:
                row = {str(key).upper(): value for key, value in raw_row.items()}
                row_year = int(str(row.get("ANO", "0")).strip() or 0)
                if row_year != year:
                    raise RuntimeError(
                        f"population year mismatch in {archive}: observed {row_year}"
                    )
                age = int(str(row.get("IDADE", "-1")).strip() or -1)
                population = float(row.get(population_key) or 0)
                source_audit[str(year)]["rows_read"] += 1
                source_audit[str(year)]["nonzero_population_rows"] += int(population > 0)
                records.append(
                    {
                        "year": year,
                        "municipio": str(row.get("COD_MUN", "")).strip().zfill(7)[:6],
                        "adult_population": population if age >= 18 else 0.0,
                        "total_population": population,
                    }
                )
    result = (
        pd.DataFrame(records)
        .groupby(["year", "municipio"], as_index=False)[
            ["adult_population", "total_population"]
        ]
        .sum()
    )
    municipality_counts = result.groupby("year")["municipio"].nunique().to_dict()
    expected_counts = {year: 5570 for year in YEARS}
    # Official POPSBR 2025 adds municipality 510183, so its official panel has
    # 5,571 units; keeping it prevents an undocumented denominator deletion.
    expected_counts[2025] = 5571
    if municipality_counts != expected_counts or result.duplicated(["year", "municipio"]).any():
        raise RuntimeError(
            "unexpected annual IBGE population keys: "
            f"observed={municipality_counts}; expected={expected_counts}"
        )
    annual = result.groupby("year", as_index=False)[
        ["adult_population", "total_population"]
    ].sum()
    annual_audit = {
        str(row.year): {
            "adult_population": float(row.adult_population),
            "total_population": float(row.total_population),
            "nonzero_denominators": bool(
                row.adult_population > 0 and row.total_population > 0
            ),
        }
        for row in annual.itertuples(index=False)
    }
    if not all(item["nonzero_denominators"] for item in annual_audit.values()):
        raise RuntimeError("IBGE population denominator was zero after field normalization")
    return result, {
        "source": source_audit,
        "annual_denominators": annual_audit,
        "municipality_counts": {str(year): int(count) for year, count in municipality_counts.items()},
        "total_municipality_year_keys": int(len(result)),
        "administrative_change": (
            "Official POPSBR contains 5,570 municipalities in 2021–2024 and "
            "5,571 in 2025, including new municipality 510183; it is retained."
        ),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--hospital-month", type=Path, required=True)
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    cohort = pq.read_table(
        args.cohorts,
        columns=["cohort", "competence_month", "SP_CNES", "MUNIC_MOV"],
    ).to_pandas()
    cohort["competence_month"] = cohort["competence_month"].astype(str)
    cohort["year"] = cohort["competence_month"].str[:4].astype(int)
    cohort["treat_municipio"] = cohort["MUNIC_MOV"].astype(str).str.zfill(6)
    cohort["hosp_uf"] = cohort["treat_municipio"].str[:2]
    cohort["region"] = cohort["hosp_uf"].map(UF_REGION)
    full_months = pd.DataFrame(
        {
            "cohort": [label for label in ("A", "B") for _ in range(60)],
            "competence_month": [
                f"{year}{month:02d}"
                for label in ("A", "B")
                for year in range(2021, 2026)
                for month in range(1, 13)
            ],
        }
    )
    monthly = (
        cohort.groupby(["cohort", "competence_month"], as_index=False)
        .size()
        .rename(columns={"size": "n_unique_aih"})
    )
    monthly = full_months.merge(
        monthly,
        on=["cohort", "competence_month"],
        how="left",
        validate="one_to_one",
    )
    monthly["n_unique_aih"] = monthly["n_unique_aih"].fillna(0).astype(int)
    active_units = (
        cohort.groupby(["cohort", "competence_month"], as_index=False)
        .agg(
            n_active_hospitals=("SP_CNES", "nunique"),
            n_active_provider_municipalities=("treat_municipio", "nunique"),
        )
    )
    monthly = monthly.merge(
        active_units,
        on=["cohort", "competence_month"],
        how="left",
        validate="one_to_one",
    ).fillna({"n_active_hospitals": 0, "n_active_provider_municipalities": 0})
    monthly[["n_active_hospitals", "n_active_provider_municipalities"]] = monthly[
        ["n_active_hospitals", "n_active_provider_municipalities"]
    ].astype(int)

    state_year = (
        cohort.groupby(["cohort", "year", "hosp_uf", "region"], as_index=False)
        .agg(n_unique_aih=("SP_CNES", "size"), n_active_hospitals=("SP_CNES", "nunique"))
    )
    population, population_audit = read_population_case_insensitive(args.population_dir)
    population_year = population.groupby("year", as_index=False)[
        ["adult_population", "total_population"]
    ].sum()
    annual = (
        cohort.groupby(["cohort", "year"], as_index=False)
        .size()
        .rename(columns={"size": "n_unique_aih"})
        .merge(population_year, on="year", how="left", validate="many_to_one")
    )
    annual["denominator_population"] = annual["total_population"]
    annual.loc[annual["cohort"].eq("B"), "denominator_population"] = annual.loc[
        annual["cohort"].eq("B"), "adult_population"
    ]
    annual["rate_per_100k"] = (
        annual["n_unique_aih"] / annual["denominator_population"] * 100000
    )
    annual["denominator_definition"] = annual["cohort"].map(
        {"A": "official total population", "B": "official adult population aged >=18"}
    )

    hospital_month = pd.read_parquet(args.hospital_month)
    hospital_summary = hospital_month.groupby("SP_CNES", as_index=False).agg(
        first_observed_month=("first_observed_month", "first"),
        left_censored_prevalent_202101=("left_censored_prevalent_202101", "first"),
        ever_maintained_6of12=("maintained_6of12", "max"),
        any_cessation_event=("cessation_event", "max"),
        any_recovery_event=("recovery_event", "max"),
    )
    evaluable = hospital_month[hospital_month["maintenance_6of12_evaluable"]].copy()
    first_window = evaluable.sort_values("month_index").groupby("SP_CNES", as_index=False).first()
    last_window = evaluable.sort_values("month_index").groupby("SP_CNES", as_index=False).last()
    hospital_summary = hospital_summary.merge(
        first_window[["SP_CNES", "maintained_6of12"]].rename(
            columns={"maintained_6of12": "maintained_first_completed_12m"}
        ),
        on="SP_CNES",
        how="left",
        validate="one_to_one",
    ).merge(
        last_window[["SP_CNES", "maintained_6of12", "competence_month"]].rename(
            columns={
                "maintained_6of12": "maintained_last_evaluable_window",
                "competence_month": "last_evaluable_window_month",
            }
        ),
        on="SP_CNES",
        how="left",
        validate="one_to_one",
    )
    hospital_summary["first_observed_year"] = hospital_summary[
        "first_observed_month"
    ].astype(str).str[:4]
    first_observed_by_year = (
        hospital_summary.groupby("first_observed_year").size().rename("n_hospitals").reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(args.output_dir / "aim1_monthly_national.parquet", index=False)
    state_year.to_parquet(args.output_dir / "aim1_state_year.parquet", index=False)
    annual.to_parquet(args.output_dir / "aim1_annual_rates.parquet", index=False)
    hospital_summary.to_parquet(args.output_dir / "aim1_hospital_continuity.parquet", index=False)
    first_observed_by_year.to_parquet(
        args.output_dir / "aim1_first_observed_by_year.parquet", index=False
    )
    checks = {
        "monthly_rows_120": len(monthly) == 120,
        "cohort_a_conservation": int(
            monthly.loc[monthly["cohort"].eq("A"), "n_unique_aih"].sum()
        ) == int(cohort["cohort"].eq("A").sum()),
        "cohort_b_conservation": int(
            monthly.loc[monthly["cohort"].eq("B"), "n_unique_aih"].sum()
        ) == int(cohort["cohort"].eq("B").sum()),
        "hospital_uf_complete": bool(cohort["hosp_uf"].notna().all()),
        "region_complete": bool(cohort["region"].notna().all()),
        "first_observed_counts_conserve_hospitals": int(first_observed_by_year["n_hospitals"].sum())
        == hospital_summary["SP_CNES"].nunique(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": status,
        "checks": checks,
        "n_hospitals_ever_observed": int(len(hospital_summary)),
        "left_censored_prevalent_202101": int(
            hospital_summary["left_censored_prevalent_202101"].sum()
        ),
        "first_observed_by_year": {
            str(row.first_observed_year): int(row.n_hospitals)
            for row in first_observed_by_year.itertuples(index=False)
        },
        "continuity": {
            "n_first_12m_evaluable": int(
                hospital_summary["maintained_first_completed_12m"].notna().sum()
            ),
            "maintained_first_completed_12m": int(
                hospital_summary["maintained_first_completed_12m"].fillna(False).sum()
            ),
            "maintained_last_evaluable_window": int(
                hospital_summary["maintained_last_evaluable_window"].fillna(False).sum()
            ),
            "cessation_event": int(hospital_summary["any_cessation_event"].sum()),
            "recovery_event": int(hospital_summary["any_recovery_event"].sum()),
        },
        "terminology": (
            "Use first observed coded use/observed uptake, not true adoption. "
            "January 2021 performers are prevalent-at-window-start."
        ),
        "geography": "Treating state derives from MUNIC_MOV, never from the CNES identifier prefix.",
        "rates": {
            "A": "all-indication unique AIHs per 100,000 total population",
            "B": "strict adult choledocholithiasis unique AIHs per 100,000 adults",
        },
        "population_read_qc": population_audit,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
