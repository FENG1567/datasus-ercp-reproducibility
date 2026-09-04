#!/usr/bin/env python3
"""Build the versioned, associational Aim 4 analytic data set.

The module deliberately makes no claim about an adoption date: ``first_observed``
means first observed coded use within the left-truncated study window.  Inputs are
provided on the command line so that raw and frozen evidence are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def first_column(frame: pd.DataFrame, names: Iterable[str], required: bool = True) -> str | None:
    lookup = {str(c).upper(): c for c in frame.columns}
    for name in names:
        if name.upper() in lookup:
            return lookup[name.upper()]
    if required:
        raise ValueError(f"Required field absent; accepted aliases: {list(names)}")
    return None


def month_index(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.extract(r"(\d{4})(\d{2})")
    year, month = pd.to_numeric(text[0], errors="coerce"), pd.to_numeric(text[1], errors="coerce")
    if ((month < 1) | (month > 12)).fillna(True).any():
        raise ValueError("Invalid competence month")
    return year * 12 + month


def unique_aih_key(frame: pd.DataFrame) -> pd.Series:
    supplied = first_column(frame, ["aih_key", "unique_aih_key", "AIH_KEY"], required=False)
    if supplied:
        return frame[supplied].astype(str)
    nai = first_column(frame, ["SP_NAIH", "NAIH", "N_AIH", "AIH", "NUM_AIH"])
    cnes = first_column(frame, ["SP_CNES", "CNES", "cnes7"])
    month = first_column(frame, ["competence_month", "COMPETENCE_MONTH", "ANO_CMPT"])
    return frame[cnes].astype(str).str.strip() + "|" + frame[month].astype(str) + "|" + frame[nai].astype(str).str.strip()


def make_trailing_volume(cohorts: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Return prior *complete* 12 calendar month volume; index month never contributes."""
    cohort_col = first_column(cohorts, ["cohort"])
    hosp_col = first_column(cohorts, ["SP_CNES", "CNES", "cnes7"])
    month_col = first_column(cohorts, ["competence_month", "COMPETENCE_MONTH"])
    a = cohorts.loc[cohorts[cohort_col].astype(str).str.upper().eq("A")].copy()
    a["_hospital"] = a[hosp_col].astype(str).str.strip().str.zfill(7)
    a["_month_index"] = month_index(a[month_col])
    a["_aih_key"] = unique_aih_key(a)
    monthly = a.groupby(["_hospital", "_month_index"], as_index=False)["_aih_key"].nunique().rename(columns={"_aih_key": "a_unique_aih"})
    t_hosp = first_column(target, ["SP_CNES", "CNES", "cnes7"])
    t_month = first_column(target, ["competence_month", "COMPETENCE_MONTH"])
    lookup = target[[t_hosp, t_month]].drop_duplicates().copy()
    lookup["cnes7"] = lookup[t_hosp].astype(str).str.strip().str.zfill(7)
    lookup["month_index"] = month_index(lookup[t_month])
    if lookup.empty:
        return lookup.assign(trailing12_a_unique_aih=pd.Series(dtype=float), trailing12_complete=pd.Series(dtype=bool))
    observed_start = int(lookup["month_index"].min())
    hospitals = sorted(set(lookup["cnes7"]) | set(monthly["_hospital"]))
    all_months = np.arange(observed_start - 12, int(lookup["month_index"].max()) + 1)
    grid = pd.MultiIndex.from_product([hospitals, all_months], names=["cnes7", "month_index"]).to_frame(index=False)
    grid = grid.merge(monthly.rename(columns={"_hospital": "cnes7", "_month_index": "month_index"}), on=["cnes7", "month_index"], how="left")
    grid["a_unique_aih"] = grid["a_unique_aih"].fillna(0.0)
    grid = grid.sort_values(["cnes7", "month_index"])
    # At m, sum m-12,...,m-1.  Calendar months before study start are unknown,
    # therefore observations before twelve complete study months are flagged, not zero-filled.
    grid["trailing12_a_unique_aih"] = grid.groupby("cnes7")["a_unique_aih"].transform(lambda s: s.shift(1).rolling(12, min_periods=12).sum())
    grid["trailing12_complete"] = grid["month_index"] - 12 >= observed_start
    out = lookup.merge(grid[["cnes7", "month_index", "trailing12_a_unique_aih", "trailing12_complete"]], on=["cnes7", "month_index"], how="left")
    out.loc[~out["trailing12_complete"], "trailing12_a_unique_aih"] = np.nan
    return out[[t_hosp, t_month, "trailing12_a_unique_aih", "trailing12_complete"]]


def diagnosis_stratum(frame: pd.DataFrame) -> pd.Series:
    col = first_column(frame, ["principal", "DIAG_PRINC", "DIAG_PRINCIPAL", "DIAGPRI"])
    code = frame[col].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    return np.select([code.str.startswith("K803"), code.str.startswith("K804"), code.str.startswith("K805")], ["K80.3", "K80.4", "K80.5"], default="OTHER")


def secondary_burden(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    cols = [c for c in frame.columns if str(c).upper().startswith(("DIAGSEC", "DIAG_SEC", "DIAGSECUND"))]
    if not cols:
        return pd.Series(np.nan, index=frame.index), "unavailable; not adjusted"
    clean = frame[cols].fillna("").astype(str).apply(
        lambda x: x.str.upper().str.strip().replace(
            {"NAN": "", "NONE": "", "0": "", "00": "", "000": "", "0000": ""}
        )
    )
    return clean.apply(lambda row: len({x for x in row if x}), axis=1).astype(float), "distinct nonempty secondary-diagnosis fields"


VALID_UF_CODES = {
    "11", "12", "13", "14", "15", "16", "17", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "31", "32", "33", "35", "41", "42", "43", "50",
    "51", "52", "53",
}


def provider_geography(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    municipality = values.astype(str).str.strip().str.zfill(6)
    state = municipality.str[:2]
    if (~state.isin(VALID_UF_CODES)).any():
        invalid = sorted(state.loc[~state.isin(VALID_UF_CODES)].unique().tolist())
        raise ValueError(f"invalid provider municipality/state codes: {invalid}")
    return municipality, state


def admission_urgency(values: pd.Series) -> pd.Series:
    code = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(2)
    return pd.Series(
        np.select(
            [code.eq("01"), code.isin({"02", "03", "04", "05", "06"})],
            ["ELECTIVE", "URGENT_OR_NON_ELECTIVE"],
            default="MISSING_OR_INVALID",
        ),
        index=values.index,
    )


def standardize_hospital_month(hospital_month: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    hcol = first_column(hospital_month, ["SP_CNES", "CNES", "cnes7"])
    mcol = first_column(hospital_month, ["competence_month", "COMPETENCE_MONTH"])
    out = hospital_month.copy()
    out["cnes7"] = out[hcol].astype(str).str.strip().str.zfill(7)
    out["competence_month"] = out[mcol].astype(str)
    first = first_column(
        out,
        ["first_observed_coded_use_month", "first_observed_use_month", "first_observed_month", "adoption_month"],
        required=False,
    )
    prevalent = first_column(
        out,
        [
            "prevalent_at_window_start",
            "left_truncated_prevalent",
            "left_censored_prevalent_202101",
            "prevalent_202101",
        ],
        required=False,
    )
    maintained = first_column(out, ["maintained_6of12", "maintenance_status"], required=False)
    maintenance_evaluable = first_column(
        out, ["maintenance_6of12_evaluable", "maintenance_evaluable"], required=False
    )
    keep = ["cnes7", "competence_month"]
    out["first_observed_coded_use_month"] = out[first].astype(str) if first else pd.NA
    out["prevalent_at_window_start"] = out[prevalent].astype("boolean") if prevalent else False
    out["maintenance_status"] = out[maintained].astype("boolean") if maintained else pd.NA
    out["maintenance_evaluable"] = (
        out[maintenance_evaluable].astype("boolean") if maintenance_evaluable else pd.NA
    )
    if maintained and maintenance_evaluable:
        out.loc[~out["maintenance_evaluable"].fillna(False), "maintenance_status"] = pd.NA
    keep += [
        "first_observed_coded_use_month",
        "prevalent_at_window_start",
        "maintenance_status",
        "maintenance_evaluable",
    ]
    # Preserve explicit monthly capacity fields; inference never derives state from a CNES prefix.
    for canonical, aliases in {
        "hospital_type": ["TP_UNID", "hospital_type"], "beds_sus": ["beds_sus", "QTLEITOSSUS"],
        "icu_beds": ["icu_beds", "beds_icu", "UTI_BEDS"], "endoscopy_capability": ["endoscopy_service", "endoscopy_capability"],
        "state_provider": ["state_provider", "UF", "uf", "CNES_UF"],
    }.items():
        col = first_column(out, aliases, required=False)
        if col:
            out[canonical] = out[col]
            keep.append(canonical)
    return out[keep].drop_duplicates(["cnes7", "competence_month"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", required=True, type=Path)
    parser.add_argument("--linked", required=True, type=Path)
    parser.add_argument("--hospital-month", required=True, type=Path)
    parser.add_argument("--eligible-hospital-month", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--ipca", type=Path)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [args.cohorts, args.linked, args.hospital_month, args.eligible_hospital_month, args.context] + [p for p in [args.ipca, args.network] if p]
    cohorts, linked = pd.read_parquet(args.cohorts), pd.read_parquet(args.linked)
    bcol = first_column(cohorts, ["cohort"])
    adult_col = first_column(cohorts, ["adult", "is_adult"])
    b = cohorts.loc[cohorts[bcol].astype(str).str.upper().eq("B") & cohorts[adult_col].astype(bool)].copy()
    b["unique_aih_key"] = unique_aih_key(b)
    if b["unique_aih_key"].duplicated().any():
        raise ValueError("Cohort B must contain one row per unique AIH")
    linked_keys = set(unique_aih_key(linked))
    b["matched_rd_record"] = b["unique_aih_key"].isin(linked_keys)
    # Exact linkage remains mandatory, but retain all B records in audit and only select matched rows for analysis.
    b = b.loc[b["matched_rd_record"]].copy()
    death_col = first_column(b, ["MORTE", "death", "MORTE_VALID"])
    death_raw = pd.to_numeric(b[death_col], errors="coerce")
    b["death_valid"] = death_raw.isin([0, 1])
    b["in_hospital_death"] = death_raw.where(b["death_valid"]).astype("Float64")
    b["cnes7"] = b[first_column(b, ["SP_CNES", "CNES", "cnes7"])].astype(str).str.strip().str.zfill(7)
    provider_municipality_col = first_column(
        b, ["MUNIC_MOV", "provider_municipality", "performing_municipality"]
    )
    b["provider_municipality"], b["state_provider"] = provider_geography(
        b[provider_municipality_col]
    )
    b["competence_month"] = b[first_column(b, ["competence_month", "COMPETENCE_MONTH"])].astype(str)
    b["calendar_month"] = b["competence_month"]
    b["year"] = b["competence_month"].str.slice(0, 4)
    b["age_years"] = pd.to_numeric(b[first_column(b, ["age_years", "IDADE", "age"])], errors="coerce")
    b["diagnostic_stratum"] = diagnosis_stratum(b)
    b["comorbidity_burden"], comorbidity_note = secondary_burden(b)
    race = first_column(b, ["RACA_COR", "race", "race_color"], required=False)
    b["race_category"] = b[race].fillna("MISSING").astype(str).str.strip().replace({"": "MISSING", "99": "MISSING"}) if race else "MISSING"
    sex = first_column(b, ["SEXO", "sex"], required=False)
    b["sex_category"] = b[sex].fillna("MISSING").astype(str).str.strip().replace({"": "MISSING", "9": "MISSING"}) if sex else "MISSING"
    # CAR_INT is the SIH admission-character field (elective/emergency).  The
    # NATUREZA field is a different administrative concept and was constant in
    # the prior data audit, so it is only a last-resort legacy alias.
    emergency = first_column(
        b, ["CAR_INT", "admission_type", "emergency_admission", "NATUREZA"], required=False
    )
    b["emergency_admission"] = (
        admission_urgency(b[emergency]) if emergency else "MISSING_OR_INVALID"
    )
    vol = make_trailing_volume(cohorts, b)
    b = b.merge(vol, left_on=[first_column(b, ["SP_CNES", "CNES", "cnes7"]), "competence_month"], right_on=[first_column(vol, ["SP_CNES", "CNES", "cnes7"]), first_column(vol, ["competence_month", "COMPETENCE_MONTH"])], how="left").drop(columns=[c for c in vol.columns[:2] if c in b.columns and c not in {"competence_month"}], errors="ignore")
    hm = standardize_hospital_month(pd.read_parquet(args.hospital_month), b)
    # The national CNES eligibility panel is large.  Read only fields used by
    # this frozen analysis and filter to cohort-B provider hospitals before the
    # semantic standardisation step.
    eligible_candidates = [
        "CNES",
        "SP_CNES",
        "cnes7",
        "competence_month",
        "COMPETENCE_MONTH",
        "beds_sus",
        "QTLEITOSSUS",
        "endoscopy_service",
        "endoscopy_capability",
        "TP_UNID",
        "hospital_type",
        "icu_beds",
        "beds_icu",
        "UTI_BEDS",
        "state_provider",
        "state",
        "UF",
        "CNES_UF",
    ]
    available_eligible = set(pq.ParquetFile(args.eligible_hospital_month).schema.names)
    eligible_columns = [column for column in eligible_candidates if column in available_eligible]
    eligible_raw = pd.read_parquet(args.eligible_hospital_month, columns=eligible_columns)
    target_hospitals = set(b["cnes7"])
    eligible_hospital_column = first_column(eligible_raw, ["CNES", "SP_CNES", "cnes7"])
    eligible_raw = eligible_raw[
        eligible_raw[eligible_hospital_column]
        .astype(str)
        .str.strip()
        .str.zfill(7)
        .isin(target_hospitals)
    ].copy()
    eligible = standardize_hospital_month(eligible_raw, b)
    # Merge by the scientific key, never by pandas row number.  The Stage-3
    # hospital-month table owns observed-use/maintenance fields, whereas the
    # CNES eligibility panel supplies monthly capacity fields.
    capacity = (
        hm.set_index(["cnes7", "competence_month"])
        .combine_first(eligible.set_index(["cnes7", "competence_month"]))
        .reset_index()
    )
    b = b.merge(capacity, on=["cnes7", "competence_month"], how="left", validate="many_to_one")
    b["state_provider"] = b["provider_municipality"].str[:2]
    if b.groupby("cnes7")["state_provider"].nunique().gt(1).any():
        raise ValueError("one CNES maps to more than one provider state within the analytic window")
    first_idx = month_index(b["first_observed_coded_use_month"].fillna("999912"))
    b["months_since_first_observed_coded_use"] = month_index(b["competence_month"]) - first_idx
    b.loc[b["first_observed_coded_use_month"].isna(), "months_since_first_observed_coded_use"] = np.nan
    context = pd.read_parquet(args.context)
    res = first_column(b, ["MUNIC_RES", "residence_municipality", "res_mun", "MUNIC_RESIDENCIA"])
    cmun = first_column(context, ["residence_municipality", "res_municipio", "MUNIC_RES", "municipality"])
    cyear = first_column(context, ["year", "ANO"])
    context = context.copy(); context["residence_municipality"] = context[cmun].astype(str).str.strip().str.zfill(6); context["year"] = context[cyear].astype(str)
    b["residence_municipality"] = b[res].astype(str).str.strip().str.zfill(6)
    wanted = ["residence_municipality", "year"] + [c for c in [first_column(context, ["ivs", "IVS"], False), first_column(context, ["ans", "supp_coverage", "ANS"], False)] if c]
    ivs_col = first_column(context, ["ivs", "IVS"], False)
    ans_col = first_column(context, ["ans", "supp_coverage", "ANS"], False)
    rename_context = {}
    if ivs_col:
        rename_context[ivs_col] = "ivs_context"
    if ans_col:
        rename_context[ans_col] = "ans_context"
    context = context[wanted].rename(columns=rename_context)
    if "ivs_context" not in context or "ans_context" not in context:
        raise ValueError("Aim 4 context requires both municipal IVS and ANS coverage")
    b = b.merge(context, on=["residence_municipality", "year"], how="left", validate="many_to_one")
    if args.network:
        net = pd.read_parquet(args.network)
        if "layer" in net:
            net = net[net["layer"].astype(str).eq("cnes")].copy()
        if "cohort" in net:
            net = net[net["cohort"].astype(str).eq("B")].copy()
        nh = first_column(net, ["SP_CNES", "CNES", "cnes7", "node", "target_node"])
        ny = first_column(net, ["year", "ANO"])
        nv = first_column(net, ["in_strength", "weighted_in_strength", "n_aih"])
        net = net[net[ny].astype(str).ne("pooled")].copy()
        net = net.assign(
            cnes7=net[nh].astype(str).str.replace(r"^T:", "", regex=True).str.strip().str.zfill(7),
            year=net[ny].astype(str),
            network_in_strength=pd.to_numeric(net[nv], errors="coerce"),
        )
        b = b.merge(net[["cnes7", "year", "network_in_strength"]].drop_duplicates(["cnes7", "year"]), on=["cnes7", "year"], how="left", validate="many_to_one")
    else:
        b["network_in_strength"] = np.nan
    # Secondary endpoints are encoded but never required for primary mortality eligibility.
    icu = first_column(b, ["MARCA_UTI", "any_icu", "icu_use"], required=False)
    if icu:
        icu_numeric = pd.to_numeric(b[icu].astype(str).str.strip(), errors="coerce")
        b["any_icu"] = icu_numeric.gt(0).where(icu_numeric.notna()).astype("Float64")
    else:
        b["any_icu"] = pd.Series(pd.NA, index=b.index, dtype="Float64")
    los = first_column(b, ["DIAS_PERM", "length_of_stay_days"], False)
    start, end = first_column(b, ["DT_INTER", "admission_date"], False), first_column(b, ["DT_SAIDA", "discharge_date"], False)
    if los:
        b["length_of_stay_days"] = pd.to_numeric(b[los], errors="coerce").where(
            lambda x: x.ge(0)
        )
    elif start and end:
        b["length_of_stay_days"] = (
            pd.to_datetime(b[end].astype(str), format="%Y%m%d", errors="coerce")
            - pd.to_datetime(b[start].astype(str), format="%Y%m%d", errors="coerce")
        ).dt.days.where(lambda x: x.ge(0))
    else:
        b["length_of_stay_days"] = np.nan
    payment = first_column(b, ["VAL_TOT", "reimbursement", "payment"], False)
    b["reimbursement_brl"] = pd.to_numeric(b[payment], errors="coerce") if payment else np.nan
    b["reimbursement_2025_brl"] = np.nan
    ipca_note = "not supplied; reimbursement secondary model disabled"
    if args.ipca:
        ipca = pd.read_parquet(args.ipca) if args.ipca.suffix.lower() == ".parquet" else pd.read_csv(args.ipca)
        im, ii = first_column(ipca, ["competence_month", "month"]), first_column(ipca, ["index", "ipca_index"])
        base = first_column(ipca, ["base_2025_index", "index_base_2025"], False)
        ipca = ipca.assign(competence_month=ipca[im].astype(str), _index=pd.to_numeric(ipca[ii], errors="coerce"))
        if not base:
            raise ValueError("IPCA input needs an explicit 2025 base-index field; implicit rebasing is not permitted")
        base_value = float(pd.to_numeric(ipca[base], errors="coerce").dropna().iloc[0])
        b = b.merge(ipca[["competence_month", "_index"]], on="competence_month", how="left", validate="many_to_one")
        b["reimbursement_2025_brl"] = b["reimbursement_brl"] * base_value / b["_index"]
        ipca_note = f"reimbursement adjusted with explicit monthly index; base index={base_value} (2025 BRL)"
    forbidden = ["any_icu", "length_of_stay_days", "reimbursement_brl", "reimbursement_2025_brl"]
    b["analysis_row_id"] = np.arange(len(b), dtype=np.int64)
    keep = ["analysis_row_id", "unique_aih_key", "cnes7", "provider_municipality", "competence_month", "calendar_month", "year", "residence_municipality", "matched_rd_record", "death_valid", "in_hospital_death", "trailing12_a_unique_aih", "trailing12_complete", "first_observed_coded_use_month", "months_since_first_observed_coded_use", "prevalent_at_window_start", "maintenance_status", "maintenance_evaluable", "age_years", "sex_category", "race_category", "emergency_admission", "diagnostic_stratum", "comorbidity_burden", "hospital_type", "beds_sus", "icu_beds", "endoscopy_capability", "state_provider", "ivs_context", "ans_context", "network_in_strength"] + forbidden
    out = b.reindex(columns=keep)
    destination = args.output_dir / "aim4_analytic_v2.parquet"; out.to_parquet(destination, index=False)
    qc = {"status": "PASS", "evidence": "associational", "n_cohort_b_adult_unique_aih": int(len(out)), "n_matched_rd": int(out.matched_rd_record.sum()), "death_valid_n": int(out.death_valid.sum()), "death_invalid_or_missing_n": int((~out.death_valid).sum()), "trailing12_complete_n": int(out.trailing12_complete.sum()), "trailing12_missing_n": int(out.trailing12_a_unique_aih.isna().sum()), "left_truncated_prevalent_n": int(out.prevalent_at_window_start.fillna(False).sum()), "maintenance_evaluable_n": int(out.maintenance_evaluable.fillna(False).sum()), "context_complete_n": int(out[["ivs_context", "ans_context"]].notna().all(axis=1).sum()), "network_in_strength_complete_n": int(out.network_in_strength.notna().sum()), "comorbidity": comorbidity_note, "admission_character_field": emergency, "ipca": ipca_note, "prohibited_primary_covariates_absent_from_primary_design": forbidden, "input_sha256": {str(p): sha256(p) for p in input_paths}, "output_sha256": sha256(destination)}
    qc.update({
        "provider_state_levels": sorted(out["state_provider"].dropna().unique().tolist()),
        "provider_state_n_levels": int(out["state_provider"].nunique()),
        "provider_state_source": "first two digits of SIH performing municipality (MUNIC_MOV); generic hospital-month state field forbidden",
        "cnes_single_state": bool(out.groupby("cnes7")["state_provider"].nunique().le(1).all()),
        "admission_character_mapping": "CAR_INT 01=ELECTIVE; 02-06=URGENT_OR_NON_ELECTIVE; other/missing explicit",
    })
    qc_path = args.output_dir / "aim4_analytic_v2_qc.json"; qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "aim4_analytic_v2_manifest.json").write_text(json.dumps({"inputs": qc["input_sha256"], "outputs": {destination.name: sha256(destination), qc_path.name: sha256(qc_path)}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
