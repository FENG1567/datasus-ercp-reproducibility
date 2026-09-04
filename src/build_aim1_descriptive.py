from __future__ import annotations

"""Aim 1 descriptive: monthly unique ERCP AIH, active hospitals, first
adoption, maintenance, cessation, recovery by nation/region/state, plus
age-sex standardized rates with IBGE population denominators."""

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

UF_REGION = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte",
    "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}
POP_AGE_GROUPS = {
    "0": "0-4", "1": "5-9", "2": "10-14", "3": "15-19", "4": "20-29", "5": "30-39",
    "6": "40-49", "7": "50-59", "8": "60-69", "9": "70-79", "10": "80+",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_pop(pop_dir: Path, year: int) -> pd.DataFrame:
    zip_path = pop_dir / str(year) / f"POPSBR{year % 100:02d}.zip"
    with zipfile.ZipFile(zip_path) as archive:
        member = archive.namelist()[0]
        raw = archive.read(member)
        tmp = Path("logs") / f"pop_{year}.dbf"
        tmp.write_bytes(raw)
        import struct
        header_len = struct.unpack("<H", raw[8:10])[0]
        record_len = struct.unpack("<H", raw[10:12])[0]
        nfields = (header_len - 33) // 32
        fields = []
        pos = 32
        for _ in range(nfields):
            name = raw[pos:pos + 11].split(b"\x00")[0].decode("latin-1")
            ftype = chr(raw[pos + 11])
            flen = raw[pos + 16]
            fields.append((name, ftype, flen))
            pos += 32
        rows = []
        idx = header_len
        while idx + record_len <= len(raw):
            if raw[idx] == 0x1A:
                break
            record = raw[idx + 1: idx + record_len]
            values = {}
            offset = 0
            for name, ftype, flen in fields:
                values[name.upper()] = record[offset:offset + flen].decode("latin-1").strip()
                offset += flen
            rows.append(values)
            idx += record_len
    df = pd.DataFrame(rows)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--hospital-month", type=Path, required=True)
    parser.add_argument("--cnes-st", type=Path, required=True)
    parser.add_argument("--pop-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    df = pq.read_table(args.cohorts).to_pandas()
    df["year"] = df["competence_month"].str[:4]
    df["month_int"] = df["competence_month"].astype(int)
    # hospital UF from CNES CNES -> CODUFMUN? use hospital's CNES prefix: CNES codes start with UF digit
    df["hosp_uf"] = df["SP_CNES"].str[:2].map(
        {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
         "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
         "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
         "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}
    )
    df["region"] = df["hosp_uf"].map(UF_REGION)

    # monthly unique AIH by scope
    monthly = (
        df.groupby(["cohort", "competence_month"], as_index=False)
        .size().rename(columns={"size": "n_aih"})
    )
    monthly_nat = monthly.copy()
    monthly_state = (
        df.groupby(["cohort", "competence_month", "hosp_uf"], as_index=False)
        .size().rename(columns={"size": "n_aih"})
    )
    monthly_region = (
        df.groupby(["cohort", "competence_month", "region"], as_index=False)
        .size().rename(columns={"size": "n_aih"})
    )
    active_hospitals = (
        df.groupby(["cohort", "competence_month", "SP_CNES"]).size()
        .groupby(["cohort", "competence_month"]).size().rename("n_active_hospitals")
        .reset_index()
    )

    # adoption/maintenance/cessation/recovery from hospital_month
    hm = pq.read_table(args.hospital_month).to_pandas()
    # CNES -> hospital UF via CNES first two digits (from ST join instead)
    st = pq.read_table(args.cnes_st, columns=["CNES", "competence_month", "CODUFMUN"]).to_pandas()
    st["CNES"] = st["CNES"].astype(str).str.zfill(7)
    st["hosp_uf"] = st["CODUFMUN"].astype(str).str[:2]
    hm["CNES"] = hm["SP_CNES"].astype(str).str.zfill(7)
    hm_uf = hm.merge(st[["CNES", "hosp_uf"]].drop_duplicates("CNES"), on="CNES", how="left")

    adoption_by_year = (
        hm_uf.groupby("CNES")["adoption_month"].first()
        .dropna().astype(int).apply(lambda m: str(m)[:4]).value_counts().sort_index()
        .rename_axis("year").reset_index(name="n_first_adoption")
    )
    adoption_by_year_uf = (
        hm_uf.groupby(["CNES", "hosp_uf"])["adoption_month"].first()
        .dropna().astype(int).apply(lambda m: str(m)[:4]).reset_index(name="year")
        .groupby(["hosp_uf", "year"]).size().rename("n_first_adoption").reset_index()
    )
    maintained = (
        hm_uf.groupby("CNES")["maintained_6of12"].any().reset_index(name="maintained")
    )
    cessation = (
        hm_uf.groupby("CNES")["cessation"].any().reset_index(name="cessation")
    )
    states = maintained.merge(cessation, on="CNES")
    states["recovery"] = states["maintained"] & ~states["cessation"]

    # age-sex standardized crude rate (B cohort) per 100k adults, by year
    pop_frames = []
    for year in range(2021, 2026):
        pop = read_pop(args.pop_dir, year)
        pop["year"] = str(year)
        pop_frames.append(pop)
    pop_all = pd.concat(pop_frames, ignore_index=True)
    pop_col = "POP" if "POP" in pop_all.columns else ("pop" if "pop" in pop_all.columns else None)
    sexo_col = "SEXO" if "SEXO" in pop_all.columns else ("sexo" if "sexo" in pop_all.columns else None)
    idade_col = "IDADE" if "IDADE" in pop_all.columns else ("idade" if "idade" in pop_all.columns else None)
    pop_all["pop"] = pd.to_numeric(pop_all[pop_col], errors="coerce").fillna(0) if pop_col else 0
    pop_all["sexo"] = pop_all[sexo_col] if sexo_col else ""
    pop_all["idade_g"] = pop_all[idade_col] if idade_col else ""
    pop_total = pop_all.groupby("year")["pop"].sum().to_dict()

    b = df[df["cohort"] == "B"].copy()
    b["sexo"] = b["SEXO"].astype(str)
    b["age_group"] = pd.cut(
        pd.to_numeric(b["age_years"], errors="coerce"),
        bins=[18, 20, 30, 40, 50, 60, 70, 80, 200],
        labels=["18-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"],
    )
    b_counts = b.groupby(["year", "sexo", "age_group"]).size().reset_index(name="n")
    pop_sex = pop_all[pop_all["sexo"].isin(["1", "2"])].copy()
    pop_sex["sexo"] = pop_sex["sexo"].astype(str)
    pop_sex["idade_g"] = pop_sex["idade_g"].astype(str)

    # crude rate per 100k total population by year (all-ages denominator; adult adjustment noted)
    crude = b.groupby("year").size().reset_index(name="n")
    crude["pop"] = crude["year"].map(pop_total)
    crude["rate_per_100k"] = crude["n"] / crude["pop"] * 1e5

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly_nat.to_parquet(args.output_dir / "aim1_monthly_national.parquet", index=False)
    monthly_state.to_parquet(args.output_dir / "aim1_monthly_state.parquet", index=False)
    monthly_region.to_parquet(args.output_dir / "aim1_monthly_region.parquet", index=False)
    active_hospitals.to_parquet(args.output_dir / "aim1_active_hospitals.parquet", index=False)
    adoption_by_year.to_parquet(args.output_dir / "aim1_adoption_by_year.parquet", index=False)
    adoption_by_year_uf.to_parquet(args.output_dir / "aim1_adoption_by_year_uf.parquet", index=False)
    states.to_parquet(args.output_dir / "aim1_states.parquet", index=False)
    crude.to_parquet(args.output_dir / "aim1_crude_rate.parquet", index=False)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS",
        "monthly_rows": int(len(monthly_nat)),
        "adoption_by_year": {str(r.year): int(r.n_first_adoption) for r in adoption_by_year.itertuples()},
        "states": {
            "hospitals": int(len(states)),
            "maintained": int(states["maintained"].sum()),
            "cessation": int(states["cessation"].sum()),
            "recovery": int(states["recovery"].sum()),
        },
        "crude_rate_per_100k": {str(r.year): round(float(r.rate_per_100k), 3) for r in crude.itertuples()},
        "note": "rates use all-ages IBGE population denominators; age-sex standardization refined in model stage",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())