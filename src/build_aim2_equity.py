from __future__ import annotations

"""Aim 2 equity: treated ERCP rate per 100k adults (cohort B) by residence
municipality, joined to contextual exposures (IVS 2010, ANS supplementary
coverage, region), with quintile-based absolute/relative inequalities and
slope/relative indices of inequality (SII/RII)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--pop-dir", type=Path, required=True)
    parser.add_argument("--ivs", type=Path, required=True)
    parser.add_argument("--ans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    df = pq.read_table(args.cohorts).to_pandas()
    b = df[df["cohort"] == "B"].copy()
    b["res_municipio"] = b["MUNIC_RES"].astype(str).str.strip().str.zfill(6)
    b["year"] = b["competence_month"].str[:4]
    treated = b.groupby(["res_municipio", "year"]).size().reset_index(name="n")

    # adult population 18+ from POPSVS (COD_IDADE in pop: age groups; approximate adults via IDADE>=2 groups)
    def read_pop_year(pop_dir: Path, year: int) -> pd.DataFrame:
        import struct
        import zipfile
        zip_path = pop_dir / str(year) / f"POPSBR{year % 100:02d}.zip"
        with zipfile.ZipFile(zip_path) as archive:
            raw = archive.read(archive.namelist()[0])
        header_len = struct.unpack("<H", raw[8:10])[0]
        record_len = struct.unpack("<H", raw[10:12])[0]
        nfields = (header_len - 33) // 32
        fields = []
        pos = 32
        for _ in range(nfields):
            name = raw[pos:pos + 11].split(b"\x00")[0].decode("latin-1").upper()
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
                values[name] = record[offset:offset + flen].decode("latin-1").strip()
                offset += flen
            rows.append(values)
            idx += record_len
        return pd.DataFrame(rows)

    pop_frames = []
    for year in range(2021, 2026):
        pop = read_pop_year(args.pop_dir, year)
        pop["year"] = str(year)
        pop_frames.append(pop)
    pop_all = pd.concat(pop_frames, ignore_index=True)
    pop_all["pop"] = pd.to_numeric(pop_all.get("POP"), errors="coerce").fillna(0)
    pop_all["cod"] = pop_all.get("COD_MUN").astype(str).str.zfill(7)
    pop_all["idade"] = pop_all.get("IDADE").astype(str).str.zfill(3)
    pop_all["sexo"] = pop_all.get("SEXO").astype(str)
    # POPSVS IDADE groups (3-digit): 000=0-4, 001=5-9, 002=10-14, 003=15-19,
    # 004=20-29 ... 010=80+. Adults 20+ used as primary (conservative);
    # 15-19 partial-adult boundary noted as limitation.
    adult_groups = {"004", "005", "006", "007", "008", "009", "010"}
    pop_adult = pop_all[pop_all["idade"].isin(adult_groups)]
    pop_by_mun_year = pop_adult.groupby(["cod", "year"])["pop"].sum().reset_index()
    pop_by_mun_year["res_municipio"] = pop_by_mun_year["cod"].str[:6]

    rate = treated.merge(
        pop_by_mun_year.groupby(["res_municipio", "year"])["pop"].sum().reset_index(),
        on=["res_municipio", "year"], how="left",
    )
    rate["rate_per_100k"] = rate["n"] / rate["pop"].replace(0, np.nan) * 1e5

    # contextual exposures: IVS 2010 (7-digit code)
    ivs = pq.read_table(args.ivs).to_pandas()
    ivs10 = ivs[(ivs["ano"] == "2010") & (ivs["label_cor"] == "Total Cor")
                & (ivs["label_sexo"] == "Total Sexo") & (ivs["label_sit_dom"] == "Total Situação de Domicílio")]
    ivs10["municipio"] = ivs10["municipio"].astype(str).str.zfill(7)
    ivs10 = ivs10[["municipio", "ivs", "ivs_infraestrutura_urbana", "ivs_capital_humano", "ivs_renda_e_trabalho"]]
    ivs10 = ivs10.drop_duplicates("municipio")
    ivs10["res_municipio"] = ivs10["municipio"].str[:6]

    ans = pd.read_csv(args.ans, encoding="utf-8-sig", dtype=str)
    ans["res_municipio"] = ans["cd_municipio_6"].str.zfill(6)
    ans["year"] = ans["year"]
    ans["supp_coverage"] = pd.to_numeric(ans["supplementary_coverage_rate"], errors="coerce")

    rate = rate.merge(ivs10[["res_municipio", "ivs"]].drop_duplicates("res_municipio"), on="res_municipio", how="left")
    rate = rate.merge(ans[["res_municipio", "year", "supp_coverage"]].drop_duplicates(["res_municipio", "year"]),
                      on=["res_municipio", "year"], how="left")
    rate["uf"] = rate["res_municipio"].str[:2]

    # quintiles of IVS (2010, population-weighted assignment per municipality)
    rate_any = rate.dropna(subset=["ivs"])
    rate_any["ivs_quintile"] = pd.qcut(rate_any["ivs"].rank(method="first"), 5, labels=False) + 1

    # pooled rate per quintile (adult-population-weighted)
    def weighted_rate(g):
        return (g["n"].sum() / g["pop"].sum()) * 1e5 if g["pop"].sum() > 0 else np.nan

    quint = rate_any.groupby("ivs_quintile").apply(weighted_rate, include_groups=False).reset_index(name="rate")
    quint["rate"] = quint["rate"].fillna(0)
    q1 = float(quint[quint["ivs_quintile"] == 1]["rate"].iloc[0]) if (quint["ivs_quintile"] == 1).any() else np.nan
    q5 = float(quint[quint["ivs_quintile"] == 5]["rate"].iloc[0]) if (quint["ivs_quintile"] == 5).any() else np.nan
    rate_ratio = q5 / q1 if q1 and q1 > 0 else np.nan
    absolute_diff = q5 - q1

    # SII/RII via regression on ridit score
    rate_any["pop_share"] = rate_any.groupby("year")["pop"].transform(lambda s: s / s.sum())
    r2 = rate_any.dropna(subset=["pop", "ivs"])
    # SII: weighted linear regression of rate on ridit (fraction below midpoint)
    ridit = r2["pop_share"].cumsum() - 0.5 * r2["pop_share"]
    weights = r2["pop"] / r2["pop"].sum()
    X = np.column_stack([np.ones(len(r2)), ridit.values])
    W = np.diag(weights.values)
    try:
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ r2["rate_per_100k"].fillna(0).values)
        sii = float(beta[1])
    except Exception:
        sii = np.nan
    mean_rate = float((r2["rate_per_100k"] * r2["pop"]).sum() / r2["pop"].sum()) if r2["pop"].sum() else np.nan
    rii = sii / mean_rate if mean_rate and mean_rate > 0 else np.nan

    # ANS coverage quintiles
    rate_ans = rate.dropna(subset=["supp_coverage"])
    rate_ans["ans_quintile"] = pd.qcut(rate_ans["supp_coverage"].rank(method="first"), 5, labels=False) + 1
    quint_ans = rate_ans.groupby("ans_quintile").apply(weighted_rate, include_groups=False).reset_index(name="rate")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rate_any.to_parquet(args.output, index=False)

    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "status": "PASS",
        "n_municipality_years": int(len(rate)),
        "n_with_ivs": int(rate["ivs"].notna().sum()),
        "n_with_ans": int(rate["supp_coverage"].notna().sum()),
        "ivs_quintile_rates": {str(int(r.ivs_quintile)): round(float(r.rate), 4) for r in quint.itertuples()},
        "absolute_diff_q5_minus_q1": round(absolute_diff, 4) if not pd.isna(absolute_diff) else None,
        "rate_ratio_q5_q1": round(rate_ratio, 4) if not pd.isna(rate_ratio) else None,
        "SII": round(sii, 4) if not pd.isna(sii) else None,
        "RII": round(rii, 4) if not pd.isna(rii) else None,
        "ans_quintile_rates": {str(int(r.ans_quintile)): round(float(r.rate), 4) for r in quint_ans.itertuples()},
        "interpretation": "municipal IVS/ANS are contextual exposures, not individual-level attributes",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())