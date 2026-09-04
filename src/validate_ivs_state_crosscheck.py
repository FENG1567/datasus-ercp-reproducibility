from __future__ import annotations

"""External validation of municipal IVS against official Ipeadata state-level
AVS_IVS values (2010): population-weighted municipal IVS aggregated to state
must agree within a pre-specified tolerance."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ivs-parquet", type=Path, required=True)
    parser.add_argument("--ipeadata-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--mean-abs-diff-tol", type=float, default=0.01)
    parser.add_argument("--correlation-min", type=float, default=0.98)
    args = parser.parse_args()

    ipeadata_raw = pd.read_json(args.ipeadata_json)
    ipeadata = ipeadata_raw["value"].apply(pd.Series) if "value" in ipeadata_raw.columns else ipeadata_raw
    ipeadata["ano"] = pd.to_datetime(ipeadata["VALDATA"], utc=True).dt.year
    state_official = ipeadata[(ipeadata["NIVNOME"] == "Estados") & (ipeadata["ano"] == 2010)]
    state_official = state_official.set_index("TERCODIGO")["VALVALOR"]

    df = pd.read_parquet(args.ivs_parquet)
    total = df[
        (df["ano"] == "2010")
        & (df["label_cor"] == "Total Cor")
        & (df["label_sexo"] == "Total Sexo")
        & (df["label_sit_dom"] == "Total Situação de Domicílio")
    ].copy()
    total["ivs"] = pd.to_numeric(total["ivs"], errors="coerce")
    total["populacao"] = pd.to_numeric(total["populacao"], errors="coerce")

    weighted = (
        total.groupby("uf")
        .apply(lambda g: (g["ivs"] * g["populacao"]).sum() / g["populacao"].sum(), include_groups=False)
        .rename("ivs_municipal_weighted")
    )
    comparison = pd.DataFrame({"uf_code": weighted.index, "ivs_municipal_weighted": weighted.values}).set_index(
        "uf_code"
    )
    comparison["ivs_ipeadata_state_2010"] = state_official
    comparison["difference"] = comparison["ivs_municipal_weighted"] - comparison["ivs_ipeadata_state_2010"]
    comparison["abs_difference"] = comparison["difference"].abs()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, encoding="utf-8-sig")

    mean_abs = float(comparison["abs_difference"].mean())
    corr = float(comparison["ivs_municipal_weighted"].corr(comparison["ivs_ipeadata_state_2010"]))
    n_states = int(len(comparison))
    passed = mean_abs <= args.mean_abs_diff_tol and corr >= args.correlation_min and n_states == 27
    audit = {
        "schema_version": "1.0",
        "status": "PASS" if passed else "FAIL",
        "n_states_compared": n_states,
        "mean_abs_difference": mean_abs,
        "max_abs_difference": float(comparison["abs_difference"].max()),
        "correlation": corr,
        "tolerance_mean_abs_diff": args.mean_abs_diff_tol,
        "tolerance_correlation_min": args.correlation_min,
        "note": "Official Ipeadata AVS_IVS state values (2010) vs population-weighted municipal IVS.",
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())