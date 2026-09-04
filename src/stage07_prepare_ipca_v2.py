from __future__ import annotations

"""Validate official SIDRA IPCA monthly variation and create a 2025-12 index."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MONTHS = pd.period_range("2021-01", "2025-12", freq="M").strftime("%Y%m").tolist()
SOURCE_URL = (
    "https://apisidra.ibge.gov.br/values/t/7060/n1/all/v/63/"
    "p/202101-202512/d/v63%202"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sidra(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 61:
        raise ValueError("SIDRA response must contain one header row and 60 monthly rows")
    header, records = payload[0], payload[1:]
    required_header = {
        "V": "Valor",
        "D2C": "Variável (Código)",
        "D3C": "Mês (Código)",
        "D4C": "Geral, grupo, subgrupo, item e subitem (Código)",
    }
    for key, label in required_header.items():
        if header.get(key) != label:
            raise ValueError(f"unexpected SIDRA header for {key}: {header.get(key)!r}")
    if any(str(row.get("D2C")) != "63" for row in records):
        raise ValueError("SIDRA rows do not all use IPCA monthly-variation variable 63")
    if any(str(row.get("D4C")) != "7169" for row in records):
        raise ValueError("SIDRA rows do not all use the general IPCA index category 7169")
    frame = pd.DataFrame(
        {
            "competence_month": [str(row.get("D3C")) for row in records],
            "monthly_variation_pct": pd.to_numeric(
                [row.get("V") for row in records], errors="raise"
            ),
            "series_label": [str(row.get("D2N")) for row in records],
            "category_label": [str(row.get("D4N")) for row in records],
        }
    ).sort_values("competence_month")
    if frame["competence_month"].tolist() != EXPECTED_MONTHS:
        raise ValueError("SIDRA response is not the complete 2021-01 to 2025-12 panel")
    if frame["competence_month"].duplicated().any():
        raise ValueError("duplicate SIDRA IPCA month")
    if not frame["monthly_variation_pct"].between(-10, 10).all():
        raise ValueError("monthly IPCA variation is outside the frozen plausibility bound")
    return frame.reset_index(drop=True)


def build_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cumulative_factor"] = (1.0 + result["monthly_variation_pct"] / 100.0).cumprod()
    december_2025 = float(
        result.loc[result["competence_month"].eq("202512"), "cumulative_factor"].iloc[0]
    )
    result["ipca_index"] = 100.0 * result["cumulative_factor"] / december_2025
    result["base_2025_index"] = 100.0
    result["index_base_month"] = "202512"
    if not np.isfinite(result["ipca_index"]).all() or result["ipca_index"].le(0).any():
        raise ValueError("derived IPCA index is nonfinite or nonpositive")
    if not np.isclose(result.loc[result["competence_month"].eq("202512"), "ipca_index"].iloc[0], 100.0):
        raise ValueError("December 2025 IPCA index is not exactly the declared base")
    return result.drop(columns="cumulative_factor")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--accessed-at", required=True)
    args = parser.parse_args()

    frame = parse_sidra(args.raw_json)
    output = build_index(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    checks = {
        "complete_60_month_panel": len(output) == 60,
        "unique_months": not output["competence_month"].duplicated().any(),
        "variable_63_ipca_monthly_variation": bool(
            output["series_label"].eq("IPCA - Variação mensal").all()
        ),
        "category_7169_general_index": bool(
            output["category_label"].eq("Índice geral").all()
        ),
        "december_2025_base_equals_100": bool(
            np.isclose(output.loc[output["competence_month"].eq("202512"), "ipca_index"].iloc[0], 100.0)
        ),
        "all_indices_finite_positive": bool(
            np.isfinite(output["ipca_index"]).all() and output["ipca_index"].gt(0).all()
        ),
    }
    audit = {
        "schema_version": "2.0",
        "generated_at": utc_now(),
        "status": "PASS" if all(checks.values()) else "FIX",
        "source": "IBGE SIDRA table 7060, variable 63, general index category 7169",
        "source_url": SOURCE_URL,
        "accessed_at": args.accessed_at,
        "unit": "monthly percentage variation",
        "index_definition": "Cumulative IPCA normalized to December 2025 = 100",
        "payment_interpretation": "SIH reimbursement converted to December-2025 BRL; not cost",
        "checks": checks,
        "hashes": {
            "raw_json_sha256": sha256_file(args.raw_json),
            "derived_parquet_sha256": sha256_file(args.output),
        },
        "artifacts": {"raw_json": str(args.raw_json), "derived_parquet": str(args.output)},
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
