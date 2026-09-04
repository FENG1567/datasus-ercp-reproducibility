from __future__ import annotations

"""Stage-7 Aim 3 service-area summaries from observed treatment flow.

Service area is a descriptive label for a modal observed treatment destination.
It never converts an absence of observed treatment into an absence of need.
Potential road coverage and IVS are contextual planning variables only.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def z6(value: object) -> str:
    return str(value).strip().zfill(6)


def find_column(frame: pd.DataFrame, choices: tuple[str, ...], required: bool = True) -> str | None:
    lookup = {column.lower(): column for column in frame.columns}
    for choice in choices:
        if choice.lower() in lookup:
            return lookup[choice.lower()]
    if required:
        raise ValueError(f"expected one of {choices}; available={list(frame.columns)}")
    return None


def _ivs_municipality_column(frame: pd.DataFrame) -> tuple[str, str]:
    """Resolve the IVS municipality key without silently choosing a 7-digit key."""
    lookup = {str(column).lower(): column for column in frame.columns}
    six_digit = lookup.get("municipio_6digt")
    if six_digit is not None:
        return six_digit, "municipio_6digt"

    compatibility = [
        lookup[name]
        for name in ("municipio", "res_municipio", "code6")
        if name in lookup
    ]
    if len(compatibility) != 1:
        raise ValueError(
            "IVS requires municipio_6digt; compatibility fallback must expose "
            "exactly one of municipio, res_municipio, or code6"
        )
    return compatibility[0], "compatibility:" + str(compatibility[0])


def _normalise_ivs_municipality(series: pd.Series, column: str, source_kind: str) -> pd.Series:
    """Return six-digit municipality codes, failing closed on malformed keys."""
    values = series.astype("string").str.strip()
    # Parquet files written from integer-valued floats may expose ``110001.0``.
    values = values.str.replace(r"(?<=\d)\.0+$", "", regex=True)
    if values.isna().any() or values.eq("").any() or (~values.str.fullmatch(r"\d+")).any():
        raise ValueError(f"IVS municipality key {column!r} contains missing or non-numeric values")

    lengths = values.str.len()
    if source_kind == "municipio_6digt":
        if (~lengths.between(1, 6)).any():
            raise ValueError("municipio_6digt must contain at most six digits")
        return values.str.zfill(6)

    # Explicit compatibility behavior: legacy ``municipio`` is commonly a
    # seven-digit IBGE code whose check digit is dropped; a six-digit legacy
    # key is already in residence-code form.  Other aliases are six-digit.
    if source_kind.lower() == "compatibility:municipio":
        if (~lengths.isin([6, 7])).any():
            raise ValueError("legacy municipio must contain six- or seven-digit numeric codes")
        return values.where(lengths.eq(6), values.str[:6]).str.zfill(6)
    if (~lengths.between(1, 6)).any():
        raise ValueError(f"compatibility IVS municipality key {column!r} must contain at most six digits")
    return values.str.zfill(6)


def _select_ivs_context(ivs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select the frozen 2010 municipal-total IVS exposure and its audit."""
    input_rows = int(len(ivs))
    strata = {
        "year": find_column(ivs, ("ano", "year")),
        "label_cor": find_column(ivs, ("label_cor",)),
        "label_sexo": find_column(ivs, ("label_sexo",)),
        "label_sit_dom": find_column(ivs, ("label_sit_dom",)),
    }
    ivs_value = find_column(ivs, ("ivs", "ivs_rank", "ivs_ridit"))
    muni_column, muni_kind = _ivs_municipality_column(ivs)

    year = pd.to_numeric(ivs[strata["year"]], errors="coerce")
    # Labels are compared exactly to the frozen Stage-2/Aim-2 strata values.
    label_cor = ivs[strata["label_cor"]].astype("string")
    label_sexo = ivs[strata["label_sexo"]].astype("string")
    label_sit_dom = ivs[strata["label_sit_dom"]].astype("string")
    selected_mask = (
        year.eq(2010)
        & label_cor.eq("Total Cor")
        & label_sexo.eq("Total Sexo")
        & label_sit_dom.eq("Total Situação de Domicílio")
    )
    selected_raw = ivs.loc[selected_mask].copy()
    if selected_raw.empty:
        raise ValueError(
            "IVS has no 2010 municipal-total row with the frozen total strata labels"
        )

    selected = pd.DataFrame(
        {
            "res_municipio": _normalise_ivs_municipality(
                selected_raw[muni_column], muni_column, muni_kind
            ),
            "ivs": pd.to_numeric(selected_raw[ivs_value], errors="coerce"),
        },
        index=selected_raw.index,
    )
    if selected["ivs"].isna().any() or (
        ~np.isfinite(selected["ivs"].to_numpy(dtype=float))
    ).any():
        raise ValueError("selected 2010 total-strata IVS values must be finite")
    if selected.duplicated("res_municipio").any():
        duplicate_count = int(selected.duplicated("res_municipio").sum())
        raise ValueError(
            "selected 2010 total-strata IVS rows must be unique by municipio_6digt; "
            f"duplicate_rows={duplicate_count}"
        )
    selected = selected.sort_values("res_municipio").reset_index(drop=True)

    audit = {
        "selection_rule": {
            "year": 2010,
            "label_cor": "Total Cor",
            "label_sexo": "Total Sexo",
            "label_sit_dom": "Total Situação de Domicílio",
            "municipality_code_column": muni_column,
            "municipality_code_source": muni_kind,
            "context_semantics": (
                "static 2010 municipal-total contextual exposure carried across coverage years"
            ),
        },
        "input_rows": input_rows,
        "selected_rows": int(len(selected)),
        "unique_municipalities": int(selected["res_municipio"].nunique()),
        "context_semantics": (
            "IVS 2010 is a municipality-level contextual exposure, not an individual attribute; "
            "the same value is carried across every coverage year"
        ),
    }
    return selected, audit


def normalise_context(coverage_path: Path | None, ivs_path: Path | None) -> pd.DataFrame:
    if coverage_path is None:
        output = pd.DataFrame(
            columns=[
                "res_municipio",
                "year",
                "adult_population",
                "potential_coverage_120",
                "potential_coverage_180",
            ]
        )
        output.attrs["ivs_audit"] = {
            "selection_rule": None,
            "input_rows": 0,
            "selected_rows": 0,
            "output_rows": 0,
            "unique_municipalities": 0,
            "context_semantics": (
                "No IVS input supplied; no contextual exposure was attached"
            ),
        }
        return output
    frame = pd.read_parquet(coverage_path)
    muni = find_column(frame, ("municipio", "res_municipio", "code6")); year = find_column(frame, ("year", "ano"))
    population = find_column(frame, ("adult_population", "population_adult", "pop_adult", "pop"))
    cov120 = find_column(frame, ("has_provider_120", "within_120", "covered_120", "has_hospital_120", "potential_coverage_120"), required=False)
    cov180 = find_column(frame, ("has_provider_180", "within_180", "covered_180", "has_hospital_180", "potential_coverage_180"), required=False)
    output = pd.DataFrame({"res_municipio": frame[muni].map(z6), "year": frame[year].astype(str), "adult_population": pd.to_numeric(frame[population], errors="coerce"), "potential_coverage_120": frame[cov120] if cov120 else np.nan, "potential_coverage_180": frame[cov180] if cov180 else np.nan})
    ivs_audit: dict[str, object] = {
        "selection_rule": None,
        "input_rows": 0,
        "selected_rows": 0,
        "unique_municipalities": 0,
        "context_semantics": "No IVS input supplied; no contextual exposure was attached",
    }
    if ivs_path is not None:
        ivs = pd.read_parquet(ivs_path)
        selected, ivs_audit = _select_ivs_context(ivs)
        output = output.merge(
            selected,
            on="res_municipio",
            how="left",
            validate="many_to_one",
        )
    output = output.drop_duplicates(["res_municipio", "year"]).reset_index(drop=True)
    ivs_audit["output_rows"] = int(len(output))
    output.attrs["ivs_audit"] = ivs_audit
    return output


def entropy_diversity(shares: np.ndarray) -> float:
    shares = shares[shares > 0]
    return float(-(shares * np.log(shares)).sum()) if len(shares) else float("nan")


def build_service_area(edges: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"layer", "cohort", "year", "res_municipio", "treat_municipio", "n_aih"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"municipality edge input missing {sorted(missing)}")
    edge = edges[(edges["layer"] == "performing_municipio") & edges["cohort"].astype(str).isin(["A", "B"])].copy()
    edge["year"] = edge["year"].astype(str); edge["res_municipio"] = edge["res_municipio"].map(z6); edge["treat_municipio"] = edge["treat_municipio"].map(z6); edge["n_aih"] = pd.to_numeric(edge["n_aih"], errors="raise")
    if (edge["n_aih"] <= 0).any() or edge.duplicated(["cohort", "year", "res_municipio", "treat_municipio"]).any():
        raise ValueError("nonpositive or nonunique municipality flow edges")
    summary_rows: list[dict[str, object]] = []
    service_rows: list[dict[str, object]] = []
    for (cohort, year), current in edge.groupby(["cohort", "year"], sort=True):
        treatment = current.groupby("res_municipio")["n_aih"].sum()
        for residence, total in treatment.items():
            destinations = current[current["res_municipio"] == residence].sort_values(["n_aih", "treat_municipio"], ascending=[False, True])
            shares = destinations["n_aih"].to_numpy(float) / float(total)
            main = destinations.iloc[0]
            summary_rows.append({"cohort": cohort, "year": year, "res_municipio": residence, "observed_treatment": True, "n_aih": int(total), "self_sufficiency": float(destinations.loc[destinations["treat_municipio"] == residence, "n_aih"].sum() / total), "n_destinations": int(len(destinations)), "destination_hhi": float(np.square(shares).sum()), "destination_diversity_entropy": entropy_diversity(shares), "main_destination": str(main.treat_municipio), "main_destination_share": float(main.n_aih / total), "cross_municipality_share": float(destinations.loc[destinations["treat_municipio"] != residence, "n_aih"].sum() / total), "cross_state_share": float(destinations.loc[destinations["treat_municipio"].str[:2] != residence[:2], "n_aih"].sum() / total)})
        dominant = pd.DataFrame(summary_rows)[lambda x: (x["cohort"] == cohort) & (x["year"] == year)]
        for destination, group in dominant.groupby("main_destination"):
            service_rows.append({"cohort": cohort, "year": year, "performing_municipio": destination, "n_residences_with_modal_destination": int(len(group)), "observed_flow_to_service_area": int(current.loc[current["treat_municipio"] == destination, "n_aih"].sum()), "adult_population_modal_service_area": np.nan})
    summary = pd.DataFrame(summary_rows)
    # Complete official municipality-year context is appended with zero observed flow.
    if not context.empty:
        complete = context.copy(); complete["year"] = complete["year"].astype(str)
        additions = []
        for cohort in ("B", "A"):
            basis = complete[["res_municipio", "year"]].copy(); basis["cohort"] = cohort
            additions.append(basis.merge(summary[["cohort", "year", "res_municipio"]], on=["cohort", "year", "res_municipio"], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge"))
        zero = pd.concat(additions, ignore_index=True)
        zero["observed_treatment"] = False; zero["n_aih"] = 0; zero["self_sufficiency"] = np.nan; zero["n_destinations"] = 0; zero["destination_hhi"] = np.nan; zero["destination_diversity_entropy"] = np.nan; zero["main_destination"] = pd.NA; zero["main_destination_share"] = np.nan; zero["cross_municipality_share"] = np.nan; zero["cross_state_share"] = np.nan
        summary = pd.concat([summary, zero[summary.columns]], ignore_index=True)
        summary = summary.merge(complete, on=["res_municipio", "year"], how="left", validate="many_to_one")
    # Ensure missing potential coverage remains missing, never false/zero.
    if "potential_coverage_120" not in summary:
        summary["potential_coverage_120"] = np.nan
    if "potential_coverage_180" not in summary:
        summary["potential_coverage_180"] = np.nan
    services = pd.DataFrame(service_rows)
    if not services.empty and not context.empty:
        population_by_origin = summary[summary["observed_treatment"]].groupby(["cohort", "year", "main_destination"], as_index=False)["adult_population"].sum(min_count=1).rename(columns={"main_destination": "performing_municipio", "adult_population": "adult_population_modal_service_area"})
        services = services.drop(columns="adult_population_modal_service_area").merge(population_by_origin, on=["cohort", "year", "performing_municipio"], how="left")
    return summary.sort_values(["cohort", "year", "res_municipio"]), services.sort_values(["cohort", "year", "performing_municipio"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--municipality-edges", type=Path, required=True)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--ivs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--min-display", type=int, default=5)
    args = parser.parse_args()
    if args.min_display != 5:
        raise ValueError("public suppression threshold is frozen at n<5")
    args.output_dir.mkdir(parents=True, exist_ok=True); args.audit.parent.mkdir(parents=True, exist_ok=True)
    edges = pd.read_parquet(args.municipality_edges); context = normalise_context(args.coverage, args.ivs)
    ivs_audit = dict(context.attrs.get("ivs_audit", {}))
    residence, service = build_service_area(edges, context)
    residence.to_parquet(args.output_dir / "aim3_service_area_residence_v2.parquet", index=False)
    service.to_parquet(args.output_dir / "aim3_service_area_provider_v2.parquet", index=False)
    display = residence.copy(); display["n_aih_display"] = np.where(display["n_aih"] < 5, "<5", display["n_aih"].astype(int).astype(str)); display.drop(columns=["n_aih"]).to_csv(args.output_dir / "aim3_service_area_residence_display_v2.csv", index=False)
    checks = {"unique_residence_key": not residence.duplicated(["cohort", "year", "res_municipio"]).any(), "all_observed_flows_positive": bool(residence.loc[residence["observed_treatment"], "n_aih"].gt(0).all()), "coverage_missing_preserved": bool(residence.loc[residence["potential_coverage_120"].isna(), "potential_coverage_120"].isna().all()), "public_suppression_n_lt_5": args.min_display == 5, "zero_treatment_not_interpreted_as_no_need": True, "finite_observed_metrics": bool(np.isfinite(residence.loc[residence["observed_treatment"], ["self_sufficiency", "destination_hhi", "main_destination_share"]].to_numpy()).all()), "ivs_context_audit_rows_match": ivs_audit.get("output_rows", len(context)) == len(context)}
    audit = {"schema_version": "2.0", "generated_at": utc_now(), "status": "PASS" if all(checks.values()) else "FIX", "evidence_level": "descriptive observed treatment flow with contextual planning covariates; no effect estimate", "definition": "service-area = modal observed treating municipality for a residence municipality", "coverage_rule": "Potential coverage is retained as missing when unavailable; observed zero treatment is not evidence of no service need.", "ivs_audit": ivs_audit, "checks": checks, "input_hashes": {"municipality_edges_sha256": sha256_file(args.municipality_edges), "coverage_sha256": sha256_file(args.coverage) if args.coverage else None, "ivs_sha256": sha256_file(args.ivs) if args.ivs else None}}
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"service-area QC failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
