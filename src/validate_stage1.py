from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
LOCKED = [
    CONFIG_DIR / "project.yaml",
    CONFIG_DIR / "cohorts.yaml",
    CONFIG_DIR / "estimands.yaml",
    CONFIG_DIR / "qc_gates.yaml",
    CONFIG_DIR / "missing_data.yaml",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    configs = {path.name: load_yaml(path) for path in LOCKED}
    project = configs["project.yaml"]
    cohorts = configs["cohorts.yaml"]
    estimands = configs["estimands.yaml"]
    gates = configs["qc_gates.yaml"]
    missing = configs["missing_data.yaml"]

    if project.get("primary_window") != {"start": "2021-01", "end": "2025-12"}:
        errors.append("Primary window must remain 2021-01 through 2025-12")
    if project.get("compute", {}).get("total_thread_ceiling") != 8:
        errors.append("Thread ceiling must equal 8")
    if cohorts.get("ercp_procedure_code") != "0407030255":
        errors.append("Therapeutic ERCP code mismatch")
    if cohorts.get("unique_aih_key") != ["competence_month", "SP_CNES", "SP_NAIH"]:
        errors.append("Unique AIH key changed")
    if set(cohorts["cohorts"]["choledocholithiasis_strict_adult"]["diagnosis_codes"]) != {"K803", "K804", "K805"}:
        errors.append("Strict choledocholithiasis phenotype mismatch")
    if estimands["eligible_hospital_risk_sets"].get("forbidden") != "using all hospitals in Brazil as the adoption denominator":
        errors.append("Eligible-hospital denominator guard missing")
    if gates["quasi_causal"].get("failure_action", "").startswith("REMOVE") is False:
        errors.append("Quasi-causal fail-closed action missing")
    if missing["principles"][0].startswith("Never impute identifiers") is False:
        errors.append("Non-imputation guard missing")

    # Construct sensitive sentinels at runtime so the validator does not
    # trigger on its own source code.
    forbidden_tokens = ["master" + "2333", "biotrainee" + ".vip"]
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden_tokens:
            if token in text:
                errors.append(f"Credential or connection secret found in {path.relative_to(ROOT)}")

    sap = ROOT / "reports" / "SAP_v1.md"
    if not sap.exists() or sap.stat().st_size < 5000:
        errors.append("SAP_v1.md is missing or implausibly short")

    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in [*LOCKED, sap]}
    status = "PASS" if not errors else "FIX"
    return {
        "schema_version": "1.0",
        "stage": 1,
        "validator": "deterministic stage-one contract check",
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "hashes": hashes,
    }


def main() -> int:
    result = validate()
    output = ROOT / "reports" / "stage01_qc.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
