#!/usr/bin/env python3
"""Versioned entry point for the Stage 7 rc_v1_2 figure/table source release.

The data transformations remain the validated rc_v1 implementation.  This
adapter binds that implementation to the updated public rc_v1_2 contract and
fails closed if either frozen file changes.
"""
from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

import stage07_build_figure_table_inputs_rc_v1 as base


CONTRACT_SCHEMA = "stage07_figures_tables_contract_rc_v1_2"
MANIFEST_SCHEMA = "stage07_figure_table_source_data_manifest_rc_v1_2"
AUDIT_SCHEMA = "stage07_figure_table_source_data_audit_rc_v1_2"
FROZEN_CONTRACT_SHA256 = "13f1988ea9067a04392ea9b02184e31600000c113dfe112883aa33540b7f7239"
BASE_BUILDER_SHA256 = "62c68f0dae738180c0ccfce30457542a0d96da902ffaef4fed93536738d1c492"

BuilderError = base.BuilderError
sha256_file = base.sha256_file
expected_outputs = base.expected_outputs


def _policy_header_role(value: str) -> str | None:
    text = unicodedata.normalize("NFKC", value).strip().lower()
    compact = re.sub(r"[\s_:/-]+", "", text)
    if compact in {"date", "data", "日期", "时间", "時間"}:
        return "date"
    if compact in {"event", "evento", "milestone", "事件", "节点", "節點"}:
        return "event"
    if compact in {"evidence", "evidencia", "evidência", "source", "官方证据", "官方證據", "证据", "證據"}:
        return "evidence"
    if compact in {"location", "local", "where", "证据位置", "證據位置", "位置"}:
        return "location"
    return None


def _parse_policy_markdown_rc_v1_2(record: dict[str, Any]) -> pd.DataFrame:
    """Parse the frozen multilingual policy table without translating evidence."""
    rows: list[list[str]] = []
    for line in record["path"].read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if not cells or all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        raise BuilderError("frozen policy markdown contains no pipe table")
    header_index = next(
        (index for index, row in enumerate(rows) if "date" in {_policy_header_role(cell) for cell in row}),
        None,
    )
    if header_index is None:
        raise BuilderError("frozen policy markdown table lacks a recognised multilingual date column")
    header = rows[header_index]
    roles = [_policy_header_role(cell) for cell in header]
    body = rows[header_index + 1 :]
    if not body:
        raise BuilderError("frozen policy markdown table has no rows")
    if any(len(row) != len(header) for row in body):
        raise BuilderError("frozen policy markdown table has inconsistent column counts")
    output = pd.DataFrame()
    for target in ("date", "event", "evidence", "location"):
        positions = [index for index, role in enumerate(roles) if role == target]
        if len(positions) > 1:
            raise BuilderError(f"frozen policy markdown has duplicate {target} columns")
        output[target] = [row[positions[0]] for row in body] if positions else pd.NA
    output["source_artifact"] = record["frozen_copy_of"] or record["path"].name
    output["source_sha256"] = record["sha256"]
    return output


def _privacy_gate_rc_v1_2(frame: pd.DataFrame, filename: str) -> None:
    """Apply small-count suppression while treating true booleans as indicators."""
    for column in frame.columns:
        if not base.SENSITIVE_COUNT_COLUMN.search(str(column)):
            continue
        if pd.api.types.is_bool_dtype(frame[column].dtype):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if ((values >= 1) & (values < 5)).any():
            raise BuilderError(f"privacy gate failed: unsuppressed count 1-4 in {filename}:{column}")


def _configure_base() -> None:
    observed = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    if observed != BASE_BUILDER_SHA256:
        raise BuilderError("validated rc_v1 builder SHA-256 mismatch")
    base.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    base.MANIFEST_SCHEMA = MANIFEST_SCHEMA
    base.AUDIT_SCHEMA = AUDIT_SCHEMA
    base.FROZEN_CONTRACT_SHA256 = FROZEN_CONTRACT_SHA256
    base._parse_policy_markdown = _parse_policy_markdown_rc_v1_2
    base._privacy_gate = _privacy_gate_rc_v1_2


def build_source_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the validated transformation code under the rc_v1_2 frozen bindings."""
    _configure_base()
    return base.build_source_data(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    try:
        _configure_base()
    except BuilderError as exc:
        print(f"SOURCE DATA BLOCKED: {exc}", file=sys.stderr)
        return 2
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
