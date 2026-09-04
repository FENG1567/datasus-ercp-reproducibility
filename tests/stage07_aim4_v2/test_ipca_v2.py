from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage07_prepare_ipca_v2", ROOT / "src" / "stage07_prepare_ipca_v2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_complete_sidra_panel_and_december_2025_base(tmp_path: Path) -> None:
    header = {
        "V": "Valor",
        "D2C": "Variável (Código)",
        "D3C": "Mês (Código)",
        "D4C": "Geral, grupo, subgrupo, item e subitem (Código)",
    }
    rows = [
        {
            "V": "0.50",
            "D2C": "63",
            "D2N": "IPCA - Variação mensal",
            "D3C": month,
            "D4C": "7169",
            "D4N": "Índice geral",
        }
        for month in MODULE.EXPECTED_MONTHS
    ]
    path = tmp_path / "ipca.json"
    path.write_text(json.dumps([header, *rows]), encoding="utf-8")
    parsed = MODULE.parse_sidra(path)
    indexed = MODULE.build_index(parsed)
    assert len(indexed) == 60
    assert np.isclose(indexed.loc[indexed.competence_month.eq("202512"), "ipca_index"].iloc[0], 100.0)
    assert indexed.loc[indexed.competence_month.eq("202101"), "ipca_index"].iloc[0] < 100.0
