from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_stage07_aim4_brglm2_v2.sh"


def runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_runner_is_server_tmux_launcher_with_strict_preflight() -> None:
    text = runner_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    for required in [
        'RUN_SESSION="stage07_aim4_brglm2_v2"',
        'COVERAGE_SESSION="stage07_aim2_coverage"',
        '[[ -f RUN_STATE.yaml && -f config/project.yaml && -d src && -d scripts && -f "$R_FITTER" && -f "$FINALIZER" && -f "$SCRIPT_PATH" ]]',
        'tmux has-session -t "$COVERAGE_SESSION"',
        'tmux new-session -d -s "$RUN_SESSION"',
        'tmux display-message -p \'#S\'',
        'R_LIBS_USER="$PROJECT_ROOT/.Rlib"',
        'required_command in tmux Rscript sha256sum',
        'brglm2", "detectseparation", "jsonlite',
        'import numpy, pandas',
        'aim4_brglm2_input_v2.csv.gz',
        'aim4_brglm2_design_v2.json',
    ]:
        assert required in text


def test_runner_has_frozen_parallel_bootstrap_and_ordered_finalization() -> None:
    text = runner_text()
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        assert f"export {variable}=1" in text
    assert "--point-only" in text
    assert 'point.get("primary", {}).get("status") == "valid"' in text
    assert 'point.get("sensitivity", {}).get("status") == "valid"' in text
    assert 'point.get("detectseparation_audit", {}).get("status") == "PASS"' in text
    assert '"$LOG_DIR/point_only_qc.log"' in text
    assert "--bootstrap-only" in text
    assert "--overwrite" not in text
    assert "SHARD_STARTS=(1 251 501 751 1001 1251 1501 1751)" in text
    assert "SHARD_ENDS=(250 500 750 1000 1250 1500 1750 2000)" in text
    assert 'if wait "${shard_pids[$shard_index]}"; then' in text
    assert '"$PYTHON_BIN" "$FINALIZER"' in text
    assert text.index('if [[ "$aggregate_rc" -ne 0 ]]; then') < text.index('"$PYTHON_BIN" "$FINALIZER"')
    assert 'record_master_exit' in text
    assert '"$LOG_DIR/$shard_label.exitcode"' in text
    assert '"$LOG_DIR/master.exitcode"' in text
    assert 'LOG_DIR="logs/stage07/aim4_v2_corrected"' in text


def test_runner_never_manages_or_stops_graphhopper() -> None:
    text = runner_text().lower()
    for forbidden in ["kill-session", "kill-server", "pkill", "graphhopper"]:
        assert forbidden not in text
    assert re.search(r"tmux\s+has-session\s+-t\s+\"\$coverage_session\"", text)
