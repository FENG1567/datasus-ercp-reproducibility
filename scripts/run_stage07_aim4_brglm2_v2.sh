#!/usr/bin/env bash
# Server-only launcher for the frozen Aim 4 brglm2 fallback.  Invoke this
# script directly; it creates exactly one detached tmux session for the run.

set -u
set -o pipefail

RUN_SESSION="stage07_aim4_brglm2_v2"
COVERAGE_SESSION="stage07_aim2_coverage"
RUN_MODE="${1:-}"

if [[ -n "$RUN_MODE" && "$RUN_MODE" != "--run-in-tmux" ]]; then
    printf 'ERROR: unsupported argument: %s\n' "$RUN_MODE" >&2
    exit 64
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/$(basename -- "${BASH_SOURCE[0]}")"
INPUT_GZ="data_analytic/stage07_rebuild/aim4_v2_corrected/brglm2/aim4_brglm2_input_v2.csv.gz"
DESIGN_JSON="data_analytic/stage07_rebuild/aim4_v2_corrected/brglm2/aim4_brglm2_design_v2.json"
OUTPUT_DIR="data_analytic/stage07_rebuild/aim4_v2_corrected/brglm2"
LOG_DIR="logs/stage07/aim4_v2_corrected"
R_FITTER="src/stage07_fit_aim4_brglm2_v2.R"
FINALIZER="src/stage07_finalize_aim4_brglm2_v2.py"
PYTHON_BIN="${PYTHON_BIN:-.venv_server/bin/python}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

cd "$PROJECT_ROOT" || exit 2
[[ -f RUN_STATE.yaml && -f config/project.yaml && -d src && -d scripts && -f "$R_FITTER" && -f "$FINALIZER" && -f "$SCRIPT_PATH" ]] || fail "project-root preflight failed: $PROJECT_ROOT"

export R_LIBS_USER="$PROJECT_ROOT/.Rlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for required_command in tmux Rscript sha256sum; do
    command -v "$required_command" >/dev/null 2>&1 || fail "required command is unavailable: $required_command"
done
[[ -x "$PYTHON_BIN" ]] || fail "required Python runtime is unavailable: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import numpy, pandas' >/dev/null 2>&1 || fail "Python dependencies numpy and pandas are required"
Rscript --vanilla -e 'ok <- all(vapply(c("brglm2", "detectseparation", "jsonlite"), requireNamespace, logical(1), quietly=TRUE)); quit(status=if (ok) 0 else 1)' || fail "R dependencies brglm2, detectseparation, and jsonlite are required in R_LIBS_USER=$R_LIBS_USER"

[[ -f "$INPUT_GZ" ]] || fail "frozen Aim 4 input is missing: $INPUT_GZ"
[[ -f "$DESIGN_JSON" ]] || fail "frozen Aim 4 design is missing: $DESIGN_JSON"
if tmux has-session -t "$COVERAGE_SESSION" 2>/dev/null; then
    fail "coverage session $COVERAGE_SESSION is still present; wait until it has ended"
fi

if [[ "$RUN_MODE" != "--run-in-tmux" ]]; then
    if tmux has-session -t "$RUN_SESSION" 2>/dev/null; then
        fail "Aim 4 runner session $RUN_SESSION already exists"
    fi
    mkdir -p "$LOG_DIR"
    printf 'LAUNCHING\n' > "$LOG_DIR/launcher.status"
    printf -v tmux_command 'cd %q && exec bash %q --run-in-tmux' "$PROJECT_ROOT" "$SCRIPT_PATH"
    tmux new-session -d -s "$RUN_SESSION" "$tmux_command"
    launcher_rc=$?
    printf '%s\n' "$launcher_rc" > "$LOG_DIR/launcher.exitcode"
    [[ "$launcher_rc" -eq 0 ]] || exit "$launcher_rc"
    printf 'Aim 4 brglm2 runner started in tmux session %s\n' "$RUN_SESSION"
    exit 0
fi

[[ -n "${TMUX:-}" ]] || fail "--run-in-tmux is only valid inside the dedicated tmux session"
[[ "$(tmux display-message -p '#S')" == "$RUN_SESSION" ]] || fail "runner must execute only in tmux session $RUN_SESSION"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
record_master_exit() {
    local status=$?
    printf '%s\n' "$status" > "$LOG_DIR/master.exitcode"
    trap - EXIT
    exit "$status"
}
trap record_master_exit EXIT
printf 'RUNNING\n' > "$LOG_DIR/master.status"

Rscript --vanilla "$R_FITTER" \
    --input-gz "$INPUT_GZ" \
    --design-json "$DESIGN_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --point-only > "$LOG_DIR/point_only.log" 2>&1
point_rc=$?
if [[ "$point_rc" -eq 0 ]]; then
    "$PYTHON_BIN" -c 'import json, sys; point = json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if point.get("schema_version") == "aim4_brglm2_v2" and point.get("primary", {}).get("status") == "valid" and point.get("sensitivity", {}).get("status") == "valid" and point.get("detectseparation_audit", {}).get("status") == "PASS" else 1)' "$OUTPUT_DIR/aim4_brglm2_point_estimate_v2.json" > "$LOG_DIR/point_only_qc.log" 2>&1
    point_rc=$?
fi
printf '%s\n' "$point_rc" > "$LOG_DIR/point_only.exitcode"
if [[ "$point_rc" -ne 0 ]]; then
    exit "$point_rc"
fi

SHARD_STARTS=(1 251 501 751 1001 1251 1501 1751)
SHARD_ENDS=(250 500 750 1000 1250 1500 1750 2000)
declare -a shard_pids

for shard_index in "${!SHARD_STARTS[@]}"; do
    start="${SHARD_STARTS[$shard_index]}"
    end="${SHARD_ENDS[$shard_index]}"
    shard_label=$(printf 'shard_%02d_%04d_%04d' "$((shard_index + 1))" "$start" "$end")
    Rscript --vanilla "$R_FITTER" \
        --input-gz "$INPUT_GZ" \
        --design-json "$DESIGN_JSON" \
        --output-dir "$OUTPUT_DIR" \
        --bootstrap-only \
        --replicate-start "$start" \
        --replicate-end "$end" > "$LOG_DIR/$shard_label.log" 2>&1 &
    shard_pids[$shard_index]=$!
done

aggregate_rc=0
for shard_index in "${!shard_pids[@]}"; do
    start="${SHARD_STARTS[$shard_index]}"
    end="${SHARD_ENDS[$shard_index]}"
    shard_label=$(printf 'shard_%02d_%04d_%04d' "$((shard_index + 1))" "$start" "$end")
    if wait "${shard_pids[$shard_index]}"; then
        shard_rc=0
    else
        shard_rc=$?
        if [[ "$aggregate_rc" -eq 0 ]]; then
            aggregate_rc=$shard_rc
        fi
    fi
    printf '%s\n' "$shard_rc" > "$LOG_DIR/$shard_label.exitcode"
done
printf '%s\n' "$aggregate_rc" > "$LOG_DIR/bootstrap_shards.exitcode"
if [[ "$aggregate_rc" -ne 0 ]]; then
    exit "$aggregate_rc"
fi

"$PYTHON_BIN" "$FINALIZER" \
    --replicate-dir "$OUTPUT_DIR/replicates" \
    --point-estimate "$OUTPUT_DIR/aim4_brglm2_point_estimate_v2.json" \
    --output-dir "$OUTPUT_DIR" > "$LOG_DIR/finalizer.log" 2>&1
finalizer_rc=$?
printf '%s\n' "$finalizer_rc" > "$LOG_DIR/finalizer.exitcode"
exit "$finalizer_rc"
