#!/usr/bin/env bash
# Launch the authorized v3 bootstrap only after the immutable point gate exists.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-.venv_server/bin/python}"
R_LIBS="${R_LIBS:-$PROJECT_ROOT/.Rlib}"
RUN_SESSION="stage07_aim4_v3_bootstrap"
COVERAGE_SESSION="stage07_aim2_coverage"
OUTPUT_DIR="data_analytic/stage07_rebuild/aim4_v3_stabilized/brglm2"
REPLICATE_DIR="$OUTPUT_DIR/replicates"
LOG_DIR="logs/stage07/aim4_v3/bootstrap"
INPUT_GZ="$OUTPUT_DIR/aim4_brglm2_input_v3_scaled.csv.gz"
DESIGN_JSON="$OUTPUT_DIR/aim4_brglm2_design_v3_scaled.json"
PREFIT_MANIFEST="$OUTPUT_DIR/aim4_brglm2_prefit_manifest_v3.json"
POINT_JSON="$OUTPUT_DIR/aim4_brglm2_point_estimate_v3.json"
ENVIRONMENT_LOCK="provenance/stage07/aim4_v3/environment_lock.json"
RUN_MANIFEST="$OUTPUT_DIR/aim4_brglm2_bootstrap_run_manifest_v3.json"
R_BOOTSTRAP="src/stage07_bootstrap_aim4_brglm2_v3.R"
FINALIZER="src/stage07_finalize_aim4_brglm2_v3.py"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
cd "$PROJECT_ROOT"
export R_LIBS
for variable in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS POLARS_MAX_THREADS; do export "$variable=1"; done

[[ -x "$PYTHON_BIN" ]] || fail "required Python runtime is unavailable: $PYTHON_BIN"
for command in tmux Rscript sha256sum curl mktemp; do command -v "$command" >/dev/null 2>&1 || fail "required command is unavailable: $command"; done
"$PYTHON_BIN" -c 'import numpy, pandas' >/dev/null 2>&1 || fail "Python dependencies numpy and pandas are required"
Rscript --vanilla -e 'for (p in c("brglm2", "detectseparation", "jsonlite")) if (!requireNamespace(p, quietly=TRUE)) quit(status=2)' >/dev/null 2>&1 || fail "R dependencies brglm2, detectseparation, and jsonlite are required"
[[ -f "$INPUT_GZ" && -f "$DESIGN_JSON" && -f "$PREFIT_MANIFEST" && -f "$POINT_JSON" && -f "$ENVIRONMENT_LOCK" && -f "$R_BOOTSTRAP" && -f "$FINALIZER" ]] || fail "required v3 artifact is unavailable"
tmux has-session -t "$COVERAGE_SESSION" 2>/dev/null && fail "coverage session $COVERAGE_SESSION is still present; bootstrap is deferred"
curl -fsS http://127.0.0.1:19999/health >/dev/null || fail "GraphHopper health check failed (runner does not manage GraphHopper)"
! tmux has-session -t "$RUN_SESSION" 2>/dev/null || fail "bootstrap session already exists: $RUN_SESSION"
for artifact in "$OUTPUT_DIR/aim4_brglm2_bootstrap_merged_v3.csv.gz" "$OUTPUT_DIR/aim4_brglm2_final_qc_v3.json" "$OUTPUT_DIR/aim4_brglm2_final_manifest_v3.json"; do [[ ! -e "$artifact" ]] || fail "immutable final artifact already exists: $artifact"; done

"$PYTHON_BIN" - "$INPUT_GZ" "$DESIGN_JSON" "$PREFIT_MANIFEST" "$POINT_JSON" "$ENVIRONMENT_LOCK" "$R_BOOTSTRAP" "$FINALIZER" "$0" "$RUN_MANIFEST" <<'PY'
import hashlib, json, pathlib, subprocess, sys
input_gz, design, prefit, point_path, environment_lock, r_bootstrap, finalizer, runner, manifest = map(pathlib.Path, sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()
point = json.loads(point_path.read_text(encoding='utf-8'))
if not (point.get('schema_version') == 'aim4_brglm2_v3' and point.get('bootstrap_eligibility') is True and point.get('formal_bootstrap_started') is False and point.get('primary', {}).get('status') == 'valid' and point.get('sensitivity', {}).get('status') == 'valid' and point.get('detectseparation_audit', {}).get('status') == 'PASS'):
    raise SystemExit('frozen point gate does not permit bootstrap')
if point.get('input_sha256') != digest(input_gz) or point.get('design_sha256') != digest(design) or point.get('prefit_manifest_sha256') != digest(prefit):
    raise SystemExit('point provenance does not bind current frozen prefit artifacts')
r_info = json.loads(subprocess.check_output(['Rscript', '--vanilla', '-e', 'RNGkind("L\'Ecuyer-CMRG"); set.seed(20260830); cat(jsonlite::toJSON(list(r=R.version.string,brglm2=as.character(packageVersion("brglm2")),detectseparation=as.character(packageVersion("detectseparation")),jsonlite=as.character(packageVersion("jsonlite")),rngkind=RNGkind(),seed=20260830), auto_unbox=TRUE))'], text=True))
payload = {
    'schema_version': 'aim4_brglm2_v3', 'evidence': 'associational/supportive', 'formal_point_rerun': False,
    'bootstrap_design': {'replicates': 2000, 'seed': 20260830, 'rng_kind': "L'Ecuyer-CMRG", 'stratifier': 'state_provider', 'cluster': 'cnes7', 'shards': 8, 'replicates_per_shard': 250, 'nested_threads': 1},
    'inputs': {p.name: digest(p) for p in (input_gz, design, prefit, point_path, environment_lock)},
    'environment_lock_sha256': digest(environment_lock),
    'code_sha256': {p.name: digest(p) for p in (r_bootstrap, finalizer, runner)},
    'r_environment': r_info, 'thread_contract': {'OMP_NUM_THREADS': 1, 'OPENBLAS_NUM_THREADS': 1, 'MKL_NUM_THREADS': 1, 'NUMEXPR_NUM_THREADS': 1, 'VECLIB_MAXIMUM_THREADS': 1, 'POLARS_MAX_THREADS': 1},
}
if manifest.exists():
    if json.loads(manifest.read_text(encoding='utf-8')) != payload:
        raise SystemExit(f'existing run manifest differs from current immutable frozen payload: {manifest}')
else:
    tmp = manifest.with_name('.' + manifest.name + '.tmp')
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    tmp.replace(manifest)
PY

mkdir -p "$REPLICATE_DIR" "$LOG_DIR"
export PROJECT_ROOT PYTHON_BIN R_LIBS OUTPUT_DIR REPLICATE_DIR LOG_DIR INPUT_GZ DESIGN_JSON PREFIT_MANIFEST POINT_JSON RUN_MANIFEST R_BOOTSTRAP FINALIZER
MASTER_SCRIPT="$PROJECT_ROOT/$LOG_DIR/master.sh"
MASTER_SCRIPT_TMP="$(mktemp "$PROJECT_ROOT/$LOG_DIR/.master.sh.tmp.XXXXXX")"
{
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
printf 'PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
printf 'R_LIBS=%q\n' "$R_LIBS"
printf 'OUTPUT_DIR=%q\n' "$OUTPUT_DIR"
printf 'REPLICATE_DIR=%q\n' "$REPLICATE_DIR"
printf 'LOG_DIR=%q\n' "$LOG_DIR"
printf 'INPUT_GZ=%q\n' "$INPUT_GZ"
printf 'DESIGN_JSON=%q\n' "$DESIGN_JSON"
printf 'PREFIT_MANIFEST=%q\n' "$PREFIT_MANIFEST"
printf 'POINT_JSON=%q\n' "$POINT_JSON"
printf 'RUN_MANIFEST=%q\n' "$RUN_MANIFEST"
printf 'R_BOOTSTRAP=%q\n' "$R_BOOTSTRAP"
printf 'FINALIZER=%q\n' "$FINALIZER"
printf '%s\n' 'export PROJECT_ROOT PYTHON_BIN R_LIBS OUTPUT_DIR REPLICATE_DIR LOG_DIR INPUT_GZ DESIGN_JSON PREFIT_MANIFEST POINT_JSON RUN_MANIFEST R_BOOTSTRAP FINALIZER'
cat <<'MASTER_SCRIPT_BODY'
set -euo pipefail
cd "$PROJECT_ROOT"
export R_LIBS="$R_LIBS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 POLARS_MAX_THREADS=1
record_master_exit() {
  local rc=$?
  printf "%s\n" "$rc" > "$LOG_DIR/master.exitcode"
  trap - EXIT
  exit "$rc"
}
trap record_master_exit EXIT
mkdir -p "$REPLICATE_DIR" "$LOG_DIR"
declare -a starts=(1 251 501 751 1001 1251 1501 1751)
declare -a ends=(250 500 750 1000 1250 1500 1750 2000)
declare -a pids=()
for idx in "${!starts[@]}"; do
  label=$(printf "shard_%02d" "$((idx + 1))")
  Rscript --vanilla "$R_BOOTSTRAP" --bootstrap-only --input-gz "$INPUT_GZ" --design-json "$DESIGN_JSON" --prefit-manifest "$PREFIT_MANIFEST" --point-json "$POINT_JSON" --run-manifest "$RUN_MANIFEST" --output-dir "$OUTPUT_DIR" --replicate-start "${starts[idx]}" --replicate-end "${ends[idx]}" > "$LOG_DIR/${label}.log" 2>&1 &
  pids+=($!)
done
status=0
for idx in "${!pids[@]}"; do
  label=$(printf "shard_%02d" "$((idx + 1))")
  if wait "${pids[idx]}"; then printf "0\n" > "$LOG_DIR/${label}.exitcode"; else rc=$?; printf "%s\n" "$rc" > "$LOG_DIR/${label}.exitcode"; status=$rc; fi
done
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
"$PYTHON_BIN" - "$REPLICATE_DIR" <<"PY"
import json, pathlib, sys
folder = pathlib.Path(sys.argv[1]); ids=[]
for path in sorted(folder.glob("replicate_*.json")):
    ids.append(int(json.loads(path.read_text(encoding="utf-8"))["replicate_id"]))
if ids != list(range(1, 2001)):
    raise SystemExit("replicate IDs are not exact sorted 1..2000")
PY
"$PYTHON_BIN" "$FINALIZER" --replicate-dir "$REPLICATE_DIR" --point-estimate "$POINT_JSON" --prefit-manifest "$PREFIT_MANIFEST" --run-manifest "$RUN_MANIFEST" --output-dir "$OUTPUT_DIR" > "$LOG_DIR/finalizer.log" 2>&1
MASTER_SCRIPT_BODY
} > "$MASTER_SCRIPT_TMP"
chmod 700 "$MASTER_SCRIPT_TMP"
mv -f -- "$MASTER_SCRIPT_TMP" "$MASTER_SCRIPT"
tmux new-session -d -s "$RUN_SESSION" "$MASTER_SCRIPT"
tmux display-message -p -t "$RUN_SESSION" '#S:#P'
