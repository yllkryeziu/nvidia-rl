#!/bin/bash
# Submit a fast smoke-test version of post-training evals on develbooster.
#
# This is intended to catch runtime/config bugs before launching long booster jobs.
# It reuses scripts/submit_eval_jobs_from_checkpoint_dir.sh but:
#   - runs conversion + eval jobs on develbooster
#   - defaults to one checkpoint step (first discovered)
#   - runs AIME/MATH with num_tests_per_prompt=1
#   - runs LCB in debug mode (first ~15 examples)

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/submit_eval_smoketest_develbooster.sh \
    --checkpoint-dir /p/scratch/envcomp/yll/checkpoints/<experiment> \
    [--results-root /p/scratch/envcomp/yll/eval_results_smoke] \
    [--logs-root /p/scratch/envcomp/yll/eval_results/slurm-logs] \
    [--account envcomp] \
    [--partition develbooster] \
    [--convert-time 00:45:00] \
    [--math-time 02:00:00] \
    [--aime-time 02:00:00] \
    [--lcb-time 02:00:00] \
    [--benchmarks math500,aime2025,lcb] \
    [--force-steps "50"] \
    [--all-steps] \
    [--math-aime-tests 1] \
    [--lcb-debug 1] \
    [--lcb-n 1] \
    [--dry-run]

Defaults are smoke-oriented and intentionally not production eval settings.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$NEMO_DIR"

CHECKPOINT_DIR=""
RESULTS_ROOT="/p/scratch/envcomp/yll/eval_results_smoke"
LOGS_ROOT="/p/scratch/envcomp/yll/eval_results/slurm-logs"
ACCOUNT="envcomp"
PARTITION="develbooster"
CONVERT_TIME="00:45:00"
MATH_TIME="02:00:00"
AIME_TIME="02:00:00"
LCB_TIME="02:00:00"
BENCHMARKS="math500,aime2025,lcb"
FORCE_STEPS=""
USE_ALL_STEPS=0
MATH_AIME_TESTS=1
LCB_DEBUG=1
LCB_N=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --logs-root) LOGS_ROOT="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --convert-time) CONVERT_TIME="$2"; shift 2 ;;
        --math-time) MATH_TIME="$2"; shift 2 ;;
        --aime-time) AIME_TIME="$2"; shift 2 ;;
        --lcb-time) LCB_TIME="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --force-steps) FORCE_STEPS="$2"; shift 2 ;;
        --all-steps) USE_ALL_STEPS=1; shift ;;
        --math-aime-tests) MATH_AIME_TESTS="$2"; shift 2 ;;
        --lcb-debug) LCB_DEBUG="$2"; shift 2 ;;
        --lcb-n) LCB_N="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "[ERROR] --checkpoint-dir is required." >&2
    usage
    exit 2
fi

if [[ ! "$CHECKPOINT_DIR" = /* ]]; then
    CHECKPOINT_DIR="$NEMO_DIR/$CHECKPOINT_DIR"
fi
CHECKPOINT_DIR="$(realpath "$CHECKPOINT_DIR")"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "[ERROR] checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 2
fi

if ! [[ "$MATH_AIME_TESTS" =~ ^[0-9]+$ ]] || [[ "$MATH_AIME_TESTS" -lt 1 ]]; then
    echo "[ERROR] --math-aime-tests must be a positive integer (got: $MATH_AIME_TESTS)." >&2
    exit 2
fi
if ! [[ "$LCB_N" =~ ^[0-9]+$ ]] || [[ "$LCB_N" -lt 1 ]]; then
    echo "[ERROR] --lcb-n must be a positive integer (got: $LCB_N)." >&2
    exit 2
fi

DISCOVERY_JSON="$(mktemp)"
trap 'rm -f "$DISCOVERY_JSON"' EXIT

discover_cmd=(uv run python "$NEMO_DIR/scripts/discover_checkpoint_run.py" --checkpoint-dir "$CHECKPOINT_DIR" --output-json "$DISCOVERY_JSON")
if [[ -n "$FORCE_STEPS" ]]; then
    discover_cmd+=(--force-steps "$FORCE_STEPS")
fi
"${discover_cmd[@]}" >/dev/null

if [[ -z "$FORCE_STEPS" ]]; then
    if [[ "$USE_ALL_STEPS" -eq 1 ]]; then
        FORCE_STEPS="$(
            python3 - "$DISCOVERY_JSON" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
print(" ".join(str(x) for x in d.get("steps_selected", [])))
PY
        )"
    else
        FORCE_STEPS="$(
            python3 - "$DISCOVERY_JSON" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
steps = d.get("steps_selected", [])
print("" if not steps else str(steps[0]))
PY
        )"
    fi
fi

if [[ -z "$FORCE_STEPS" ]]; then
    echo "[ERROR] No steps available for smoke test in $CHECKPOINT_DIR." >&2
    exit 1
fi

echo "[INFO] Smoke-test submission config:"
echo "       checkpoint_dir=$CHECKPOINT_DIR"
echo "       partition=$PARTITION"
echo "       force_steps=$FORCE_STEPS"
echo "       math_aime_num_tests_per_prompt=$MATH_AIME_TESTS"
echo "       lcb_debug=$LCB_DEBUG"
echo "       lcb_n=$LCB_N"

submit_cmd=(
    bash "$NEMO_DIR/scripts/submit_eval_jobs_from_checkpoint_dir.sh"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --results-root "$RESULTS_ROOT"
    --logs-root "$LOGS_ROOT"
    --account "$ACCOUNT"
    --convert-partition "$PARTITION"
    --eval-partition "$PARTITION"
    --convert-time "$CONVERT_TIME"
    --math-time "$MATH_TIME"
    --aime-time "$AIME_TIME"
    --lcb-time "$LCB_TIME"
    --benchmarks "$BENCHMARKS"
    --force-steps "$FORCE_STEPS"
)
if [[ "$DRY_RUN" -eq 1 ]]; then
    submit_cmd+=(--dry-run)
fi

MATH_AIME_NUM_TESTS_PER_PROMPT="$MATH_AIME_TESTS" \
LCB_DEBUG="$LCB_DEBUG" \
LCB_N="$LCB_N" \
"${submit_cmd[@]}"
