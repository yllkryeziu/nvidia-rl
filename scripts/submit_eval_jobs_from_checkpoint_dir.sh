#!/bin/bash
# Submit conversion + benchmark eval jobs from one checkpoint experiment directory.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/submit_eval_jobs_from_checkpoint_dir.sh \
    --checkpoint-dir /p/scratch/envcomp/yll/checkpoints/<experiment> \
    [--results-root /p/scratch/envcomp/yll/eval_results] \
    [--logs-root /p/scratch/envcomp/yll/eval_results/slurm-logs] \
    [--account envcomp] \
    [--convert-partition develbooster] \
    [--eval-partition booster] \
    [--convert-time 01:00:00] \
    [--math-time 08:00:00] \
    [--aime-time 08:00:00] \
    [--lcb-time 12:00:00] \
    [--benchmarks math500,aime2025,lcb] \
    [--force-steps "50 200 250"] \
    [--dry-run]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$NEMO_DIR"

CHECKPOINT_DIR=""
RESULTS_ROOT="/p/scratch/envcomp/yll/eval_results"
LOGS_ROOT="/p/scratch/envcomp/yll/eval_results/slurm-logs"
ACCOUNT="envcomp"
CONVERT_PARTITION="develbooster"
EVAL_PARTITION="booster"
CONVERT_TIME="01:00:00"
MATH_TIME="08:00:00"
AIME_TIME="08:00:00"
LCB_TIME="12:00:00"
BENCHMARKS="math500,aime2025,lcb"
FORCE_STEPS=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --logs-root) LOGS_ROOT="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --convert-partition) CONVERT_PARTITION="$2"; shift 2 ;;
        --eval-partition) EVAL_PARTITION="$2"; shift 2 ;;
        --convert-time) CONVERT_TIME="$2"; shift 2 ;;
        --math-time) MATH_TIME="$2"; shift 2 ;;
        --aime-time) AIME_TIME="$2"; shift 2 ;;
        --lcb-time) LCB_TIME="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --force-steps) FORCE_STEPS="$2"; shift 2 ;;
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

mkdir -p "$LOGS_ROOT" "$RESULTS_ROOT"

DISCOVERY_JSON="$(mktemp)"
trap 'rm -f "$DISCOVERY_JSON"' EXIT

discover_cmd=(uv run python scripts/discover_checkpoint_run.py --checkpoint-dir "$CHECKPOINT_DIR" --output-json "$DISCOVERY_JSON")
if [[ -n "$FORCE_STEPS" ]]; then
    discover_cmd+=(--force-steps "$FORCE_STEPS")
fi

if ! "${discover_cmd[@]}" >/dev/null; then
    echo "[ERROR] checkpoint discovery/preflight failed for $CHECKPOINT_DIR" >&2
    exit 1
fi

eval "$(
    uv run python - "$DISCOVERY_JSON" <<'PY'
import json
import shlex
import sys

d = json.load(open(sys.argv[1]))
steps = " ".join(str(x) for x in d["steps_selected"])
print(f"DISC_EXPERIMENT={shlex.quote(d['experiment_name'])}")
print(f"DISC_CHECKPOINT_DIR={shlex.quote(d['checkpoint_dir'])}")
print(f"DISC_BASE_MODEL={shlex.quote(d['base_model'])}")
print(f"DISC_METRIC_NAME={shlex.quote(d.get('metric_name','val:accuracy'))}")
print(f"DISC_HIGHER_IS_BETTER={'1' if d.get('higher_is_better', True) else '0'}")
print(f"DISC_STEPS={shlex.quote(steps)}")
print(f"DISC_CONFIG_SOURCE={shlex.quote(d.get('config_source') or '')}")
PY
)"

if [[ -z "${DISC_STEPS:-}" ]]; then
    echo "[ERROR] No selected steps discovered for $CHECKPOINT_DIR" >&2
    exit 1
fi

IFS=',' read -r -a BENCH_TOKENS <<<"$BENCHMARKS"
BENCH_LIST=()
for b in "${BENCH_TOKENS[@]}"; do
    tok="$(echo "$b" | xargs)"
    [[ -z "$tok" ]] && continue
    case "$tok" in
        math500|aime2025|lcb) BENCH_LIST+=("$tok") ;;
        *)
            echo "[ERROR] Invalid benchmark token: $tok (allowed: math500,aime2025,lcb)" >&2
            exit 2
            ;;
    esac
done
if [[ "${#BENCH_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] No valid benchmarks selected." >&2
    exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
manifest_dir="$RESULTS_ROOT/submission_manifests/$DISC_EXPERIMENT"
manifest_path="$manifest_dir/$timestamp.json"
mkdir -p "$manifest_dir"

cmd_to_string() {
    local out=""
    for arg in "$@"; do
        out+=$(printf " %q" "$arg")
    done
    echo "${out# }"
}

parse_job_id() {
    local out="$1"
    if [[ "$out" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

CONVERT_JOB_NAME="convert_${DISC_EXPERIMENT}"
CONVERT_LOG="$LOGS_ROOT/${CONVERT_JOB_NAME}_%j.out"
CONVERT_EXPORT="ALL,NEMO_DIR=$NEMO_DIR,CHECKPOINT_DIR=$DISC_CHECKPOINT_DIR,BASE_MODEL=$DISC_BASE_MODEL,UV_PROJECT_ENVIRONMENT=$NEMO_DIR/.venv"
convert_cmd=(
    sbatch
    -J "$CONVERT_JOB_NAME"
    -p "$CONVERT_PARTITION"
    --account "$ACCOUNT"
    --nodes 1
    --ntasks 1
    --cpus-per-task 8
    --time "$CONVERT_TIME"
    --output "$CONVERT_LOG"
    --export="$CONVERT_EXPORT"
    "$NEMO_DIR/scripts/slurm_convert_checkpoint_run.sh"
)
convert_cmd_str="$(cmd_to_string "${convert_cmd[@]}")"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] FORCE_EVAL_STEPS=$DISC_STEPS $convert_cmd_str"
    CONVERT_JOB_ID="DRYRUN_CONVERT"
else
    convert_out="$(FORCE_EVAL_STEPS="$DISC_STEPS" "${convert_cmd[@]}")"
    echo "$convert_out"
    CONVERT_JOB_ID="$(parse_job_id "$convert_out")" || {
        echo "[ERROR] Failed to parse conversion job id." >&2
        exit 1
    }
fi

MATH_JOB_ID=""
AIME_JOB_ID=""
LCB_JOB_ID=""
math_cmd_str=""
aime_cmd_str=""
lcb_cmd_str=""

for bench in "${BENCH_LIST[@]}"; do
    RUN_AIME=0
    RUN_MATH=0
    RUN_LCB=0
    JOB_TIME="$MATH_TIME"
    case "$bench" in
        math500)
            RUN_MATH=1
            JOB_TIME="$MATH_TIME"
            JOB_NAME="eval_math500_${DISC_EXPERIMENT}"
            ;;
        aime2025)
            RUN_AIME=1
            JOB_TIME="$AIME_TIME"
            JOB_NAME="eval_aime2025_${DISC_EXPERIMENT}"
            ;;
        lcb)
            RUN_LCB=1
            JOB_TIME="$LCB_TIME"
            JOB_NAME="eval_lcb_${DISC_EXPERIMENT}"
            ;;
    esac

    JOB_LOG="$LOGS_ROOT/${JOB_NAME}_%j.out"
    EVAL_EXPORT="ALL,NEMO_DIR=$NEMO_DIR,TRAIN_JOB_ID=$CONVERT_JOB_ID,CHECKPOINT_DIR_OVERRIDE=$DISC_CHECKPOINT_DIR,BASE_MODEL_OVERRIDE=$DISC_BASE_MODEL,METRIC_NAME_OVERRIDE=$DISC_METRIC_NAME,HIGHER_IS_BETTER_OVERRIDE=$DISC_HIGHER_IS_BETTER,EXPERIMENT_NAME_OVERRIDE=$DISC_EXPERIMENT,AUTO_CONSOLIDATE=0,ALLOW_PARTIAL_CHECKPOINT_SET=0,RUN_AIME2025=$RUN_AIME,RUN_MATH500=$RUN_MATH,RUN_LCB=$RUN_LCB,RESULTS_ROOT=$RESULTS_ROOT,EVAL_DEPENDENCY_KIND=afterok,UV_PROJECT_ENVIRONMENT=$NEMO_DIR/.venv,RUN_TAG=${bench}__from_convert_${CONVERT_JOB_ID}"

    eval_cmd=(
        sbatch
        -J "$JOB_NAME"
        -p "$EVAL_PARTITION"
        --account "$ACCOUNT"
        --nodes 1
        --ntasks 1
        --gres "gpu:4"
        --cpus-per-task 32
        --time "$JOB_TIME"
        --dependency "afterok:${CONVERT_JOB_ID}"
        --output "$JOB_LOG"
        --export="$EVAL_EXPORT"
        "$NEMO_DIR/scripts/slurm_posttrain_eval_suite.sh"
    )
    eval_cmd_str="$(cmd_to_string "${eval_cmd[@]}")"

    case "$bench" in
        math500) math_cmd_str="$eval_cmd_str" ;;
        aime2025) aime_cmd_str="$eval_cmd_str" ;;
        lcb) lcb_cmd_str="$eval_cmd_str" ;;
    esac

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY RUN] FORCE_EVAL_STEPS=$DISC_STEPS $eval_cmd_str"
        job_id="DRYRUN_${bench^^}"
    else
        out="$(FORCE_EVAL_STEPS="$DISC_STEPS" "${eval_cmd[@]}")"
        echo "$out"
        job_id="$(parse_job_id "$out")" || {
            echo "[ERROR] Failed to parse eval job id for $bench." >&2
            exit 1
        }
    fi

    case "$bench" in
        math500) MATH_JOB_ID="$job_id" ;;
        aime2025) AIME_JOB_ID="$job_id" ;;
        lcb) LCB_JOB_ID="$job_id" ;;
    esac
done

uv run python - \
  "$manifest_path" \
  "$timestamp" \
  "$DISC_CHECKPOINT_DIR" \
  "$DISC_EXPERIMENT" \
  "$DISC_CONFIG_SOURCE" \
  "$DISC_BASE_MODEL" \
  "$DISC_METRIC_NAME" \
  "$DISC_HIGHER_IS_BETTER" \
  "$DISC_STEPS" \
  "$ACCOUNT" \
  "$CONVERT_PARTITION" \
  "$EVAL_PARTITION" \
  "$RESULTS_ROOT" \
  "$LOGS_ROOT" \
  "$DRY_RUN" \
  "$CONVERT_JOB_ID" \
  "$MATH_JOB_ID" \
  "$AIME_JOB_ID" \
  "$LCB_JOB_ID" \
  "$convert_cmd_str" \
  "$math_cmd_str" \
  "$aime_cmd_str" \
  "$lcb_cmd_str" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path,
    timestamp,
    checkpoint_dir,
    experiment_name,
    config_source,
    base_model,
    metric_name,
    higher_is_better,
    steps,
    account,
    convert_partition,
    eval_partition,
    results_root,
    logs_root,
    dry_run,
    convert_job_id,
    math_job_id,
    aime_job_id,
    lcb_job_id,
    convert_cmd,
    math_cmd,
    aime_cmd,
    lcb_cmd,
) = sys.argv[1:]

doc = {
    "timestamp": timestamp,
    "checkpoint_dir": checkpoint_dir,
    "experiment_name": experiment_name,
    "config_source": config_source,
    "base_model": base_model,
    "metric_name": metric_name,
    "higher_is_better": bool(int(higher_is_better)),
    "steps": steps.split(),
    "account": account,
    "convert_partition": convert_partition,
    "eval_partition": eval_partition,
    "results_root": results_root,
    "logs_root": logs_root,
    "dry_run": bool(int(dry_run)),
    "jobs": {
        "convert": {"job_id": convert_job_id, "command": convert_cmd},
        "math500": {"job_id": math_job_id, "command": math_cmd},
        "aime2025": {"job_id": aime_job_id, "command": aime_cmd},
        "lcb": {"job_id": lcb_job_id, "command": lcb_cmd},
    },
}
path = Path(manifest_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(doc, indent=2))
PY

echo "[INFO] Experiment: $DISC_EXPERIMENT"
echo "[INFO] Checkpoint dir: $DISC_CHECKPOINT_DIR"
echo "[INFO] Steps: $DISC_STEPS"
echo "[INFO] Conversion job: $CONVERT_JOB_ID"
[[ -n "$MATH_JOB_ID" ]] && echo "[INFO] Math500 job:   $MATH_JOB_ID (afterok:$CONVERT_JOB_ID)"
[[ -n "$AIME_JOB_ID" ]] && echo "[INFO] AIME2025 job:  $AIME_JOB_ID (afterok:$CONVERT_JOB_ID)"
[[ -n "$LCB_JOB_ID" ]] && echo "[INFO] LCB job:       $LCB_JOB_ID (afterok:$CONVERT_JOB_ID)"
echo "[INFO] Submission manifest: $manifest_path"
