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
    [--math-aime-num-tests-per-prompt 4] \
    [--lcb-n 4] \
    [--lcb-temperature 0.6] \
    [--lcb-top-p 0.95] \
    [--lcb-max-tokens 16384] \
    [--benchmarks math500,aime2025,lcb] \
    [--force-steps "50 200 250"] \
    [--always-convert] \
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
MATH_AIME_NUM_TESTS_PER_PROMPT="4"
LCB_N="4"
LCB_TEMPERATURE="0.6"
LCB_TOP_P="0.95"
LCB_MAX_TOKENS="16384"
BENCHMARKS="math500,aime2025,lcb"
FORCE_STEPS=""
ALWAYS_CONVERT=0
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
        --math-aime-num-tests-per-prompt) MATH_AIME_NUM_TESTS_PER_PROMPT="$2"; shift 2 ;;
        --lcb-n) LCB_N="$2"; shift 2 ;;
        --lcb-temperature) LCB_TEMPERATURE="$2"; shift 2 ;;
        --lcb-top-p) LCB_TOP_P="$2"; shift 2 ;;
        --lcb-max-tokens) LCB_MAX_TOKENS="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        --force-steps) FORCE_STEPS="$2"; shift 2 ;;
        --always-convert) ALWAYS_CONVERT=1; shift ;;
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

read -r -a STEP_LIST <<<"$DISC_STEPS"
MISSING_CONSOLIDATED_STEPS=()
for step in "${STEP_LIST[@]}"; do
    [[ -z "$step" ]] && continue
    consolidated_cfg="$DISC_CHECKPOINT_DIR/step_${step}/consolidated/config.json"
    if [[ ! -f "$consolidated_cfg" ]]; then
        MISSING_CONSOLIDATED_STEPS+=("$step")
    fi
done

NEED_CONVERT=0
if [[ "$ALWAYS_CONVERT" -eq 1 || "${#MISSING_CONSOLIDATED_STEPS[@]}" -gt 0 ]]; then
    NEED_CONVERT=1
fi

CONVERT_JOB_ID=""
convert_cmd_str=""
EVAL_DEPENDENCY_KIND="none"
RUN_TAG_SUFFIX="from_ready_consolidated"
TRAIN_ANCHOR_ID="preconverted"
dependency_note="no dependency (pre-converted checkpoints)"
dependency_args=()

if [[ "$NEED_CONVERT" -eq 1 ]]; then
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

    EVAL_DEPENDENCY_KIND="afterok"
    RUN_TAG_SUFFIX="from_convert_${CONVERT_JOB_ID}"
    TRAIN_ANCHOR_ID="$CONVERT_JOB_ID"
    dependency_note="afterok:$CONVERT_JOB_ID"
    dependency_args=(--dependency "afterok:${CONVERT_JOB_ID}")
else
    echo "[INFO] All selected steps already consolidated; skipping conversion job."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY RUN] Conversion skipped (all selected steps already consolidated)."
    fi
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
    EVAL_EXPORT="ALL,NEMO_DIR=$NEMO_DIR,TRAIN_JOB_ID=$TRAIN_ANCHOR_ID,CHECKPOINT_DIR_OVERRIDE=$DISC_CHECKPOINT_DIR,BASE_MODEL_OVERRIDE=$DISC_BASE_MODEL,METRIC_NAME_OVERRIDE=$DISC_METRIC_NAME,HIGHER_IS_BETTER_OVERRIDE=$DISC_HIGHER_IS_BETTER,EXPERIMENT_NAME_OVERRIDE=$DISC_EXPERIMENT,AUTO_CONSOLIDATE=0,ALLOW_PARTIAL_CHECKPOINT_SET=0,RUN_AIME2025=$RUN_AIME,RUN_MATH500=$RUN_MATH,RUN_LCB=$RUN_LCB,RESULTS_ROOT=$RESULTS_ROOT,EVAL_DEPENDENCY_KIND=$EVAL_DEPENDENCY_KIND,UV_PROJECT_ENVIRONMENT=$NEMO_DIR/.venv,RUN_TAG=${bench}__${RUN_TAG_SUFFIX},MATH_AIME_NUM_TESTS_PER_PROMPT=$MATH_AIME_NUM_TESTS_PER_PROMPT${EVAL_CONFIG_PREFIX:+,EVAL_CONFIG_PREFIX=$EVAL_CONFIG_PREFIX}"
    if [[ "$RUN_LCB" -eq 1 ]]; then
        # Pin the official LCB generation settings so stale submit-shell LCB_* env vars
        # cannot silently override the intended defaults via `sbatch --export=ALL,...`.
        EVAL_EXPORT+=",LCB_N=$LCB_N,LCB_TEMPERATURE=$LCB_TEMPERATURE,LCB_TOP_P=$LCB_TOP_P,LCB_MAX_TOKENS=$LCB_MAX_TOKENS"
    fi

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
        "${dependency_args[@]}"
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
  "$NEED_CONVERT" \
  "$ALWAYS_CONVERT" \
  "$(IFS=' '; echo "${MISSING_CONSOLIDATED_STEPS[*]}")" \
  "$EVAL_DEPENDENCY_KIND" \
  "$TRAIN_ANCHOR_ID" \
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
    need_convert,
    always_convert,
    missing_consolidated_steps,
    eval_dependency_kind,
    train_anchor_id,
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
    "conversion": {
        "required": bool(int(need_convert)),
        "always_convert": bool(int(always_convert)),
        "missing_consolidated_steps": missing_consolidated_steps.split(),
    },
    "eval_dependency_kind": eval_dependency_kind,
    "train_anchor_id": train_anchor_id,
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
if [[ "$NEED_CONVERT" -eq 1 ]]; then
    echo "[INFO] Conversion job: $CONVERT_JOB_ID"
else
    echo "[INFO] Conversion job: skipped (already consolidated)"
fi
[[ -n "$MATH_JOB_ID" ]] && echo "[INFO] Math500 job:   $MATH_JOB_ID ($dependency_note)"
[[ -n "$AIME_JOB_ID" ]] && echo "[INFO] AIME2025 job:  $AIME_JOB_ID ($dependency_note)"
[[ -n "$LCB_JOB_ID" ]] && echo "[INFO] LCB job:       $LCB_JOB_ID ($dependency_note)"
echo "[INFO] Submission manifest: $manifest_path"
