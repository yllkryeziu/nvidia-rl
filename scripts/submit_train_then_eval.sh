#!/bin/bash
# Submit a training job and a dependent post-train eval job (afterany).

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/submit_train_then_eval.sh \
    --train-script scripts/my_training_job.sh \
    --recipe-config examples/configs/recipes/llm/dev_topk_512_zero_true.yaml \
    [--train-export K=V,...] \
    [--eval-export K=V,...] \
    [--eval-script scripts/slurm_posttrain_eval_suite.sh] \
    [--dry-run]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$NEMO_DIR"

TRAIN_SCRIPT=""
RECIPE_CONFIG=""
TRAIN_EXPORT_EXTRA=""
EVAL_EXPORT_EXTRA=""
EVAL_SCRIPT="scripts/slurm_posttrain_eval_suite.sh"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-script) TRAIN_SCRIPT="$2"; shift 2 ;;
        --recipe-config) RECIPE_CONFIG="$2"; shift 2 ;;
        --train-export) TRAIN_EXPORT_EXTRA="$2"; shift 2 ;;
        --eval-export) EVAL_EXPORT_EXTRA="$2"; shift 2 ;;
        --eval-script) EVAL_SCRIPT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$TRAIN_SCRIPT" || -z "$RECIPE_CONFIG" ]]; then
    echo "[ERROR] --train-script and --recipe-config are required." >&2
    usage
    exit 2
fi

if [[ ! -f "$TRAIN_SCRIPT" && -f "$NEMO_DIR/$TRAIN_SCRIPT" ]]; then
    TRAIN_SCRIPT="$NEMO_DIR/$TRAIN_SCRIPT"
fi
if [[ ! -f "$EVAL_SCRIPT" && -f "$NEMO_DIR/$EVAL_SCRIPT" ]]; then
    EVAL_SCRIPT="$NEMO_DIR/$EVAL_SCRIPT"
fi
if [[ ! -f "$RECIPE_CONFIG" && -f "$NEMO_DIR/$RECIPE_CONFIG" ]]; then
    RECIPE_CONFIG="$NEMO_DIR/$RECIPE_CONFIG"
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "[ERROR] Training sbatch script not found: $TRAIN_SCRIPT" >&2
    exit 2
fi
if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "[ERROR] Eval sbatch script not found: $EVAL_SCRIPT" >&2
    exit 2
fi
if [[ ! -f "$RECIPE_CONFIG" ]]; then
    echo "[ERROR] Recipe config not found: $RECIPE_CONFIG" >&2
    exit 2
fi

train_export="ALL,TRAIN_RECIPE_CONFIG=$RECIPE_CONFIG"
if [[ -n "$TRAIN_EXPORT_EXTRA" ]]; then
    train_export="$train_export,$TRAIN_EXPORT_EXTRA"
fi

echo "[INFO] Training script: $TRAIN_SCRIPT"
echo "[INFO] Recipe config:   $RECIPE_CONFIG"
echo "[INFO] Eval script:     $EVAL_SCRIPT"
echo "[INFO] Train export:    $train_export"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY RUN] sbatch --export=$train_export $TRAIN_SCRIPT"
    echo "[DRY RUN] sbatch --dependency=afterany:<TRAIN_JOB_ID> --export=ALL,TRAIN_JOB_ID=<TRAIN_JOB_ID>,TRAIN_RECIPE_CONFIG=$RECIPE_CONFIG,EVAL_DEPENDENCY_KIND=afterany${EVAL_EXPORT_EXTRA:+,$EVAL_EXPORT_EXTRA} $EVAL_SCRIPT"
    exit 0
fi

train_submit_output="$(sbatch --export="$train_export" "$TRAIN_SCRIPT")"
echo "$train_submit_output"
if [[ ! "$train_submit_output" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
    echo "[ERROR] Failed to parse training job id from sbatch output." >&2
    exit 1
fi
train_job_id="${BASH_REMATCH[1]}"

eval_export="ALL,TRAIN_JOB_ID=$train_job_id,TRAIN_RECIPE_CONFIG=$RECIPE_CONFIG,EVAL_DEPENDENCY_KIND=afterany"
if [[ -n "$EVAL_EXPORT_EXTRA" ]]; then
    eval_export="$eval_export,$EVAL_EXPORT_EXTRA"
fi

eval_submit_output="$(sbatch --dependency="afterany:${train_job_id}" --export="$eval_export" "$EVAL_SCRIPT")"
echo "$eval_submit_output"
if [[ ! "$eval_submit_output" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
    echo "[ERROR] Failed to parse eval job id from sbatch output." >&2
    exit 1
fi
eval_job_id="${BASH_REMATCH[1]}"

echo "[INFO] Submitted training job: $train_job_id"
echo "[INFO] Submitted eval job:     $eval_job_id"
echo "[INFO] Dependency:             afterany:$train_job_id"
