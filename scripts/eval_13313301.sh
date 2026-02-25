#!/bin/bash
# Evaluate job 13313301 (dev-topk-512-zero-true) checkpoints on AIME 2025 and MATH500.
# Runs 6 evals sequentially: 3 models × 2 benchmarks.
#
# Prerequisites: run scripts/convert_checkpoints.sh first.
#
# Usage:
#   sbatch scripts/eval_13313301.sh
#   # or to override which steps to eval:
#   STEPS="base 100 200" sbatch scripts/eval_13313301.sh
#   # optional: custom run tag (otherwise defaults to job_<SLURM_JOB_ID>)
#   EVAL_RUN_TAG=rerun1 sbatch scripts/eval_13313301.sh
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=eval-13313301-%j.out

set -euo pipefail

NEMO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$NEMO_DIR"

module --force purge
module load Stages/2026
module load CUDA
if ! module load NCCL 2>/dev/null; then
    echo "[WARN] Could not load NCCL module; continuing without it."
fi

export UV_PROJECT_ENVIRONMENT="$NEMO_DIR/.venv"
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

BASE_MODEL="/p/project1/envcomp/yll/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CKPT_BASE="$NEMO_DIR/checkpoints/dev-topk-512-zero-true"
if [[ -n "${EVAL_RUN_TAG:-}" ]]; then
    RUN_TAG="$EVAL_RUN_TAG"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_TAG="job_${SLURM_JOB_ID}"
else
    RUN_TAG="manual_$(date +%Y%m%d_%H%M%S)"
fi
RESULTS_BASE="${RESULTS_BASE:-$NEMO_DIR/eval_results/13313301/$RUN_TAG}"
echo "[INFO] Results base: $RESULTS_BASE"

# Which steps to evaluate (override via STEPS env var)
STEPS="${STEPS:-base 100 200}"

for STEP in $STEPS; do
    if [[ "$STEP" == "base" ]]; then
        MODEL_PATH="$BASE_MODEL"
        MODEL_LABEL="base"
    else
        MODEL_PATH="${CKPT_BASE}/step_${STEP}/consolidated"
        MODEL_LABEL="step_${STEP}"
        if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
            echo "[ERROR] Consolidated checkpoint not found: $MODEL_PATH"
            echo "[ERROR] Run scripts/convert_checkpoints.sh first."
            exit 1
        fi
    fi

    for DATASET in aime2025 math500; do
        SAVE_PATH="${RESULTS_BASE}/${MODEL_LABEL}/${DATASET}"
        echo ""
        echo "================================================================"
        echo " Evaluating: $MODEL_LABEL on $DATASET"
        echo " Model:      $MODEL_PATH"
        echo " Results:    $SAVE_PATH"
        echo "================================================================"

        uv run python examples/run_eval.py \
            --config "examples/configs/evals/qwen3_1b7_${DATASET}.yaml" \
            generation.model_name="$MODEL_PATH" \
            eval.save_path="$SAVE_PATH"
    done
done

echo ""
echo "All evaluations complete. Results in: $RESULTS_BASE"
