#!/bin/bash
# Evaluate job 13313301 checkpoints (base, step_100, step_200) on MATH500 using avg@4
# (pass@k, k_value=1, 4 samples per problem). One node, 4 GPUs, DP=4 for max throughput.
#
# Usage:
#   sbatch scripts/eval_step100_math500_avg4.sh
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --account=envcomp
#SBATCH --output=eval-math500-avg4-%j.out

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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_TAG="job_${SLURM_JOB_ID}"
else
    RUN_TAG="manual_$(date +%Y%m%d_%H%M%S)"
fi
RESULTS_BASE="$NEMO_DIR/eval_results/13313301/${RUN_TAG}"
echo "[INFO] Results base: $RESULTS_BASE"

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

    SAVE_PATH="${RESULTS_BASE}/${MODEL_LABEL}/math500_avg4"
    echo ""
    echo "================================================================"
    echo " Evaluating: $MODEL_LABEL on math500 | avg@4 (pass@1, 4 samples)"
    echo " Model:      $MODEL_PATH"
    echo " Results:    $SAVE_PATH"
    echo "================================================================"

    uv run python examples/run_eval.py \
        --config "examples/configs/evals/qwen3_1b7_math500.yaml" \
        generation.model_name="$MODEL_PATH" \
        eval.save_path="$SAVE_PATH" \
        eval.metric="pass@k" \
        eval.num_tests_per_prompt=4 \
        eval.k_value=1 \
        generation.vllm_cfg.gpu_memory_utilization=0.95 \
        env.math.num_workers=32
done

echo ""
echo "All evaluations complete. Results in: $RESULTS_BASE"
