#!/bin/bash
# SLURM wrapper for the generic post-training eval suite.
#
# Required env:
#   TRAIN_RECIPE_CONFIG=/path/to/recipe.yaml
# Optional env:
#   TRAIN_JOB_ID=<jobid> (used for run tagging)
#   all vars consumed by scripts/run_posttrain_eval_suite.sh
#
# Usage:
#   sbatch --dependency=afterany:<train_job_id> \
#     --export=ALL,TRAIN_JOB_ID=<train_job_id>,TRAIN_RECIPE_CONFIG=examples/configs/recipes/llm/dev.yaml \
#     scripts/slurm_posttrain_eval_suite.sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=posttrain-eval-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    NEMO_DIR="$(realpath "$SLURM_SUBMIT_DIR")"
elif [[ -n "${NEMO_DIR:-}" ]]; then
    NEMO_DIR="$(realpath "$NEMO_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    NEMO_DIR="$(realpath "$SCRIPT_DIR/..")"
fi
cd "$NEMO_DIR"

module --force purge
module load Stages/2026
module load CUDA
if ! module load NCCL 2>/dev/null; then
    echo "[WARN] Could not load NCCL module; continuing without it."
fi
if ! command -v git >/dev/null 2>&1; then
    module load Git 2>/dev/null || module load git 2>/dev/null || true
fi
if ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
    module load GCC/14.3.0 2>/dev/null || module load GCC 2>/dev/null || true
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$NEMO_DIR/.venv}"
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export EVAL_JOB_ID="${EVAL_JOB_ID:-${SLURM_JOB_ID:-unknown}}"

bash "$NEMO_DIR/scripts/run_posttrain_eval_suite.sh"
