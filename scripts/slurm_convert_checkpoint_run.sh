#!/bin/bash
# SLURM wrapper for converting one checkpoint run to consolidated HF checkpoints.
#
# Required env:
#   CHECKPOINT_DIR
#   BASE_MODEL
#   FORCE_EVAL_STEPS   (space/comma separated)
#
# Optional env:
#   UV_PROJECT_ENVIRONMENT
#   HF_HOME

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster
#SBATCH --account=envcomp
#SBATCH --output=convert-checkpoint-run-%j.out

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

if [[ -z "${CHECKPOINT_DIR:-}" || -z "${BASE_MODEL:-}" || -z "${FORCE_EVAL_STEPS:-}" ]]; then
    echo "[ERROR] CHECKPOINT_DIR, BASE_MODEL, and FORCE_EVAL_STEPS are required." >&2
    exit 2
fi

module --force purge
module load Stages/2026
module load CUDA
if ! module load NCCL 2>/dev/null; then
    echo "[WARN] Could not load NCCL module; continuing without it."
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$NEMO_DIR/.venv}"
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[INFO] Converting checkpoints for run:"
echo "       CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "       BASE_MODEL=$BASE_MODEL"
echo "       STEPS=$FORCE_EVAL_STEPS"

bash "$NEMO_DIR/scripts/ensure_consolidated_checkpoints.sh" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --base-model "$BASE_MODEL" \
    --steps "$FORCE_EVAL_STEPS"

echo "[INFO] Conversion completed successfully."
