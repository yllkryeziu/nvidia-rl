#!/bin/bash
# Consolidate sharded nemo-rl safetensors checkpoints into standard HF format
# so they can be loaded by vLLM for evaluation.
#
# Usage:
#   sbatch scripts/convert_checkpoints.sh
#   # or with custom steps:
#   STEPS="100 200 300" sbatch scripts/convert_checkpoints.sh
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=convert-%j.out

set -euo pipefail

NEMO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$NEMO_DIR"

module --force purge
module load Stages/2026
module load CUDA

export UV_PROJECT_ENVIRONMENT="$NEMO_DIR/.venv"
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# nemo_automodel is a workspace extra not installed in .venv by default.
# Add it to PYTHONPATH so the consolidation script can import it directly.
export PYTHONPATH="$NEMO_DIR/3rdparty/Automodel-workspace/Automodel:${PYTHONPATH:-}"

BASE_MODEL="/p/project1/envcomp/yll/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CKPT_BASE="$NEMO_DIR/checkpoints/dev-topk-512-zero-true"
CONSOLIDATION_SCRIPT="$NEMO_DIR/3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py"

# Steps to consolidate (override via STEPS env var)
STEPS="${STEPS:-100 200}"

for STEP in $STEPS; do
    INPUT_DIR="${CKPT_BASE}/step_${STEP}/policy/weights/model"
    OUTPUT_DIR="${CKPT_BASE}/step_${STEP}/consolidated"

    echo "[INFO] Consolidating step_${STEP}: $INPUT_DIR -> $OUTPUT_DIR"

    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "[WARN] Input dir not found, skipping: $INPUT_DIR"
        continue
    fi

    if [[ -f "${OUTPUT_DIR}/config.json" ]]; then
        echo "[INFO] Already consolidated, skipping: $OUTPUT_DIR"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    uv run python "$CONSOLIDATION_SCRIPT" \
        --model-name "$BASE_MODEL" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --backend gloo

    # Copy tokenizer so vLLM can load the output dir as a complete model
    TOKENIZER_DIR="${CKPT_BASE}/step_${STEP}/policy/tokenizer"
    if [[ -d "$TOKENIZER_DIR" ]]; then
        echo "[INFO] Copying tokenizer from $TOKENIZER_DIR"
        cp -r "$TOKENIZER_DIR"/. "$OUTPUT_DIR/"
    else
        echo "[WARN] Tokenizer dir not found: $TOKENIZER_DIR — using base model tokenizer"
        cp "$BASE_MODEL"/tokenizer* "$OUTPUT_DIR/" 2>/dev/null || true
        cp "$BASE_MODEL"/special_tokens_map.json "$OUTPUT_DIR/" 2>/dev/null || true
    fi

    echo "[INFO] Done: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
done

echo "[INFO] All consolidations complete."
