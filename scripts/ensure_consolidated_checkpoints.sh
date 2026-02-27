#!/bin/bash
# Consolidate selected NeMo-RL checkpoints into HF format for vLLM evaluation.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/ensure_consolidated_checkpoints.sh \
    --checkpoint-dir checkpoints/my-run \
    --base-model /path/to/base/model \
    --steps "100 200 300"

Options:
  --checkpoint-dir PATH   Root checkpoint dir containing step_*/...
  --base-model PATH       Base HF model path used for metadata/consolidation
  --steps "..."           Space/comma separated step numbers (e.g. "100 200")
  --backend NAME          Consolidation backend (default: gloo)
  --continue-on-error     Continue other steps if one fails
EOF
}

CHECKPOINT_DIR=""
BASE_MODEL=""
STEPS_RAW=""
BACKEND="${BACKEND:-gloo}"
CONTINUE_ON_ERROR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"; shift 2 ;;
        --base-model)
            BASE_MODEL="$2"; shift 2 ;;
        --steps)
            STEPS_RAW="$2"; shift 2 ;;
        --backend)
            BACKEND="$2"; shift 2 ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 2 ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" || -z "$BASE_MODEL" || -z "$STEPS_RAW" ]]; then
    echo "[ERROR] --checkpoint-dir, --base-model, and --steps are required." >&2
    usage
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONSOLIDATION_SCRIPT="$NEMO_DIR/3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py"

if [[ ! -f "$CONSOLIDATION_SCRIPT" ]]; then
    echo "[ERROR] Consolidation script not found: $CONSOLIDATION_SCRIPT" >&2
    exit 1
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$NEMO_DIR/.venv}"
export PYTHONPATH="$NEMO_DIR/3rdparty/Automodel-workspace/Automodel:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

STEPS_NORM="$(echo "$STEPS_RAW" | tr ',' ' ')"

overall_rc=0
for STEP in $STEPS_NORM; do
    if [[ "$STEP" =~ ^step_([0-9]+)$ ]]; then
        STEP="${BASH_REMATCH[1]}"
    fi
    if [[ ! "$STEP" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid step token: $STEP" >&2
        overall_rc=1
        [[ "$CONTINUE_ON_ERROR" -eq 1 ]] && continue || exit 1
    fi

    INPUT_DIR="${CHECKPOINT_DIR}/step_${STEP}/policy/weights/model"
    OUTPUT_DIR="${CHECKPOINT_DIR}/step_${STEP}/consolidated"
    TOKENIZER_DIR="${CHECKPOINT_DIR}/step_${STEP}/policy/tokenizer"

    echo "[INFO] Ensuring consolidated checkpoint for step_${STEP}"
    echo "       input:  $INPUT_DIR"
    echo "       output: $OUTPUT_DIR"

    if [[ -f "${OUTPUT_DIR}/config.json" ]]; then
        echo "[INFO] Already consolidated: $OUTPUT_DIR"
        continue
    fi
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "[ERROR] Missing sharded checkpoint dir: $INPUT_DIR" >&2
        overall_rc=1
        [[ "$CONTINUE_ON_ERROR" -eq 1 ]] && continue || exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    if ! uv run python "$CONSOLIDATION_SCRIPT" \
        --model-name "$BASE_MODEL" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --backend "$BACKEND"; then
        echo "[ERROR] Consolidation failed for step_${STEP}" >&2
        overall_rc=1
        [[ "$CONTINUE_ON_ERROR" -eq 1 ]] && continue || exit 1
    fi

    if [[ -d "$TOKENIZER_DIR" ]]; then
        echo "[INFO] Copying tokenizer from $TOKENIZER_DIR"
        cp -r "$TOKENIZER_DIR"/. "$OUTPUT_DIR/"
    else
        echo "[WARN] Tokenizer dir not found for step_${STEP}; attempting base tokenizer copy."
        cp "$BASE_MODEL"/tokenizer* "$OUTPUT_DIR/" 2>/dev/null || true
        cp "$BASE_MODEL"/special_tokens_map.json "$OUTPUT_DIR/" 2>/dev/null || true
    fi

    if [[ ! -f "${OUTPUT_DIR}/config.json" ]]; then
        echo "[ERROR] Consolidation output missing config.json: $OUTPUT_DIR" >&2
        overall_rc=1
        [[ "$CONTINUE_ON_ERROR" -eq 1 ]] && continue || exit 1
    fi
done

exit "$overall_rc"
