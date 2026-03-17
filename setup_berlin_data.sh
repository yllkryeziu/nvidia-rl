#!/bin/bash
# setup_berlin_data.sh - Download models and dataset for Berlin cluster
# Usage: bash setup_berlin_data.sh
set -euo pipefail

# Source env to get HF_HOME, YLLI_DATA_DIR, etc.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_berlin.sh"

# Override offline flags so we can actually download
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

DATASET_DIR="$YLLI_DATA_DIR/openthoughts114k-math-qwen3"

mkdir -p "$HF_HOME"

echo "=== Step 1: Download smallest model (Qwen3-1.7B) ==="
huggingface-cli download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

EXPECTED_1B7="$HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
if [[ ! -d "$EXPECTED_1B7" ]]; then
  echo "ERROR: Qwen3-1.7B not found at expected path: $EXPECTED_1B7"
  echo "Actual contents of $HF_HOME/hub/models--Qwen--Qwen3-1.7B/:"
  ls -la "$HF_HOME/hub/models--Qwen--Qwen3-1.7B/" 2>/dev/null || echo "(dir does not exist)"
  exit 1
fi
echo "OK: Qwen3-1.7B downloaded to $EXPECTED_1B7"

echo ""
echo "=== Step 2: Download remaining models ==="

echo "--- Qwen3-4B ---"
huggingface-cli download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c

echo "--- Qwen3-8B ---"
huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218

echo "--- Qwen3-14B ---"
huggingface-cli download Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18

echo ""
echo "=== Step 3: Verify all model paths ==="
ALL_OK=true
for model_path in \
  "$HF_HOME/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" \
  "$HF_HOME/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
  "$HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
  "$HF_HOME/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18" \
; do
  if [[ -d "$model_path" ]]; then
    echo "OK: $model_path"
  else
    echo "MISSING: $model_path"
    ALL_OK=false
  fi
done

if ! $ALL_OK; then
  echo "ERROR: Some models are missing. Check output above."
  exit 1
fi

echo ""
echo "=== Step 4: Download dataset ==="
if [[ -d "$DATASET_DIR" ]]; then
  echo "Dataset already exists at $DATASET_DIR, skipping download."
else
  echo "Downloading openthoughts114k-math-qwen3 dataset..."
  huggingface-cli download yllkryeziu/openthoughts114k-math-qwen3 --repo-type dataset --local-dir "$DATASET_DIR"
fi

if [[ -d "$DATASET_DIR" ]]; then
  echo "OK: Dataset at $DATASET_DIR"
  echo "Contents:"
  ls "$DATASET_DIR"
else
  echo "ERROR: Dataset not found at $DATASET_DIR"
  exit 1
fi

echo ""
echo "=== All downloads complete ==="
echo "Models: $HF_HOME/hub/"
echo "Dataset: $DATASET_DIR"
