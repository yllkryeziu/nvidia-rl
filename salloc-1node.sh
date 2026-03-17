#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/env_berlin.sh"
# Always run uv/ray from this checkout's virtualenv.
export UV_PROJECT_ENVIRONMENT="$REPO_DIR/.venv"

RAY_BIN="$UV_PROJECT_ENVIRONMENT/bin/ray"
CONFIG_PATH="${DISTILL_CONFIG:-examples/configs/recipes/llm/berlin/distill_topk512_qwen3_1b7.yaml}"
MAX_STEPS="${MAX_STEPS:-2}"
RAY_PORT="${RAY_PORT:-6379}"

if [[ ! -x "$RAY_BIN" ]]; then
  echo "ERROR: ray binary not found at $RAY_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config file not found at $CONFIG_PATH" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
else
  DETECTED_GPUS=0
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-$DETECTED_GPUS}"
if [[ -z "$GPUS_PER_NODE" || "$GPUS_PER_NODE" -le 0 ]]; then
  echo "ERROR: Could not detect GPUs. Set GPUS_PER_NODE explicitly." >&2
  exit 1
fi

CPUS_PER_NODE="${CPUS_PER_NODE:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"

cleanup() {
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[INFO] Starting Ray head on $(hostname) (GPUs=$GPUS_PER_NODE, CPUs=$CPUS_PER_NODE)"
echo "[INFO] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
"$RAY_BIN" stop --force >/dev/null 2>&1 || true
"$RAY_BIN" start --head \
  --port="$RAY_PORT" \
  --num-gpus="$GPUS_PER_NODE" \
  --num-cpus="$CPUS_PER_NODE" \
  --resources="{\"worker_units\": $GPUS_PER_NODE, \"slurm_managed_ray_cluster\": 1}" \
  --disable-usage-stats

sleep 3

echo "[INFO] Launching one-node distillation with colocated generation"
uv run python examples/run_distillation.py \
  --config "$CONFIG_PATH" \
  cluster.num_nodes=1 \
  cluster.gpus_per_node="$GPUS_PER_NODE" \
  distillation.resource_isolation.enabled=false \
  policy.generation.colocated.enabled=true \
  policy.generation.colocated.resources.gpus_per_node=null \
  policy.generation.colocated.resources.num_nodes=null \
  distillation.max_num_steps="$MAX_STEPS" \
  distillation.val_at_start=false \
  distillation.val_at_end=false \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=false
