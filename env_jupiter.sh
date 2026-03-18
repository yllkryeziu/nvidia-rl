#!/bin/bash
# env_jupiter.sh - source this before running distillation on the Jupiter cluster
# Usage: source env_jupiter.sh

# Base project data directory
export YLLI_DATA_DIR="/e/project1/scifi/kryeziu1"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$YLLI_DATA_DIR/.cache}"

# HuggingFace
export HF_HOME="$YLLI_DATA_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# vLLM / compiler caches
export VLLM_CACHE_ROOT_BASE="${VLLM_CACHE_ROOT_BASE:-$XDG_CACHE_HOME}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/nv/ComputeCache}"

# Weights & Biases
export WANDB_MODE="${WANDB_MODE:-offline}"

# UV / Ray
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Ensure cache roots exist on project storage.
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" \
  "$VLLM_CACHE_ROOT_BASE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
