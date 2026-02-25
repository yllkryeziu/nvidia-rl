#!/bin/bash
# Official LiveCodeBench evaluation for job 13313301 checkpoints (base + checkpoints).
#
# This intentionally replaces the older NeMo-Gym "LCB accuracy test" flow. It runs the
# official LiveCodeBench runner (`python -m lcb_runner.runner.main`) with pinned settings.
#
# Best-practice defaults (as of 2026-02-25):
#   - scenario=codegeneration
#   - release_version=release_v6 (pinned for reproducibility)
#   - n=10, temperature=0.2, top_p=0.95, max_tokens=2000
#   - `--evaluate` to produce pass@1/pass@5 metrics
#   - `--continue_existing_with_eval` + `--use_cache` for resumable runs
#
# First run auto-clones LiveCodeBench and installs dependencies into `3rdparty/LiveCodeBench/.venv`.
# Requires outbound network access on first run (GitHub + HuggingFace datasets/models).
#
# Usage:
#   sbatch scripts/eval_13313301_lcb.sh
#   # smoke test (LiveCodeBench runner debug mode: first 15 examples)
#   LCB_DEBUG=1 sbatch scripts/eval_13313301_lcb.sh
#   # full/original checks (slower than default code_generation_lite)
#   LCB_NOT_FAST=1 sbatch scripts/eval_13313301_lcb.sh
#   # eval only some models
#   STEPS="base 100" sbatch scripts/eval_13313301_lcb.sh
#   # optional: custom run tag (otherwise defaults to job_<SLURM_JOB_ID>)
#   EVAL_RUN_TAG=rerun1 sbatch scripts/eval_13313301_lcb.sh
#   # override LCB model key or style selection if needed
#   LCB_MODEL_KEY='Qwen/Qwen3-235B-A22B' sbatch scripts/eval_13313301_lcb.sh
#   # style-based fallback selection (maps to a valid store key)
#   LCB_MODEL_STYLE=QwQ sbatch scripts/eval_13313301_lcb.sh
#
# Notes:
#   - This script writes each model's LCB workspace under:
#       eval_results/13313301/<model_label>/livecodebench/official_lcb/
#   - It is resumable. Re-running the same script/params continues generation/eval.
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=eval-13313301-lcb-%j.out

set -euo pipefail

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: $1"
        exit 1
    fi
}

download_file() {
    local url="$1"
    local out="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -o "$out" "$url"
        return
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -O "$out" "$url"
        return
    fi

    echo "[ERROR] Neither curl nor wget is available to download: $url"
    exit 1
}

download_lcb_archive() {
    local ref="$1"
    local archive_url=""
    local tmpdir=""
    local archive_fpath=""
    local extracted_dir=""

    if [[ "$ref" == "main" || "$ref" == "master" ]]; then
        archive_url="https://github.com/LiveCodeBench/LiveCodeBench/archive/refs/heads/${ref}.tar.gz"
    else
        # Works for tags and commit SHAs on GitHub. For branches, set LCB_ARCHIVE_URL explicitly if needed.
        archive_url="${LCB_ARCHIVE_URL:-https://github.com/LiveCodeBench/LiveCodeBench/archive/${ref}.tar.gz}"
    fi

    tmpdir="$(mktemp -d)"
    archive_fpath="$tmpdir/livecodebench.tar.gz"

    echo "[INFO] Downloading LiveCodeBench source archive: $archive_url"
    download_file "$archive_url" "$archive_fpath"

    mkdir -p "$(dirname "$LCB_DIR")"
    tar -xzf "$archive_fpath" -C "$tmpdir"

    extracted_dir="$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d ! -path "$tmpdir" | head -n1)"
    if [[ -z "$extracted_dir" || ! -d "$extracted_dir/lcb_runner" ]]; then
        echo "[ERROR] Failed to extract a valid LiveCodeBench archive."
        rm -rf "$tmpdir"
        exit 1
    fi

    rm -rf "$LCB_DIR"
    mv "$extracted_dir" "$LCB_DIR"
    rm -rf "$tmpdir"
}

setup_lcb_repo() {
    local has_git=0
    local repo_present=0

    if [[ -d "$LCB_DIR/lcb_runner" ]]; then
        repo_present=1
    fi

    if ! command -v git >/dev/null 2>&1; then
        if [[ -x /usr/bin/git ]]; then
            GIT_BIN="/usr/bin/git"
            has_git=1
        else
            if [[ "$repo_present" -eq 1 ]]; then
                echo "[WARN] git is unavailable; reusing existing LiveCodeBench source without update."
                LCB_COMMIT="unknown (git unavailable)"
                return
            fi
            echo "[WARN] git is unavailable; falling back to GitHub source archive download."
            download_lcb_archive "$LCB_GIT_REF"
            LCB_COMMIT="unknown (downloaded archive: ${LCB_GIT_REF})"
            echo "[INFO] LiveCodeBench repo: $LCB_DIR"
            echo "[INFO] LiveCodeBench commit: $LCB_COMMIT"
            return
        fi
    else
        GIT_BIN="$(command -v git)"
        has_git=1
    fi

    if [[ "$repo_present" -eq 0 ]]; then
        echo "[INFO] Cloning LiveCodeBench into: $LCB_DIR"
        mkdir -p "$(dirname "$LCB_DIR")"
        "$GIT_BIN" clone "$LCB_REPO_URL" "$LCB_DIR"
    elif [[ ! -d "$LCB_DIR/.git" ]]; then
        echo "[WARN] Existing LiveCodeBench source at $LCB_DIR has no .git metadata; reusing as-is."
        LCB_COMMIT="unknown (source tree without .git)"
        echo "[INFO] LiveCodeBench repo: $LCB_DIR"
        echo "[INFO] LiveCodeBench commit: $LCB_COMMIT"
        return
    fi

    if [[ "$has_git" -eq 1 && -n "$LCB_GIT_REF" ]]; then
        echo "[INFO] Checking out LiveCodeBench ref: $LCB_GIT_REF"
        "$GIT_BIN" -C "$LCB_DIR" fetch --tags origin >/dev/null 2>&1 || true
        "$GIT_BIN" -C "$LCB_DIR" checkout "$LCB_GIT_REF"
    fi

    LCB_COMMIT="$("$GIT_BIN" -C "$LCB_DIR" rev-parse HEAD)"
    echo "[INFO] LiveCodeBench repo: $LCB_DIR"
    echo "[INFO] LiveCodeBench commit: $LCB_COMMIT"
}

setup_lcb_env() {
    local install_needed=0

    if [[ ! -x "$LCB_VENV/bin/python" ]]; then
        install_needed=1
    fi

    if [[ "${LCB_SYNC:-0}" == "1" ]]; then
        install_needed=1
    fi

    if [[ "$install_needed" -eq 0 ]]; then
        echo "[INFO] Reusing LiveCodeBench environment: $LCB_VENV"
    else
        echo "[INFO] Setting up LiveCodeBench environment in: $LCB_VENV"
        mkdir -p "$(dirname "$LCB_VENV")"

        pushd "$LCB_DIR" >/dev/null
        # Prefer locked env if available; fall back to editable install if sync fails.
        if ! UV_PROJECT_ENVIRONMENT="$LCB_VENV" uv sync --frozen; then
            echo "[WARN] \`uv sync --frozen\` failed; falling back to editable install."
            UV_PROJECT_ENVIRONMENT="$LCB_VENV" uv venv --python 3.11 "$LCB_VENV"
            UV_PROJECT_ENVIRONMENT="$LCB_VENV" uv pip install -e .
            # LCB docs recommend datasets==2.18.0 in some flows; allow caller to force it if needed.
            if [[ -n "${LCB_DATASETS_PIN:-}" ]]; then
                UV_PROJECT_ENVIRONMENT="$LCB_VENV" uv pip install "datasets==${LCB_DATASETS_PIN}"
            fi
        fi
        popd >/dev/null
    fi

    LCB_PYTHON="$LCB_VENV/bin/python"

    if "$LCB_PYTHON" -c "import lcb_runner" >/dev/null 2>&1; then
        LCB_PYTHONPATH=""
    elif PYTHONPATH="$LCB_DIR${PYTHONPATH:+:$PYTHONPATH}" "$LCB_PYTHON" -c "import lcb_runner" >/dev/null 2>&1; then
        # Some LCB snapshots install deps but do not install the local package into the venv.
        LCB_PYTHONPATH="$LCB_DIR"
        echo "[WARN] lcb_runner is not installed in the venv; using PYTHONPATH fallback: $LCB_PYTHONPATH"
    else
        echo "[ERROR] LiveCodeBench environment setup completed but \`import lcb_runner\` still fails."
        echo "[ERROR] Repo: $LCB_DIR"
        echo "[ERROR] Venv: $LCB_VENV"
        exit 1
    fi
}

detect_model_key() {
    local py_script
    py_script='
import os
import sys
from lcb_runner.lm_styles import LanguageModelStore

store = LanguageModelStore

req_key = os.environ.get("LCB_MODEL_KEY", "").strip()
req_style = os.environ.get("LCB_MODEL_STYLE", "").strip()

def emit(key: str):
    model = store[key]
    print(f"{key}\t{model.model_style.value}")
    raise SystemExit(0)

def fail(msg: str):
    print(msg, file=sys.stderr)
    raise SystemExit(1)

def first_key_with_style(style: str, preferred_prefixes=()):
    keys = [k for k, v in store.items() if v.model_style.value == style]
    for prefix in preferred_prefixes:
        for k in keys:
            if k.startswith(prefix):
                return k
    return keys[0] if keys else None

# 1) Exact key override wins.
if req_key:
    if req_key in store:
        emit(req_key)
    fail(f"[ERROR] LCB_MODEL_KEY is not a valid LanguageModelStore key: {req_key}")

# 2) LCB_MODEL_STYLE can be either an exact key or a style selector.
if req_style:
    if req_style in store:
        emit(req_style)

    # Friendly aliases -> choose a valid store key using the matching style.
    alias = req_style
    if alias == "Qwen3":
        if "Qwen/Qwen3-235B-A22B" in store:
            emit("Qwen/Qwen3-235B-A22B")
        k = first_key_with_style("CodeQwenInstruct", preferred_prefixes=("Qwen/Qwen3", "Qwen/"))
        if k:
            emit(k)
        fail("[ERROR] Could not resolve LCB_MODEL_STYLE=Qwen3 to a valid LanguageModelStore key.")

    if alias in {"QwQ", "CodeQwenInstruct", "DeepSeekR1"}:
        k = first_key_with_style(alias, preferred_prefixes=("Qwen/", "deepseek-ai/"))
        if k:
            emit(k)
        fail(f"[ERROR] Could not resolve LCB_MODEL_STYLE={alias} to a valid LanguageModelStore key.")

    # Generic style name fallback.
    k = first_key_with_style(alias)
    if k:
        emit(k)

    fail(f"[ERROR] LCB_MODEL_STYLE is neither a valid key nor a known style selector: {req_style}")

# 3) Auto-detect best Qwen-compatible key for local Qwen3 checkpoints.
for candidate in [
    "Qwen/Qwen3-235B-A22B",     # best family match when present
    "Qwen/QwQ-32B",             # reasoning-style fallback
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]:
    if candidate in store:
        emit(candidate)

# Auto-search fallbacks by prefix/style.
for k, v in store.items():
    if k.startswith("Qwen/Qwen3"):
        emit(k)
for k, v in store.items():
    if k.startswith("Qwen/") and v.model_style.value in {"QwQ", "CodeQwenInstruct"}:
        emit(k)

fail("[ERROR] Could not auto-detect a Qwen-compatible LiveCodeBench model key. Set LCB_MODEL_KEY explicitly.")
'

    if [[ -n "$LCB_PYTHONPATH" ]]; then
        PYTHONPATH="$LCB_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" "$LCB_PYTHON" -c "$py_script"
    else
        "$LCB_PYTHON" -c "$py_script"
    fi
}

resolve_model_path() {
    local step="$1"
    local model_path=""
    local model_label=""

    if [[ "$step" == "base" ]]; then
        model_path="$BASE_MODEL"
        model_label="base"
        if [[ ! -f "$model_path/config.json" ]]; then
            echo "[ERROR] Base model path missing config.json: $model_path"
            exit 1
        fi
    else
        model_path="${CKPT_BASE}/step_${step}/consolidated"
        model_label="step_${step}"
        if [[ ! -f "$model_path/config.json" ]]; then
            echo "[ERROR] Consolidated checkpoint not found: $model_path"
            echo "[ERROR] Run scripts/convert_checkpoints.sh first."
            exit 1
        fi
    fi

    printf "%s\t%s\n" "$model_label" "$model_path"
}

parse_eval_metrics() {
    local eval_json="$1"
    "$LCB_PYTHON" - "$eval_json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

summary = None
if isinstance(data, list) and data and isinstance(data[0], dict):
    summary = data[0]
elif isinstance(data, dict):
    summary = data
else:
    summary = {}

pass1 = summary.get("pass@1")
pass5 = summary.get("pass@5")
pass10 = summary.get("pass@10")

def fmt(x):
    return "" if x is None else f"{x:.6f}" if isinstance(x, (int, float)) else str(x)

print("\t".join([fmt(pass1), fmt(pass5), fmt(pass10)]))
PY
}

run_lcb_eval() {
    local model_label="$1"
    local model_path="$2"

    local results_dir="$RESULTS_BASE/${model_label}/livecodebench/official_lcb"
    local run_dir="$results_dir/workspace"
    local log_fpath="$results_dir/lcb_runner.log"
    local summary_fpath="$results_dir/run_info.txt"
    local lcb_prompts_link="$run_dir/lcb_runner"

    mkdir -p "$run_dir"
    # LiveCodeBench prompt templates are loaded with cwd-relative paths like
    # `lcb_runner/prompts/...`, so each isolated workspace needs a link back
    # to the repo's lcb_runner tree.
    if [[ -e "$lcb_prompts_link" && ! -L "$lcb_prompts_link" ]]; then
        rm -rf "$lcb_prompts_link"
    fi
    if [[ ! -L "$lcb_prompts_link" ]]; then
        ln -s "$LCB_DIR/lcb_runner" "$lcb_prompts_link"
    fi

    {
        echo "model_label=$model_label"
        echo "model_path=$model_path"
        echo "lcb_repo=$LCB_DIR"
        echo "lcb_commit=$LCB_COMMIT"
        echo "lcb_model_key=$LCB_MODEL_KEY_RESOLVED"
        echo "lcb_model_style=$LCB_MODEL_STYLE_RESOLVED"
        echo "release_version=$LCB_RELEASE_VERSION"
        echo "n=$LCB_N"
        echo "temperature=$LCB_TEMPERATURE"
        echo "top_p=$LCB_TOP_P"
        echo "max_tokens=$LCB_MAX_TOKENS"
        echo "tensor_parallel_size=$LCB_TP"
        echo "num_process_evaluate=$LCB_NUM_PROCESS_EVALUATE"
        echo "timeout=$LCB_TIMEOUT"
        echo "not_fast=$LCB_NOT_FAST"
        echo "debug=$LCB_DEBUG"
        echo "start_date=${LCB_START_DATE:-}"
        echo "end_date=${LCB_END_DATE:-}"
    } >"$summary_fpath"

    echo ""
    echo "================================================================"
    echo " Official LCB Eval: $model_label"
    echo " Model:             $model_path"
    echo " LCB model key:     $LCB_MODEL_KEY_RESOLVED"
    echo " LCB style:         $LCB_MODEL_STYLE_RESOLVED"
    echo " Release:           $LCB_RELEASE_VERSION"
    echo " Results dir:       $results_dir"
    [[ "$LCB_NOT_FAST" == "1" ]] && echo " Mode:              full/original (--not_fast)"
    [[ "$LCB_DEBUG" == "1" ]] && echo " Mode:              debug (15 examples)"
    [[ -n "$LCB_START_DATE" ]] && echo " Start date:        $LCB_START_DATE"
    [[ -n "$LCB_END_DATE" ]] && echo " End date:          $LCB_END_DATE"
    echo "================================================================"

    local -a cmd
    cmd=(
        "$LCB_PYTHON" -m lcb_runner.runner.main
        --model "$LCB_MODEL_KEY_RESOLVED"
        --local_model_path "$model_path"
        --scenario codegeneration
        --evaluate
        --continue_existing_with_eval
        --use_cache
        --trust_remote_code
        --release_version "$LCB_RELEASE_VERSION"
        --n "$LCB_N"
        --temperature "$LCB_TEMPERATURE"
        --top_p "$LCB_TOP_P"
        --max_tokens "$LCB_MAX_TOKENS"
        --tensor_parallel_size "$LCB_TP"
        --num_process_evaluate "$LCB_NUM_PROCESS_EVALUATE"
        --timeout "$LCB_TIMEOUT"
    )

    if [[ "$LCB_ENABLE_PREFIX_CACHING" == "1" ]]; then
        cmd+=(--enable_prefix_caching)
    fi
    if [[ "$LCB_NOT_FAST" == "1" ]]; then
        cmd+=(--not_fast)
    fi
    if [[ "$LCB_DEBUG" == "1" ]]; then
        cmd+=(--debug)
    fi
    if [[ -n "$LCB_START_DATE" ]]; then
        cmd+=(--start_date "$LCB_START_DATE")
    fi
    if [[ -n "$LCB_END_DATE" ]]; then
        cmd+=(--end_date "$LCB_END_DATE")
    fi

    pushd "$run_dir" >/dev/null
    echo "[INFO] Running command:"
    printf '  %q' "${cmd[@]}"
    echo
    if [[ -n "$LCB_PYTHONPATH" ]]; then
        PYTHONPATH="$LCB_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}" 2>&1 | tee -a "$log_fpath"
    else
        "${cmd[@]}" 2>&1 | tee -a "$log_fpath"
    fi
    local lcb_rc=${PIPESTATUS[0]}
    popd >/dev/null

    if [[ "$lcb_rc" -ne 0 ]]; then
        echo "[ERROR] LiveCodeBench runner failed for $model_label (rc=$lcb_rc)"
        return "$lcb_rc"
    fi

    local eval_json=""
    eval_json="$(find "$run_dir/output" -type f -name '*_eval.json' ! -name '*_eval_all.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2- || true)"
    if [[ -z "$eval_json" || ! -f "$eval_json" ]]; then
        echo "[WARN] Could not locate *_eval.json under $run_dir/output"
        echo -e "${model_label}\t${model_path}\t\t\t\t" >>"$SUMMARY_TABLE"
        return 0
    fi

    local metric_line=""
    metric_line="$(parse_eval_metrics "$eval_json" || true)"
    local pass1="" pass5="" pass10=""
    IFS=$'\t' read -r pass1 pass5 pass10 <<<"$metric_line"

    {
        echo ""
        echo "latest_eval_json=$eval_json"
        echo "pass@1=${pass1:-}"
        echo "pass@5=${pass5:-}"
        echo "pass@10=${pass10:-}"
    } >>"$summary_fpath"

    echo "[INFO] $model_label metrics: pass@1=${pass1:-N/A} pass@5=${pass5:-N/A} pass@10=${pass10:-N/A}"
    echo -e "${model_label}\t${model_path}\t${pass1}\t${pass5}\t${pass10}\t${eval_json}" >>"$SUMMARY_TABLE"
}

NEMO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

LCB_REPO_URL="${LCB_REPO_URL:-https://github.com/LiveCodeBench/LiveCodeBench.git}"
LCB_DIR="${LCB_DIR:-$NEMO_DIR/3rdparty/LiveCodeBench}"
LCB_VENV="${LCB_VENV:-$LCB_DIR/.venv}"
LCB_GIT_REF="${LCB_GIT_REF:-main}"  # Pin to a commit hash for strict reproducibility.
LCB_COMMIT=""
LCB_PYTHON=""
LCB_PYTHONPATH=""

if [[ -n "${EVAL_RUN_TAG:-}" ]]; then
    RUN_TAG="$EVAL_RUN_TAG"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_TAG="job_${SLURM_JOB_ID}"
else
    RUN_TAG="manual_$(date +%Y%m%d_%H%M%S)"
fi
RESULTS_BASE="${RESULTS_BASE:-$NEMO_DIR/eval_results/13313301/$RUN_TAG}"

BASE_MODEL="${BASE_MODEL:-/p/project1/envcomp/yll/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
CKPT_BASE="${CKPT_BASE:-$NEMO_DIR/checkpoints/dev-topk-512-zero-true}"
STEPS="${STEPS:-base 100 200}"

# LiveCodeBench generation/eval defaults (official-codegen comparable).
LCB_RELEASE_VERSION="${LCB_RELEASE_VERSION:-release_v6}"
LCB_N="${LCB_N:-10}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-0.2}"
LCB_TOP_P="${LCB_TOP_P:-0.95}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-2000}"
LCB_TP="${LCB_TP:-4}"
LCB_NUM_PROCESS_EVALUATE="${LCB_NUM_PROCESS_EVALUATE:-12}"
LCB_TIMEOUT="${LCB_TIMEOUT:-6}"
LCB_ENABLE_PREFIX_CACHING="${LCB_ENABLE_PREFIX_CACHING:-1}"
LCB_NOT_FAST="${LCB_NOT_FAST:-0}"  # 0 => code_generation_lite (default/recommended fast path)
LCB_DEBUG="${LCB_DEBUG:-0}"        # 1 => first 15 examples only (smoke test)
LCB_START_DATE="${LCB_START_DATE:-}"
LCB_END_DATE="${LCB_END_DATE:-}"
LCB_MODEL_KEY_RESOLVED=""
LCB_MODEL_STYLE_RESOLVED=""

SUMMARY_TABLE="$RESULTS_BASE/livecodebench_official_summary.tsv"
mkdir -p "$RESULTS_BASE"
echo -e "model_label\tmodel_path\tpass@1\tpass@5\tpass@10\teval_json" >"$SUMMARY_TABLE"

module --force purge
module load Stages/2026
module load CUDA
if ! command -v git >/dev/null 2>&1; then
    module load Git 2>/dev/null || module load git 2>/dev/null || true
fi
if ! module load NCCL 2>/dev/null; then
    echo "[WARN] Could not load NCCL module; continuing without it."
fi

# TP>1 vLLM can trigger Triton/Torch Inductor runtime compilation, which needs a host C/C++ compiler.
if ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
    if module load GCC/14.3.0 2>/dev/null || module load GCC 2>/dev/null; then
        echo "[INFO] Loaded compiler module for Triton/Torch Inductor: GCC"
    else
        echo "[WARN] Could not load GCC module automatically."
    fi
fi
if [[ -z "${CC:-}" ]]; then
    if command -v gcc >/dev/null 2>&1; then
        export CC
        CC="$(command -v gcc)"
    elif command -v cc >/dev/null 2>&1; then
        export CC
        CC="$(command -v cc)"
    fi
fi
if [[ -z "${CXX:-}" ]]; then
    if command -v g++ >/dev/null 2>&1; then
        export CXX
        CXX="$(command -v g++)"
    elif command -v c++ >/dev/null 2>&1; then
        export CXX
        CXX="$(command -v c++)"
    fi
fi
if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
    echo "[ERROR] No usable C/C++ compiler found (CC='${CC:-}', CXX='${CXX:-}')."
    echo "[ERROR] vLLM TP mode may require a compiler for Triton/Torch Inductor runtime builds."
    echo "[ERROR] Try loading a compiler module (e.g., GCC/14.3.0) or set CC/CXX explicitly."
    exit 1
fi
echo "[INFO] Using compiler: CC=$CC CXX=$CXX"

need_cmd uv

# LCB needs online access on first run for repo + datasets + model fetches unless already cached.
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NEMO_DIR/.cache/uv}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export GIT_TERMINAL_PROMPT=0

setup_lcb_repo
setup_lcb_env
if [[ "${HF_HUB_ENABLE_HF_TRANSFER}" == "1" ]]; then
    if [[ -n "$LCB_PYTHONPATH" ]]; then
        if ! PYTHONPATH="$LCB_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" "$LCB_PYTHON" -c "import hf_transfer" >/dev/null 2>&1; then
            echo "[WARN] HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not importable in $LCB_VENV; disabling."
            export HF_HUB_ENABLE_HF_TRANSFER=0
        fi
    else
        if ! "$LCB_PYTHON" -c "import hf_transfer" >/dev/null 2>&1; then
            echo "[WARN] HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not importable in $LCB_VENV; disabling."
            export HF_HUB_ENABLE_HF_TRANSFER=0
        fi
    fi
fi
model_sel="$(detect_model_key)"
LCB_MODEL_KEY_RESOLVED="${model_sel%%$'\t'*}"
LCB_MODEL_STYLE_RESOLVED="${model_sel#*$'\t'}"
echo "[INFO] Using LiveCodeBench model key: $LCB_MODEL_KEY_RESOLVED"
echo "[INFO] Using LiveCodeBench model style: $LCB_MODEL_STYLE_RESOLVED"

overall_rc=0
for STEP in $STEPS; do
    model_info="$(resolve_model_path "$STEP")"
    model_label="${model_info%%$'\t'*}"
    model_path="${model_info#*$'\t'}"
    if ! run_lcb_eval "$model_label" "$model_path"; then
        overall_rc=1
    fi
done

echo ""
echo "================================================================"
echo "LCB official-run summary (tsv): $SUMMARY_TABLE"
cat "$SUMMARY_TABLE"
echo "Results root: $RESULTS_BASE"
echo "================================================================"

exit "$overall_rc"
