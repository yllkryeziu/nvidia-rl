#!/bin/bash
# Evaluate base model + all checkpoints from one NeMo-RL checkpoint directory.
# - Auto-discovers base model and step list from checkpoint metadata.
# - Auto-builds a fixed-size local eval set (default: 128 samples).
# - Consolidates sharded checkpoints to HF format when needed.
# - Runs high-throughput eval on one 8xH100 node (DP across 8 GPUs).
# - Writes summary CSV with accuracy + generation length.
#
# Usage:
#   sbatch scripts/eval_all_checkpoints_128.sh \
#     --checkpoint-dir /fast/project/.../checkpoints/my-run
#
# Optional:
#   --results-root /fast/project/.../eval_results
#   --eval-samples 128
#   --dataset-path /fast/project/.../my_eval_data
#   --problem-key problem
#   --solution-key ground_truth_solution
#   --max-model-len 16384
#   --max-new-tokens 16384
#   --gpu-mem-util 0.90
#   --math-workers 64
#   --force-steps "10 20 30"
#   --skip-consolidate

#SBATCH --job-name=eval-all-ckpts-128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --account=hfmi_profound
#SBATCH --output=%x-%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/eval_all_checkpoints_128.sh --checkpoint-dir /path/to/checkpoints/run

Required:
  --checkpoint-dir PATH      Checkpoint run dir containing step_*/...

Optional:
  --results-root PATH        Output root (default: <repo>/eval_results)
  --base-model PATH          Override discovered base model path
  --dataset-path PATH        Override eval source dataset path (local/HF path)
  --dataset-split NAME       Split to load from dataset (default: config split or train)
  --problem-key NAME         Problem/question column (default: auto-detect)
  --solution-key NAME        Ground-truth answer column (default: auto-detect)
  --filter-column NAME       Filter column for source dataset (default: from train config)
  --filter-value VALUE       Filter value for source dataset (default: from train config)
  --eval-samples N           Number of eval samples (default: 128)
  --max-model-len N          vLLM max model len (default: 16384)
  --max-new-tokens N         Generation max new tokens (default: 16384)
  --gpu-mem-util FLOAT       vLLM gpu memory utilization (default: 0.90)
  --math-workers N           Math env workers (default: 64)
  --force-steps "..."        Space/comma list of steps to evaluate
  --skip-consolidate         Skip checkpoint consolidation step
  -h, --help                 Show this help
EOF
}

CHECKPOINT_DIR=""
RESULTS_ROOT=""
BASE_MODEL_OVERRIDE=""
DATASET_PATH_OVERRIDE=""
DATASET_SPLIT_OVERRIDE=""
PROBLEM_KEY_OVERRIDE=""
SOLUTION_KEY_OVERRIDE=""
FILTER_COLUMN_OVERRIDE=""
FILTER_VALUE_OVERRIDE=""
EVAL_SAMPLES=128
MAX_MODEL_LEN=16384
MAX_NEW_TOKENS=16384
GPU_MEM_UTIL=0.90
MATH_WORKERS=64
FORCE_STEPS=""
SKIP_CONSOLIDATE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --base-model) BASE_MODEL_OVERRIDE="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH_OVERRIDE="$2"; shift 2 ;;
        --dataset-split) DATASET_SPLIT_OVERRIDE="$2"; shift 2 ;;
        --problem-key) PROBLEM_KEY_OVERRIDE="$2"; shift 2 ;;
        --solution-key) SOLUTION_KEY_OVERRIDE="$2"; shift 2 ;;
        --filter-column) FILTER_COLUMN_OVERRIDE="$2"; shift 2 ;;
        --filter-value) FILTER_VALUE_OVERRIDE="$2"; shift 2 ;;
        --eval-samples) EVAL_SAMPLES="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
        --math-workers) MATH_WORKERS="$2"; shift 2 ;;
        --force-steps) FORCE_STEPS="$2"; shift 2 ;;
        --skip-consolidate) SKIP_CONSOLIDATE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "[ERROR] --checkpoint-dir is required" >&2
    usage
    exit 2
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    NEMO_DIR="$(realpath "$SLURM_SUBMIT_DIR")"
elif [[ -n "${NEMO_DIR:-}" ]]; then
    NEMO_DIR="$(realpath "$NEMO_DIR")"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    NEMO_DIR="$(realpath "$SCRIPT_DIR/..")"
fi
cd "$NEMO_DIR"

if [[ ! "$CHECKPOINT_DIR" = /* ]]; then
    CHECKPOINT_DIR="$NEMO_DIR/$CHECKPOINT_DIR"
fi
CHECKPOINT_DIR="$(realpath "$CHECKPOINT_DIR")"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "[ERROR] checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 2
fi

if [[ -z "$RESULTS_ROOT" ]]; then
    RESULTS_ROOT="$NEMO_DIR/eval_results"
fi
if [[ ! "$RESULTS_ROOT" = /* ]]; then
    RESULTS_ROOT="$NEMO_DIR/$RESULTS_ROOT"
fi
RESULTS_ROOT="$(realpath -m "$RESULTS_ROOT")"

# Source Berlin environment variables if available (same pattern as ray_bare_berlin.sub).
if [[ -f "$NEMO_DIR/env_berlin.sh" ]]; then
    source "$NEMO_DIR/env_berlin.sh"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$NEMO_DIR/.venv}"
export HF_HOME="${HF_HOME:-/fast/project/HFMI_SynergyUnit/ylli/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export PYTHONPATH="$NEMO_DIR:${PYTHONPATH:-}"

# Keep Ray temp and object spilling off /tmp (which is often full on shared nodes).
# Use a short symlink path for UNIX socket limits while storing data on project storage.
RAY_TMP_LINK_ROOT="${RAY_TMP_LINK_ROOT:-/dev/shm}"
if [[ ! -d "$RAY_TMP_LINK_ROOT" ]]; then
    RAY_TMP_LINK_ROOT="/tmp"
fi
RAY_SPILL_ROOT="${RAY_SPILL_ROOT:-/fast/project/HFMI_SynergyUnit/ylli/r}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RAY_JOB_TAG="j${SLURM_JOB_ID}"
    RAY_TMP_LINK="$RAY_TMP_LINK_ROOT/r${SLURM_JOB_ID}"
else
    RAY_JOB_TAG="m$(date +%s)"
    RAY_TMP_LINK="$RAY_TMP_LINK_ROOT/r$(date +%s)"
fi
SPILL_JOB_DIR="$RAY_SPILL_ROOT/$RAY_JOB_TAG"
RAY_TMP_REAL="$SPILL_JOB_DIR/raytmp"
mkdir -p "$RAY_TMP_REAL" "$SPILL_JOB_DIR/spill"
if [[ -e "$RAY_TMP_LINK" && ! -L "$RAY_TMP_LINK" ]]; then
    rm -rf "$RAY_TMP_LINK"
fi
ln -sfn "$RAY_TMP_REAL" "$RAY_TMP_LINK"
export RAY_TMPDIR="$RAY_TMP_LINK"
export RAY_OBJECT_SPILLING_DIRECTORY="${RAY_OBJECT_SPILLING_DIRECTORY:-$SPILL_JOB_DIR/spill}"
export TMPDIR="$RAY_TMPDIR/tmp"
mkdir -p "$TMPDIR"

# Standalone eval should never inherit a stale remote Ray address.
unset RAY_ADDRESS || true
unset RAY_HEAD_IP || true
export NRL_FORCE_LOCAL_RAY=1

# Also clear stale local Ray attachment hints from previous jobs.
if [[ -x "$UV_PROJECT_ENVIRONMENT/bin/ray" ]]; then
    "$UV_PROJECT_ENVIRONMENT/bin/ray" stop --force >/dev/null 2>&1 || true
fi
rm -f /tmp/ray/ray_current_cluster >/dev/null 2>&1 || true
rm -f "$RAY_TMPDIR/ray_current_cluster" >/dev/null 2>&1 || true

echo "[INFO] Ray preflight:"
echo "       NRL_FORCE_LOCAL_RAY=$NRL_FORCE_LOCAL_RAY"
echo "       PYTHONPATH=$PYTHONPATH"
echo "       RAY_TMPDIR=$RAY_TMPDIR"
echo "       RAY_OBJECT_SPILLING_DIRECTORY=$RAY_OBJECT_SPILLING_DIRECTORY"
echo "       TMPDIR=$TMPDIR"
df -h /tmp "$RAY_TMPDIR" || true
uv run python - <<'PY'
import os
import nemo_rl.distributed.virtual_cluster as vc
print(f"[INFO] virtual_cluster.py: {vc.__file__}")
print(f"[INFO] NRL_FORCE_LOCAL_RAY seen by Python: {os.environ.get('NRL_FORCE_LOCAL_RAY')}")
print(f"[INFO] RAY_TMPDIR seen by Python: {os.environ.get('RAY_TMPDIR')}")
PY

RUN_TAG="eval128_$(date +%Y%m%d_%H%M%S)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_TAG="${RUN_TAG}_job${SLURM_JOB_ID}"
fi
RUN_ROOT="$RESULTS_ROOT/$(basename "$CHECKPOINT_DIR")/$RUN_TAG"
mkdir -p "$RUN_ROOT"

DISCOVERY_JSON="$RUN_ROOT/discovery.json"
discover_cmd=(
    uv run python scripts/discover_checkpoint_run.py
    --checkpoint-dir "$CHECKPOINT_DIR"
    --output-json "$DISCOVERY_JSON"
)
if [[ -n "$FORCE_STEPS" ]]; then
    discover_cmd+=(--force-steps "$FORCE_STEPS")
fi
"${discover_cmd[@]}" >/dev/null

eval "$(
    uv run python - "$DISCOVERY_JSON" <<'PY'
import json
import shlex
import sys

d = json.load(open(sys.argv[1]))
steps = " ".join(str(x) for x in d["steps_selected"])
print(f"DISC_BASE_MODEL={shlex.quote(d['base_model'])}")
print(f"DISC_CONFIG_SOURCE={shlex.quote(d.get('config_source') or '')}")
print(f"DISC_STEPS={shlex.quote(steps)}")
PY
)"

BASE_MODEL="${BASE_MODEL_OVERRIDE:-$DISC_BASE_MODEL}"
if [[ -z "$BASE_MODEL" ]]; then
    echo "[ERROR] Could not determine base model (set --base-model)." >&2
    exit 1
fi

# Avoid CPU deadlock: leave headroom for generation workers and Ray runtime.
RAY_CPUS_RAW="${SLURM_CPUS_PER_TASK:-64}"
if [[ "$RAY_CPUS_RAW" =~ ^[0-9]+$ ]]; then
    RAY_CPUS="$RAY_CPUS_RAW"
else
    RAY_CPUS=64
fi
CPU_RESERVE_FOR_GEN="${CPU_RESERVE_FOR_GEN:-16}"
if [[ ! "$CPU_RESERVE_FOR_GEN" =~ ^[0-9]+$ ]]; then
    CPU_RESERVE_FOR_GEN=16
fi
MAX_SAFE_MATH_WORKERS=$((RAY_CPUS - CPU_RESERVE_FOR_GEN))
if (( MAX_SAFE_MATH_WORKERS < 1 )); then
    MAX_SAFE_MATH_WORKERS=1
fi
if [[ "$MATH_WORKERS" =~ ^[0-9]+$ ]] && (( MATH_WORKERS > MAX_SAFE_MATH_WORKERS )); then
    echo "[WARN] Reducing math workers from $MATH_WORKERS to $MAX_SAFE_MATH_WORKERS to avoid CPU starvation."
    MATH_WORKERS="$MAX_SAFE_MATH_WORKERS"
fi

if [[ -n "$DISC_CONFIG_SOURCE" && -f "$DISC_CONFIG_SOURCE" ]]; then
    eval "$(
        uv run python - "$DISC_CONFIG_SOURCE" <<'PY'
import shlex
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
train = (cfg.get("data") or {}).get("train") or {}
if isinstance(train, list):
    train = train[0] if train else {}

items = {
    "CFG_DATA_PATH": "" if train.get("data_path") is None else str(train.get("data_path")),
    "CFG_DATA_SPLIT": "" if train.get("split") is None else str(train.get("split")),
    "CFG_FILTER_COLUMN": "" if train.get("filter_column") is None else str(train.get("filter_column")),
    "CFG_FILTER_VALUE": "" if train.get("filter_value") is None else str(train.get("filter_value")),
}
for k, v in items.items():
    print(f"{k}={shlex.quote(v)}")
PY
    )"
else
    CFG_DATA_PATH=""
    CFG_DATA_SPLIT=""
    CFG_FILTER_COLUMN=""
    CFG_FILTER_VALUE=""
fi

DATASET_PATH="${DATASET_PATH_OVERRIDE:-$CFG_DATA_PATH}"
DATASET_SPLIT="${DATASET_SPLIT_OVERRIDE:-${CFG_DATA_SPLIT:-train}}"
FILTER_COLUMN="${FILTER_COLUMN_OVERRIDE:-$CFG_FILTER_COLUMN}"
FILTER_VALUE="${FILTER_VALUE_OVERRIDE:-$CFG_FILTER_VALUE}"
PROBLEM_KEY="${PROBLEM_KEY_OVERRIDE:-}"
SOLUTION_KEY="${SOLUTION_KEY_OVERRIDE:-}"

if [[ -z "$DATASET_PATH" ]]; then
    echo "[ERROR] Could not determine source eval dataset path." >&2
    echo "        Provide --dataset-path explicitly." >&2
    exit 1
fi

EVAL_DATA_DIR="$RUN_ROOT/eval_data"
mkdir -p "$EVAL_DATA_DIR"
EVAL_DATA_JSON="$EVAL_DATA_DIR/eval_${EVAL_SAMPLES}.jsonl"

DATASET_PATH="$DATASET_PATH" \
DATASET_SPLIT="$DATASET_SPLIT" \
FILTER_COLUMN="$FILTER_COLUMN" \
FILTER_VALUE="$FILTER_VALUE" \
PROBLEM_KEY="$PROBLEM_KEY" \
SOLUTION_KEY="$SOLUTION_KEY" \
EVAL_SAMPLES="$EVAL_SAMPLES" \
EVAL_DATA_JSON="$EVAL_DATA_JSON" \
uv run python - <<'PY'
import json
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

dataset_path = os.environ["DATASET_PATH"]
split = os.environ.get("DATASET_SPLIT") or "train"
filter_column = os.environ.get("FILTER_COLUMN", "").strip()
filter_value_raw = os.environ.get("FILTER_VALUE", "")
problem_key_pref = os.environ.get("PROBLEM_KEY", "").strip()
solution_key_pref = os.environ.get("SOLUTION_KEY", "").strip()
eval_samples = int(os.environ["EVAL_SAMPLES"])
out_path = Path(os.environ["EVAL_DATA_JSON"])
out_path.parent.mkdir(parents=True, exist_ok=True)

def load_any(path: str, split_name: str):
    p = Path(path)
    if p.is_file():
        suffix = p.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(p), split=split_name)
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(p), split=split_name)
    try:
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            if split_name in ds:
                return ds[split_name]
            return ds[list(ds.keys())[0]]
        return ds
    except Exception:
        ds = load_dataset(path)
        if isinstance(ds, DatasetDict):
            if split_name in ds:
                return ds[split_name]
            return ds[list(ds.keys())[0]]
        return ds

def choose_key(columns, preferred, candidates, label):
    if preferred:
        if preferred in columns:
            return preferred
        print(f"[WARN] Requested {label} key '{preferred}' not found. Auto-detecting.")
    for key in candidates:
        if key in columns:
            return key
    raise RuntimeError(
        f"Could not detect {label} key. Available columns: {sorted(columns)}"
    )

ds = load_any(dataset_path, split)

if filter_column and filter_value_raw:
    try:
        filter_value = json.loads(filter_value_raw)
    except Exception:
        filter_value = filter_value_raw
    before = len(ds)
    ds = ds.filter(lambda x: x.get(filter_column) == filter_value)
    print(f"[INFO] Applied filter {filter_column}={filter_value!r}: {before} -> {len(ds)}")

columns = set(ds.column_names)
problem_key = choose_key(
    columns,
    problem_key_pref,
    ["problem", "question", "prompt", "input"],
    "problem",
)
solution_key = choose_key(
    columns,
    solution_key_pref,
    ["ground_truth_solution", "expected_answer", "answer", "solution"],
    "solution",
)

if len(ds) < eval_samples:
    raise RuntimeError(
        f"Dataset only has {len(ds)} rows after filtering, but eval_samples={eval_samples}"
    )

ds = ds.select(range(eval_samples))
ds = ds.map(
    lambda ex: {
        "problem": ex[problem_key],
        "expected_answer": ex[solution_key],
    },
    remove_columns=ds.column_names,
)
ds.to_json(str(out_path))
print(
    f"[INFO] Wrote eval dataset: {out_path} | rows={len(ds)} | keys=(problem, expected_answer)"
)
PY

if [[ "$SKIP_CONSOLIDATE" -eq 0 ]]; then
    if [[ -n "${DISC_STEPS// }" ]]; then
        set +e
        bash "$NEMO_DIR/scripts/ensure_consolidated_checkpoints.sh" \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            --base-model "$BASE_MODEL" \
            --steps "$DISC_STEPS" \
            --continue-on-error
        consolidate_rc=$?
        set -e
        if [[ "$consolidate_rc" -ne 0 ]]; then
            echo "[WARN] Some checkpoint consolidations failed; proceeding with available consolidated checkpoints."
        fi
    fi
else
    echo "[INFO] --skip-consolidate set; assuming consolidated checkpoints already exist."
fi

MODELS_MANIFEST="$RUN_ROOT/models_manifest.tsv"
{
    echo -e "label\tmodel_path"
    echo -e "base\t$BASE_MODEL"
    for step in $DISC_STEPS; do
        model_path="$CHECKPOINT_DIR/step_${step}/consolidated"
        if [[ -f "$model_path/config.json" ]]; then
            echo -e "step_${step}\t$model_path"
        else
            echo "[WARN] Skipping step_${step}: missing consolidated config at $model_path/config.json"
        fi
    done
} >"$MODELS_MANIFEST"

echo "[INFO] Models to evaluate:"
cat "$MODELS_MANIFEST"

FAILED_MODELS="$RUN_ROOT/failed_models.txt"
: >"$FAILED_MODELS"

while IFS=$'\t' read -r label model_path; do
    [[ "$label" == "label" ]] && continue
    save_path="$RUN_ROOT/models/$label/eval128_pass1"
    mkdir -p "$save_path"

    echo
    echo "================================================================"
    echo "Evaluating $label"
    echo "  model:   $model_path"
    echo "  output:  $save_path"
    echo "================================================================"

    set +e
    uv run python examples/run_eval.py \
        --config examples/configs/evals/eval.yaml \
        generation.model_name="$model_path" \
        tokenizer.name="$model_path" \
        eval.save_path="$save_path" \
        eval.metric=pass@k \
        eval.k_value=1 \
        eval.num_tests_per_prompt=1 \
        generation.temperature=0.0 \
        generation.top_p=1.0 \
        generation.top_k=-1 \
        generation.num_prompts_per_step=-1 \
        generation.max_new_tokens="$MAX_NEW_TOKENS" \
        generation.vllm_cfg.max_model_len="$MAX_MODEL_LEN" \
        generation.vllm_cfg.tensor_parallel_size=1 \
        generation.vllm_cfg.gpu_memory_utilization="$GPU_MEM_UTIL" \
        cluster.num_nodes=1 \
        cluster.gpus_per_node=8 \
        env.math.num_workers="$MATH_WORKERS" \
        data.dataset_name="$EVAL_DATA_JSON" \
        data.file_format=json \
        data.split=train \
        data.problem_key=problem \
        data.solution_key=expected_answer \
        data.max_input_seq_length=4096 \
        2>&1 | tee "$save_path/run.log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -ne 0 ]]; then
        echo "[ERROR] Eval failed for $label (rc=$rc)" | tee -a "$FAILED_MODELS"
        continue
    fi
done <"$MODELS_MANIFEST"

SUMMARY_CSV="$RUN_ROOT/summary_eval128.csv"
uv run python - "$RUN_ROOT/models" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

models_root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

for model_dir in sorted([p for p in models_root.iterdir() if p.is_dir()]):
    eval_json = model_dir / "eval128_pass1" / "evaluation_data.json"
    if not eval_json.exists():
        continue
    data = json.load(open(eval_json))["evaluation_data"]
    if not data:
        continue
    n = len(data)
    rewards = [float(x["reward"]) for x in data]
    tok = [int(x.get("completion_tokens", 0)) for x in data]
    rows.append(
        {
            "model": model_dir.name,
            "n": n,
            "accuracy": sum(rewards) / n,
            "avg_completion_tokens": sum(tok) / n,
            "total_completion_tokens": sum(tok),
        }
    )

rows.sort(key=lambda r: (0 if r["model"] == "base" else 1, r["model"]))

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "n",
            "accuracy",
            "avg_completion_tokens",
            "total_completion_tokens",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[INFO] Wrote summary: {summary_csv}")
for r in rows:
    print(
        f"{r['model']}: n={r['n']} accuracy={r['accuracy']:.4f} "
        f"avg_completion_tokens={r['avg_completion_tokens']:.1f}"
    )
PY

echo
echo "[INFO] Run root: $RUN_ROOT"
echo "[INFO] Summary CSV: $SUMMARY_CSV"
if [[ -s "$FAILED_MODELS" ]]; then
    echo "[WARN] Some models failed. See: $FAILED_MODELS"
    exit 1
fi
echo "[INFO] All model evaluations completed successfully."
