#!/bin/bash
# Generic post-training eval suite runner (called inside an eval SLURM job).

set -euo pipefail

bool_is_true() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="$(realpath "$SCRIPT_DIR/..")"
cd "$NEMO_DIR"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$NEMO_DIR/.venv}"
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"

TRAIN_JOB_ID="${TRAIN_JOB_ID:-unknown}"
EVAL_JOB_ID="${EVAL_JOB_ID:-${SLURM_JOB_ID:-unknown}}"
NUM_CKPTS="${NUM_CKPTS:-3}"
RESULTS_ROOT="${RESULTS_ROOT:-$NEMO_DIR/eval_results}"
FORCE_EVAL_STEPS="${FORCE_EVAL_STEPS:-}"
ALLOW_PARTIAL_CHECKPOINT_SET="${ALLOW_PARTIAL_CHECKPOINT_SET:-1}"
AUTO_CONSOLIDATE="${AUTO_CONSOLIDATE:-1}"
ALLOW_UNRANKED_STEPS="${ALLOW_UNRANKED_STEPS:-0}"
RUN_LCB="${RUN_LCB:-1}"
RUN_AIME2025="${RUN_AIME2025:-1}"
RUN_MATH500="${RUN_MATH500:-1}"
MATH_AIME_GPUS="${MATH_AIME_GPUS:-4}"
ENV_MATH_NUM_WORKERS="${ENV_MATH_NUM_WORKERS:-32}"
EVAL_DEPENDENCY_KIND="${EVAL_DEPENDENCY_KIND:-afterany}"

CHECKPOINT_DIR_OVERRIDE="${CHECKPOINT_DIR_OVERRIDE:-}"
BASE_MODEL_OVERRIDE="${BASE_MODEL_OVERRIDE:-}"
METRIC_NAME_OVERRIDE="${METRIC_NAME_OVERRIDE:-val:accuracy}"
HIGHER_IS_BETTER_OVERRIDE="${HIGHER_IS_BETTER_OVERRIDE:-1}"
EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-}"
KEEP_TOP_K_OVERRIDE="${KEEP_TOP_K_OVERRIDE:-}"

if [[ -n "$CHECKPOINT_DIR_OVERRIDE" || -n "$BASE_MODEL_OVERRIDE" ]]; then
    if [[ -z "$CHECKPOINT_DIR_OVERRIDE" || -z "$BASE_MODEL_OVERRIDE" ]]; then
        echo "[ERROR] CHECKPOINT_DIR_OVERRIDE and BASE_MODEL_OVERRIDE must both be set in override mode." >&2
        exit 2
    fi
    if [[ ! "$CHECKPOINT_DIR_OVERRIDE" = /* ]]; then
        CHECKPOINT_DIR_OVERRIDE="$NEMO_DIR/$CHECKPOINT_DIR_OVERRIDE"
    fi
    RECIPE_CONFIG_ABS="${TRAIN_RECIPE_CONFIG:-override_mode}"
    RECIPE_CHECKPOINT_DIR="$(realpath "$CHECKPOINT_DIR_OVERRIDE")"
    RECIPE_BASE_MODEL="$BASE_MODEL_OVERRIDE"
    RECIPE_METRIC_NAME="$METRIC_NAME_OVERRIDE"
    RECIPE_HIGHER_IS_BETTER="$HIGHER_IS_BETTER_OVERRIDE"
    RECIPE_KEEP_TOP_K="$KEEP_TOP_K_OVERRIDE"
    if [[ -n "$EXPERIMENT_NAME_OVERRIDE" ]]; then
        EXPERIMENT_NAME="$EXPERIMENT_NAME_OVERRIDE"
    else
        EXPERIMENT_NAME="$(basename "$RECIPE_CHECKPOINT_DIR")"
    fi
else
    if [[ -z "${TRAIN_RECIPE_CONFIG:-}" ]]; then
        echo "[ERROR] TRAIN_RECIPE_CONFIG is required when override mode is not used." >&2
        exit 2
    fi
    if [[ ! -f "$TRAIN_RECIPE_CONFIG" ]]; then
        if [[ -f "$NEMO_DIR/$TRAIN_RECIPE_CONFIG" ]]; then
            TRAIN_RECIPE_CONFIG="$NEMO_DIR/$TRAIN_RECIPE_CONFIG"
        else
            echo "[ERROR] TRAIN_RECIPE_CONFIG not found: $TRAIN_RECIPE_CONFIG" >&2
            exit 2
        fi
    fi

    eval "$(
        UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" uv run python - "$TRAIN_RECIPE_CONFIG" "$NEMO_DIR" <<'PY'
import os
import shlex
import sys
from pathlib import Path

import yaml

recipe_path = Path(sys.argv[1]).resolve()
nemo_dir = Path(sys.argv[2]).resolve()
cfg = yaml.safe_load(recipe_path.read_text())

ckpt = cfg.get("checkpointing", {}).get("checkpoint_dir")
if not ckpt:
    raise SystemExit("Missing checkpointing.checkpoint_dir in recipe config")
ckpt_path = Path(ckpt)
if not ckpt_path.is_absolute():
    ckpt_path = (nemo_dir / ckpt_path).resolve()

base_model = cfg.get("policy", {}).get("model_name")
if not base_model:
    raise SystemExit("Missing policy.model_name in recipe config")

metric_name = cfg.get("checkpointing", {}).get("metric_name", "val:accuracy")
higher = bool(cfg.get("checkpointing", {}).get("higher_is_better", True))
keep_top_k = cfg.get("checkpointing", {}).get("keep_top_k", None)

items = {
    "RECIPE_CONFIG_ABS": str(recipe_path),
    "RECIPE_CHECKPOINT_DIR": str(ckpt_path),
    "RECIPE_BASE_MODEL": str(base_model),
    "RECIPE_METRIC_NAME": str(metric_name),
    "RECIPE_HIGHER_IS_BETTER": "1" if higher else "0",
    "RECIPE_KEEP_TOP_K": "" if keep_top_k is None else str(keep_top_k),
    "EXPERIMENT_NAME": ckpt_path.name,
}
for k, v in items.items():
    print(f"{k}={shlex.quote(v)}")
PY
    )"
fi

if bool_is_true "$RECIPE_HIGHER_IS_BETTER"; then
    RECIPE_HIGHER_IS_BETTER=1
else
    RECIPE_HIGHER_IS_BETTER=0
fi

RUN_TAG="${RUN_TAG:-}"
if [[ -z "$RUN_TAG" ]]; then
    if [[ "$TRAIN_JOB_ID" != "unknown" && "$EVAL_JOB_ID" != "unknown" ]]; then
        RUN_TAG="train_${TRAIN_JOB_ID}__eval_${EVAL_JOB_ID}"
    else
        RUN_TAG="manual_$(date +%Y%m%d_%H%M%S)"
    fi
fi

RUN_ROOT="${RUN_ROOT:-$RESULTS_ROOT/$EXPERIMENT_NAME/runs/$RUN_TAG}"
MODELS_ROOT="$RUN_ROOT/models"
SELECTION_DIR="$RUN_ROOT/selection"
SUMMARY_DIR="$RUN_ROOT/summary"
mkdir -p "$MODELS_ROOT" "$SELECTION_DIR" "$SUMMARY_DIR"

write_manifest() {
    local status="$1"
    local model_labels_csv="$2"
    local selected_steps_csv="$3"
    local evaluable_steps_csv="$4"
    uv run python - "$RUN_ROOT/manifest.json" <<PY
import json
from pathlib import Path

path = Path("$RUN_ROOT/manifest.json")
doc = {
    "suite_name": "posttrain_eval_suite",
    "run_tag": "$RUN_TAG",
    "created_at": "$(date -Iseconds)",
    "train_job_id": "$TRAIN_JOB_ID",
    "eval_job_id": "$EVAL_JOB_ID",
    "dependency_kind": "$EVAL_DEPENDENCY_KIND",
    "recipe_config": "$RECIPE_CONFIG_ABS",
    "experiment_name": "$EXPERIMENT_NAME",
    "checkpoint_base_path": "$RECIPE_CHECKPOINT_DIR",
    "base_model_path": "$RECIPE_BASE_MODEL",
    "checkpoint_metric_name": "$RECIPE_METRIC_NAME",
    "checkpoint_higher_is_better": bool(int("${RECIPE_HIGHER_IS_BETTER:-1}")),
    "checkpoint_keep_top_k": (int("$RECIPE_KEEP_TOP_K") if "$RECIPE_KEEP_TOP_K".isdigit() else None),
    "num_ckpts_requested": int("$NUM_CKPTS"),
    "selected_steps": [s for s in "$selected_steps_csv".split(",") if s],
    "evaluable_steps": [s for s in "$evaluable_steps_csv".split(",") if s],
    "models": [m for m in "$model_labels_csv".split(",") if m],
    "benchmarks": [b for b in [
        "aime2025_avg16" if "$RUN_AIME2025" in ("1","true","TRUE","yes","YES","on","ON") else None,
        "math500_avg16" if "$RUN_MATH500" in ("1","true","TRUE","yes","YES","on","ON") else None,
        "livecodebench_official" if "$RUN_LCB" in ("1","true","TRUE","yes","YES","on","ON") else None,
    ] if b is not None],
    "math_aime_mode": "pass@1_avg@16",
    "lcb_mode": "official_as_is",
    "status": "$status",
}
path.write_text(json.dumps(doc, indent=2))
PY
}

selection_json="$SELECTION_DIR/selected_checkpoints.json"
selector_args=(
    --checkpoint-dir "$RECIPE_CHECKPOINT_DIR"
    --metric-name "$RECIPE_METRIC_NAME"
    --higher-is-better "$RECIPE_HIGHER_IS_BETTER"
    --num-ckpts "$NUM_CKPTS"
    --allow-unranked-steps "$ALLOW_UNRANKED_STEPS"
    --output-json "$selection_json"
    --steps-only
)
if [[ -n "$FORCE_EVAL_STEPS" ]]; then
    selector_args+=(--force-steps "$FORCE_EVAL_STEPS")
fi

mapfile -t SELECTED_STEPS < <(uv run python "$NEMO_DIR/scripts/select_best_checkpoints.py" "${selector_args[@]}")

printf "%s\n" "${SELECTED_STEPS[@]:-}" >"$SELECTION_DIR/selected_steps.txt"

EVALABLE_STEPS=()
FAILED_STEP_SETUP=()
for step in "${SELECTED_STEPS[@]:-}"; do
    [[ -z "$step" ]] && continue
    consolidated_cfg="$RECIPE_CHECKPOINT_DIR/step_${step}/consolidated/config.json"
    if [[ -f "$consolidated_cfg" ]]; then
        EVALABLE_STEPS+=("$step")
        continue
    fi
    if bool_is_true "$AUTO_CONSOLIDATE"; then
        if bash "$NEMO_DIR/scripts/ensure_consolidated_checkpoints.sh" \
            --checkpoint-dir "$RECIPE_CHECKPOINT_DIR" \
            --base-model "$RECIPE_BASE_MODEL" \
            --steps "$step"; then
            if [[ -f "$consolidated_cfg" ]]; then
                EVALABLE_STEPS+=("$step")
            else
                FAILED_STEP_SETUP+=("$step:no_consolidated_output")
            fi
        else
            FAILED_STEP_SETUP+=("$step:consolidation_failed")
        fi
    else
        FAILED_STEP_SETUP+=("$step:missing_consolidated")
    fi
done

printf "%s\n" "${EVALABLE_STEPS[@]:-}" >"$SELECTION_DIR/evaluable_steps.txt"
printf "%s\n" "${FAILED_STEP_SETUP[@]:-}" >"$SELECTION_DIR/failed_step_setup.txt"

if [[ "${#EVALABLE_STEPS[@]}" -lt "${#SELECTED_STEPS[@]}" ]] && ! bool_is_true "$ALLOW_PARTIAL_CHECKPOINT_SET"; then
    echo "[ERROR] Some selected checkpoints are not evaluable and ALLOW_PARTIAL_CHECKPOINT_SET=0." >&2
    write_manifest "failed" "base" "$(IFS=,; echo "${SELECTED_STEPS[*]}")" "$(IFS=,; echo "${EVALABLE_STEPS[*]}")"
    exit 1
fi

MODEL_LABELS=("base")
for s in "${EVALABLE_STEPS[@]:-}"; do
    [[ -n "$s" ]] && MODEL_LABELS+=("step_${s}")
done
write_manifest "pending" "$(IFS=,; echo "${MODEL_LABELS[*]}")" "$(IFS=,; echo "${SELECTED_STEPS[*]}")" "$(IFS=,; echo "${EVALABLE_STEPS[*]}")"

suite_rc=0
suite_state="success"
if [[ ${#FAILED_STEP_SETUP[@]} -gt 0 || ${#EVALABLE_STEPS[@]} -lt ${#SELECTED_STEPS[@]} ]]; then
    suite_state="partial"
fi
if [[ ${#SELECTED_STEPS[@]} -eq 0 ]]; then
    suite_state="partial_no_checkpoints"
fi

run_math_eval() {
    local model_label="$1"
    local model_path="$2"
    local dataset="$3"
    local variant="${dataset}_avg16"
    local save_path="$MODELS_ROOT/${model_label}/${variant}"
    local log_path="$save_path/run.log"
    local cfg_path="examples/configs/evals/qwen3_1b7_${dataset}.yaml"
    mkdir -p "$save_path"

    echo "[INFO] Running ${dataset} avg@16 for ${model_label}"
    echo "       model:  $model_path"
    echo "       output: $save_path"

    set +e
    uv run python examples/run_eval.py \
        --config "$cfg_path" \
        generation.model_name="$model_path" \
        eval.save_path="$save_path" \
        eval.metric=pass@k \
        eval.k_value=1 \
        eval.num_tests_per_prompt=16 \
        cluster.gpus_per_node="$MATH_AIME_GPUS" \
        env.math.num_workers="$ENV_MATH_NUM_WORKERS" \
        2>&1 | tee "$log_path"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -ne 0 ]]; then
        echo "[ERROR] ${dataset} eval failed for ${model_label} (rc=$rc)" >&2
        suite_rc=1
        suite_state="partial"
        return 1
    fi

    if [[ -f "$save_path/config.json" ]]; then
        if ! uv run python - "$save_path/config.json" "$dataset" <<'PY'
import json
import sys
cfg = json.load(open(sys.argv[1]))
dataset = sys.argv[2]
assert cfg.get("dataset_name") == dataset, cfg
assert cfg.get("metric") == "pass@k", cfg
assert cfg.get("k_value") == 1, cfg
assert cfg.get("num_tests_per_prompt") == 16, cfg
PY
        then
            echo "[ERROR] Saved config validation failed for $save_path" >&2
            suite_rc=1
            suite_state="partial"
            return 1
        fi
    else
        echo "[ERROR] Missing $save_path/config.json after eval" >&2
        suite_rc=1
        suite_state="partial"
        return 1
    fi
}

if bool_is_true "$RUN_AIME2025" || bool_is_true "$RUN_MATH500"; then
    for dataset in aime2025 math500; do
        if [[ "$dataset" == "aime2025" ]] && ! bool_is_true "$RUN_AIME2025"; then
            continue
        fi
        if [[ "$dataset" == "math500" ]] && ! bool_is_true "$RUN_MATH500"; then
            continue
        fi
        run_math_eval "base" "$RECIPE_BASE_MODEL" "$dataset" || true
        for step in "${EVALABLE_STEPS[@]:-}"; do
            [[ -z "$step" ]] && continue
            model_label="step_${step}"
            model_path="$RECIPE_CHECKPOINT_DIR/step_${step}/consolidated"
            if [[ ! -f "$model_path/config.json" ]]; then
                echo "[WARN] Skipping $model_label $dataset: missing consolidated config" >&2
                suite_state="partial"
                continue
            fi
            run_math_eval "$model_label" "$model_path" "$dataset" || true
        done
    done
fi

if bool_is_true "$RUN_LCB"; then
    LCB_STEPS_TOKENS=("base")
    for s in "${EVALABLE_STEPS[@]:-}"; do
        [[ -n "$s" ]] && LCB_STEPS_TOKENS+=("$s")
    done
    set +e
    BASE_MODEL="$RECIPE_BASE_MODEL" \
    CKPT_BASE="$RECIPE_CHECKPOINT_DIR" \
    STEPS="${LCB_STEPS_TOKENS[*]}" \
    RESULTS_BASE="$MODELS_ROOT" \
    EVAL_RUN_TAG="$RUN_TAG" \
    bash "$NEMO_DIR/scripts/eval_13313301_lcb.sh"
    lcb_rc=$?
    set -e
    if [[ "$lcb_rc" -ne 0 ]]; then
        echo "[ERROR] LiveCodeBench eval script failed (rc=$lcb_rc)" >&2
        suite_rc=1
        suite_state="partial"
    fi
fi

set +e
enabled_benchmarks=()
bool_is_true "$RUN_AIME2025" && enabled_benchmarks+=("aime2025_avg16")
bool_is_true "$RUN_MATH500" && enabled_benchmarks+=("math500_avg16")
bool_is_true "$RUN_LCB" && enabled_benchmarks+=("livecodebench")
benchmarks_csv="$(IFS=,; echo "${enabled_benchmarks[*]:-}")"
uv run python "$NEMO_DIR/scripts/summarize_eval_run.py" --run-root "$RUN_ROOT" --benchmarks "$benchmarks_csv"
summary_rc=$?
set -e
if [[ "$summary_rc" -ne 0 ]]; then
    echo "[ERROR] Summary generation failed (rc=$summary_rc)" >&2
    suite_rc=1
    suite_state="failed"
    mkdir -p "$SUMMARY_DIR"
    cat >"$SUMMARY_DIR/run_status.json" <<EOF
{"run_tag":"$RUN_TAG","overall_status":"failed","reason":"summary_failed","summary_rc":$summary_rc}
EOF
else
    if [[ -f "$SUMMARY_DIR/run_status.json" ]]; then
        overall_from_summary="$(uv run python - "$SUMMARY_DIR/run_status.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("overall_status", "unknown"))
PY
)"
        if [[ "$overall_from_summary" != "success" && "$suite_state" == "success" ]]; then
            suite_state="$overall_from_summary"
        fi
    fi
fi

if [[ "$suite_rc" -ne 0 && "$suite_state" == "success" ]]; then
    suite_state="failed"
fi

write_manifest "$suite_state" "$(IFS=,; echo "${MODEL_LABELS[*]}")" "$(IFS=,; echo "${SELECTED_STEPS[*]}")" "$(IFS=,; echo "${EVALABLE_STEPS[*]}")"

echo "[INFO] Post-train eval suite complete."
echo "[INFO] Run root: $RUN_ROOT"
echo "[INFO] Final status: $suite_state"

exit "$suite_rc"
