#!/bin/bash
# Submit all pending eval jobs.
#
# What this covers:
#   8b  - new steps 200/225/250: convert + math500_avg4 + aime2025_avg4 + lcb
#   14b - new steps 50/75/100:   convert + math500_avg4 + aime2025_avg4 + lcb
#   8b  - lcb rerun for 75/100/125 (100+125 never had lcb; 75 rerun)
#   14b - lcb rerun for step 25
#   1b7 - lcb rerun for 50/250/400/450
#   1b7-p64-g4 - lcb rerun for step 50
#   4b  - lcb rerun for 50/200/250
#
# All math/aime evals use num_tests_per_prompt=4 (avg4) and max_model_len=16384.
# All lcb evals use n=4 (pass@4) and max_tokens=16384 (script defaults).
#
# Usage (from nemo-rl root):
#   bash scripts/submit_pending_evals.sh
#   bash scripts/submit_pending_evals.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$NEMO_DIR"

CKPT_ROOT="/p/scratch/envcomp/yll/checkpoints"
SUBMIT="scripts/submit_eval_jobs_from_checkpoint_dir.sh"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    echo "[INFO] Dry-run mode"
fi

# ---------------------------------------------------------------------------
# 8b — new steps (need conversion): math500 + aime2025 + lcb
# ---------------------------------------------------------------------------
echo ""
echo "=== 8b: new steps 200/225/250 (convert + math500_avg4 + aime2025_avg4 + lcb) ==="
MATH_AIME_NUM_TESTS_PER_PROMPT=4 bash "$SUBMIT" \
    --checkpoint-dir "$CKPT_ROOT/distill-topk512-qwen3-8b-1node" \
    --benchmarks math500,aime2025,lcb \
    --force-steps "200 225 250" \
    --eval-partition booster \
    --convert-partition develbooster \
    --math-time  02:00:00 \
    --aime-time  02:00:00 \
    --lcb-time   04:00:00 \
    $DRY_RUN_FLAG

# ---------------------------------------------------------------------------
# 14b — new steps (need conversion): math500 + aime2025 + lcb
# ---------------------------------------------------------------------------
echo ""
echo "=== 14b: new steps 50/75/100 (convert + math500_avg4 + aime2025_avg4 + lcb) ==="
MATH_AIME_NUM_TESTS_PER_PROMPT=4 bash "$SUBMIT" \
    --checkpoint-dir "$CKPT_ROOT/distill-topk512-qwen3-14b-2node" \
    --benchmarks math500,aime2025,lcb \
    --force-steps "50 75 100" \
    --eval-partition booster \
    --convert-partition develbooster \
    --math-time  04:00:00 \
    --aime-time  04:00:00 \
    --lcb-time   08:00:00 \
    $DRY_RUN_FLAG

# ---------------------------------------------------------------------------
# 8b — lcb for existing consolidated steps 75/100/125
#   (75 = rerun, 100+125 = first lcb run)
# ---------------------------------------------------------------------------
echo ""
echo "=== 8b: lcb for steps 75/100/125 (already consolidated) ==="
bash "$SUBMIT" \
    --checkpoint-dir "$CKPT_ROOT/distill-topk512-qwen3-8b-1node" \
    --benchmarks lcb \
    --force-steps "75 100 125" \
    --eval-partition booster \
    --lcb-time   04:00:00 \
    $DRY_RUN_FLAG

# ---------------------------------------------------------------------------
# 14b — lcb rerun for step 25 (already consolidated)
# ---------------------------------------------------------------------------
echo ""
echo "=== 14b: lcb rerun for step 25 (already consolidated) ==="
bash "$SUBMIT" \
    --checkpoint-dir "$CKPT_ROOT/distill-topk512-qwen3-14b-2node" \
    --benchmarks lcb \
    --force-steps "25" \
    --eval-partition booster \
    --lcb-time   08:00:00 \
    $DRY_RUN_FLAG

# ---------------------------------------------------------------------------
# 1b7 — lcb rerun for steps 50/250/400/450 (all consolidated)
# ---------------------------------------------------------------------------
echo ""
echo "=== 1b7: lcb rerun for steps 50/250/400/450 ==="
bash "$SUBMIT" \
    --checkpoint-dir "$CKPT_ROOT/distill-topk512-qwen3-1b7" \
    --benchmarks lcb \
    --force-steps "50 250 400 450" \
    --eval-partition booster \
    --lcb-time   04:00:00 \
    $DRY_RUN_FLAG

# ---------------------------------------------------------------------------
# 1b7-p64-g4 — lcb rerun for step 50 (already consolidated)
# ---------------------------------------------------------------------------
echo ""
echo "=== 1b7-p64-g4: lcb rerun for step 50 ==="
bash "$SUBMIT" \
    --checkpoint-dir "$CKPT_ROOT/distill-topk512-qwen3-1b7-p64-g4" \
    --benchmarks lcb \
    --force-steps "50" \
    --eval-partition booster \
    --lcb-time   04:00:00 \
    $DRY_RUN_FLAG

# ---------------------------------------------------------------------------
# 4b — lcb rerun for steps 50/200/250 (all consolidated)
# ---------------------------------------------------------------------------
echo ""
echo "=== 4b: lcb rerun for steps 50/200/250 ==="
bash "$SUBMIT" \
    --checkpoint-dir "$CKPT_ROOT/distill-topk512-qwen3-4b" \
    --benchmarks lcb \
    --force-steps "50 200 250" \
    --eval-partition booster \
    --lcb-time   04:00:00 \
    $DRY_RUN_FLAG

echo ""
echo "[INFO] All submissions done."
