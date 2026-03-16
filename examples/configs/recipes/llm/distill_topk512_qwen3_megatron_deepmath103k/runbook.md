# TopK512 Qwen3 Megatron DeepMath103k Distillation Runbook

This runbook is for launching the grouped Megatron distillation recipes in this folder.

## Configs in Scope

- `examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_1b7_megatron.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_4b_megatron.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_8b_megatron.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_14b_megatron.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_1b7_megatron_concise_baseline.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_1b7_megatron_improvement_op.yaml`

## Launch Commands

Run from repo root (`nemo-rl`) and submit with `ray_bare.sub`.

Runtime caches are forced under `/p/project1/envcomp/yll/.cache`. The launcher
defaults `NETWORK_IFACE` to `ib0`; override it at submit time if your allocation
needs a different interface, for example:

```bash
sbatch --export=ALL,NETWORK_IFACE=bond0,COMMAND="$COMMAND" ray_bare.sub
```

These recipes use the local `save_to_disk` mirror at
`/p/project1/envcomp/yll/deepmath-103k-qwen3`.

### 1.7B (7 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_1b7_megatron.yaml'
sbatch -J distill_topk512_qwen3_1b7_megatron_deepmath103k -N7 -t 24:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 4B (9 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_4b_megatron.yaml'
sbatch -J distill_topk512_qwen3_4b_megatron_deepmath103k -N9 -t 24:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 8B (9 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_8b_megatron.yaml'
sbatch -J distill_topk512_qwen3_8b_megatron_deepmath103k -N9 -t 24:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 14B (13 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_14b_megatron.yaml'
sbatch -J distill_topk512_qwen3_14b_megatron_deepmath103k -N13 -t 24:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 1.7B concise baseline (7 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_1b7_megatron_concise_baseline.yaml'
sbatch -J distill_topk512_qwen3_1b7_megatron_concise_baseline_deepmath103k -N7 -t 24:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 1.7B improvement op (7 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_megatron_deepmath103k/distill_topk512_qwen3_1b7_megatron_improvement_op.yaml'
sbatch -J distill_topk512_qwen3_1b7_megatron_improvement_op_deepmath103k -N7 -t 24:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

## Keep W&B synced from the login node

`ray_bare.sub` sets `WANDB_MODE=offline` on compute nodes, so metrics land under each
run's `logger.log_dir/exp_XXX/wandb/wandb/offline-run-*` directory. To keep the
remote W&B run updating while the job is still running, start a sync loop on a login
node and point it at the base `logger.log_dir`.

First make sure the login node session is authenticated once:

```bash
cd /p/project1/envcomp/yll/nemo-rl
uv run wandb login
```

Then start the background sync loop in `tmux`:

```bash
tmux new -s wandb-sync
cd /p/project1/envcomp/yll/nemo-rl
uv run python scripts/wandb_sync_loop.py \
  /p/scratch/envcomp/yll/logs/distill-topk512-qwen3-1b7-p64-g4-megatron_improvement_op-deepmath103k \
  --interval-minutes 15
```

Useful variants:

```bash
# Dry-run one scan pass to confirm it sees the run(s).
uv run python scripts/wandb_sync_loop.py \
  /p/scratch/envcomp/yll/logs/distill-topk512-qwen3-1b7-p64-g4-megatron_improvement_op-deepmath103k \
  --once \
  --dry-run

# Watch only a single experiment directory once you know the exp id.
uv run python scripts/wandb_sync_loop.py \
  /p/scratch/envcomp/yll/logs/distill-topk512-qwen3-1b7-p64-g4-megatron_improvement_op-deepmath103k/exp_001 \
  --interval-minutes 15
```

Leave that running in `tmux`; attach later with:

```bash
tmux attach -t wandb-sync
```

## Quick Checks

1. Confirm the config path exists before submitting.
2. Confirm the requested `-N` matches each config's `cluster.num_nodes`.
3. Tail the driver log after launch to verify role allocation (training/generation/teacher).
