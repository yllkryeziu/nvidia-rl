# Context Self-Distillation v2 Runbook (16 prompts/step, A100 40GB)

Run from repo root:

```bash
cd /p/project1/envcomp/yll/nemo-rl
```

Validation setup for all `context-self-distill-qwen3-*.yaml` recipes:
- Validation uses a deterministic holdout from OpenThoughts train data:
  - `data.train.split_validation_size: 50`
  - `data.train.seed: 42`
  - `data.train.filter_column: domain`, `data.train.filter_value: math`
- Validation runs every 50 steps:
  - `distillation.val_period: 50`
  - `distillation.max_val_samples: 50`
- Validation logs include rollout metrics and train-comparable distillation metrics:
  - `validation/accuracy`, `validation/avg_length`
  - plus keys like `validation/loss`, `validation/global_valid_toks`, `validation/context_distillation_*`

## Full Runs (W&B enabled)

```bash
# 1.7B — 3 nodes (1 train + 1 gen + 1 teacher), TP=2, DP=2
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-1p7b-3n4g.v2.yaml'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 4B — 4 nodes (2 train + 1 gen + 1 teacher), TP=4, DP=2
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-4b-3n4g.v2.yaml'
sbatch -N4 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 8B — 4 nodes (2 train + 1 gen + 1 teacher), TP=2, DP=4
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-8b-4n4g.v2.yaml'
sbatch -N4 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 14B — 6 nodes (4 train + 1 gen + 1 teacher), TP=4, DP=4
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-14b-6n4g.v2.yaml'
sbatch -N6 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 32B — 10 nodes (8 train + 1 gen + 1 teacher), TP=4, DP=8
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-32b-10n4g.v2.yaml'
sbatch -N10 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

## Smoke Tests (2 steps, no checkpointing, no W&B)

```bash
# 1.7B
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-1p7b-3n4g.v2.yaml distillation.max_num_steps=2 checkpointing.enabled=False logger.wandb_enabled=False'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 4B
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-4b-3n4g.v2.yaml distillation.max_num_steps=2 checkpointing.enabled=False logger.wandb_enabled=False'
sbatch -N4 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 8B
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-8b-4n4g.v2.yaml distillation.max_num_steps=2 checkpointing.enabled=False logger.wandb_enabled=False'
sbatch -N4 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 14B
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-14b-6n4g.v2.yaml distillation.max_num_steps=2 checkpointing.enabled=False logger.wandb_enabled=False'
sbatch -N6 --export=ALL,COMMAND="$COMMAND" ray_bare.sub

# 32B
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-32b-10n4g.v2.yaml distillation.max_num_steps=2 checkpointing.enabled=False logger.wandb_enabled=False'
sbatch -N10 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

## Override: 32 prompts/step (any model)

Add these three overrides to any command above:

```bash
# Example: 8B with 32 prompts/step
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-8b-4n4g.v2.yaml distillation.num_prompts_per_step=32 policy.train_global_batch_size=32 teacher.train_global_batch_size=32'
sbatch -N4 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```
