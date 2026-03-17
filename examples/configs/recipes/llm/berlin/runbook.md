# Berlin Distillation Runbook

All commands are run from `nvidia-rl/` directory.

---

## v2 configs (1-node colocated, 150 steps, no in-training validation)

These v2 configs are set to:
- no in-training validation (`distillation.val_period=0`, `val_at_start=false`, `val_at_end=false`)
- checkpoint every `20` steps (`checkpointing.save_period=20`)
- keep all periodic checkpoints (`checkpointing.metric_name=null`, `keep_top_k=null`)

You can validate checkpoints separately after training.

### `distill_topk512_qwen3_1b7_1node_v2`
- Model: Qwen3-1.7B | TP=2 DP=4 | grad_accum=8 | lr=1e-5 | seq_packing=off
- Validation: disabled | checkpoints every 20 steps

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_1b7_1node_v2.yaml policy.generation.vllm_cfg.gpu_memory_utilization=0.65'
sbatch -J distill_topk512_qwen3_1b7_1node_v2 -N1 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

---

### `distill_topk512_qwen3_4b_1node_v2`
- Model: Qwen3-4B | TP=4 DP=2 | grad_accum=16 | lr=1e-5 | seq_packing=on
- Validation: disabled | checkpoints every 20 steps

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_4b_1node_v2.yaml policy.generation.vllm_cfg.gpu_memory_utilization=0.65'
sbatch -J distill_topk512_qwen3_4b_1node_v2 -N1 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

---

### `distill_topk512_qwen3_8b_1node_v2`
- Model: Qwen3-8B | TP=4 DP=2 | grad_accum=16 | lr=4e-6 | seq_packing=on | fresh training
- Validation: disabled | checkpoints every 20 steps

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_8b_1node_v2.yaml policy.generation.vllm_cfg.gpu_memory_utilization=0.60'
sbatch -J distill_topk512_qwen3_8b_1node_v2 -N1 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

---

## Original configs (1-node colocated, 1000 steps)

### `distill_topk512_qwen3_8b_1node`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_8b_1node.yaml'
sbatch -J distill_topk512_qwen3_8b_1node -N1 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

---

## Multi-node configs (3-node isolated, 1000 steps)

### `distill_topk512_qwen3_1b7`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_1b7.yaml'
sbatch -J distill_topk512_qwen3_1b7 -N3 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

### `distill_topk512_qwen3_4b`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_4b.yaml'
sbatch -J distill_topk512_qwen3_4b -N3 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

### `distill_topk512_qwen3_8b`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_8b.yaml'
sbatch -J distill_topk512_qwen3_8b -N3 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

### `distill_topk512_qwen3_14b`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_14b.yaml'
sbatch -J distill_topk512_qwen3_14b -N3 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```
