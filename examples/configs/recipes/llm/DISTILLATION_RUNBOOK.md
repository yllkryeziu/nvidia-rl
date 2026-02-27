# Scaled Distillation Runbook (Qwen3 1.7B / 4B / 8B / 14B)

This runbook covers launching and verifying the `distill_topk512_*` distillation configs on JUWELS Booster.

## Scope

Configs covered:

- `examples/configs/recipes/llm/distill_topk512_qwen3_1b7.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_4b.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_8b.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_14b.yaml`
- `examples/configs/recipes/llm/distill_topk512_qwen3_1b7_p64_g4.yaml` (64 prompts, 4 samples/prompt)

Common settings already baked into the `topk512` configs:

- `distillation.max_val_samples: 128`
- `data.train.split_validation_size: 128`
- `distillation.val_at_start: true`
- `checkpointing.save_period: 50`
- `distillation.val_period: 50`
- `distillation.topk_logits_k: 512`
- `loss_fn.zero_outside_topk: true`

## Cluster Assumptions

- JUWELS Booster
- `4` GPUs per node (A100 40GB)
- Slurm submission via `ray_bare.sub`

## Preflight Checklist

Run these checks before submitting:

1. Confirm you are in the `nemo-rl` repo root.
2. Confirm the config file exists.
3. Confirm model snapshot paths referenced in the config exist on disk.
4. Confirm dataset path exists (`/p/project1/envcomp/yll/openthoughts114k-math-qwen3`).
5. Confirm `ray_bare.sub` is present and matches your cluster launcher expectations.
6. Confirm requested node count matches the config `cluster.num_nodes`.

Optional quick parse check:

```bash
uv run python - <<'PY'
import yaml
for p in [
    "examples/configs/recipes/llm/distill_topk512_qwen3_1b7.yaml",
    "examples/configs/recipes/llm/distill_topk512_qwen3_4b.yaml",
    "examples/configs/recipes/llm/distill_topk512_qwen3_8b.yaml",
    "examples/configs/recipes/llm/distill_topk512_qwen3_14b.yaml",
    "examples/configs/recipes/llm/distill_topk512_qwen3_1b7_p64_g4.yaml",
]:
    yaml.safe_load(open(p))
    print("OK", p)
PY
```

## Config Summary

| Config | Nodes | Training | Generation | Teacher | Notes |
|---|---:|---|---|---|---|
| `distill_topk512_qwen3_1b7.yaml` | 4 | 1 node, TP=2, DP=2 | 2 nodes, vLLM TP=1 (8 engines) | 1 node, TP=2, DP=2 | baseline topk512 1.7B |
| `distill_topk512_qwen3_4b.yaml` | 7 | 4 nodes, TP=8, DP=2 | 2 nodes, vLLM TP=1 (8 engines) | 1 node, TP=2, DP=2 | sequence packing on |
| `distill_topk512_qwen3_8b.yaml` | 7 | 4 nodes, TP=8, DP=2 | 2 nodes, vLLM TP=2 (4 engines) | 1 node, TP=2, DP=2 | sequence packing on |
| `distill_topk512_qwen3_14b.yaml` | 11 | 8 nodes, TP=8, DP=4 | 2 nodes, vLLM TP=4 (2 engines) | 1 node, TP=4, DP=1 | sequence packing on |
| `distill_topk512_qwen3_1b7_p64_g4.yaml` | 7 | 2 nodes, TP=4, DP=2 | 4 nodes, vLLM TP=1 (16 engines) | 1 node, TP=2, DP=2 | 64 prompts x 4 samples/prompt |

## Launch Commands

### `distill_topk512_qwen3_1b7.yaml` (4 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_1b7.yaml'
sbatch -J distill_topk512_qwen3_1b7 -N4 -t 16:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### `distill_topk512_qwen3_4b.yaml` (7 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_4b.yaml'
sbatch -J distill_topk512_qwen3_4b -N7 -t 16:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### `distill_topk512_qwen3_8b.yaml` (7 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_8b.yaml'
sbatch -J distill_topk512_qwen3_8b -N7 -t 16:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### `distill_topk512_qwen3_14b.yaml` (11 nodes)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_14b.yaml'
sbatch -J distill_topk512_qwen3_14b -N11 -t 16:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### `distill_topk512_qwen3_1b7_p64_g4.yaml` (7 nodes)
```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distill_topk512_qwen3_1b7_p64_g4.yaml'
sbatch -J distill_topk512_qwen3_1b7_p64_g4 -N7 -t 16:00:00 -p booster --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

## Startup Verification (First Minutes)

Check logs for these conditions:

1. Ray cluster initializes with the expected node count.
2. Resource isolation role split matches the config (training + generation + teacher = cluster nodes).
3. Validation dataloader loads with `128` samples.
4. Initial validation runs at step `0` (`val_at_start: true`).
5. No startup `ValueError` for distillation step batch divisibility.

Expected divisibility checks that must hold:

- `policy.train_global_batch_size % student_DP == 0`
- `(distillation.num_prompts_per_step * distillation.num_generations_per_prompt) % student_DP == 0`

## Runtime Verification

Confirm these during training:

1. Validation runs every `50` steps.
2. Checkpoint saves every `50` steps.
3. `topk_logits_k=512` path is active (no fallback to a different top-k).
4. GPU memory stays stable after warmup (watch generation workers and teacher workers separately).

## OOM Guidance

What changed that affects memory:

- `max_val_samples`, `split_validation_size`, `save_period`, `val_at_start`: no peak memory impact
- Extra generation nodes: lowers per-engine load, usually safer
- `topk_logits_k: 512`: moderate increase vs lower top-k configs
- `1b7_p64_g4`: much higher throughput target; generation pressure is mitigated by 4 generation nodes and conservative vLLM memory utilization

First knobs to reduce if OOM happens:

1. `policy.generation_batch_size` (reduce first)
2. `policy.generation.vllm_cfg.gpu_memory_utilization` (reduce second)
3. `distillation.val_batch_size` (only if validation OOMs)

Training-side knobs should be changed only if needed:

1. `policy.train_global_batch_size` (must remain divisible by student DP)
2. `distillation.num_prompts_per_step` or `distillation.num_generations_per_prompt` (step batch must remain divisible by student DP)

## Quick Failure Triage

If you see resource isolation assertion failures:

- Check `distillation.resource_isolation.roles.*.num_nodes`
- Check `cluster.num_nodes`
- Check `distillation.resource_isolation.roles.*.gpus_per_node == cluster.gpus_per_node`

If you see distillation step batch divisibility failures:

- Check `student DP = (training_nodes * gpus_per_node) / policy.dtensor_cfg.tensor_parallel_size`
- Ensure `(num_prompts_per_step * num_generations_per_prompt)` is a multiple of student DP
- Ensure `policy.train_global_batch_size` is a multiple of student DP

If validation is unstable:

- Keep `distillation.val_batch_size` divisible by student DP
- Prefer `max_val_samples` divisible by `val_batch_size`
- Prefer `split_validation_size >= max_val_samples`

## Recommended Bring-Up Order

1. `distill_topk512_qwen3_1b7.yaml`
2. `distill_topk512_qwen3_4b.yaml`
3. `distill_topk512_qwen3_8b.yaml`
4. `distill_topk512_qwen3_14b.yaml`
5. `distill_topk512_qwen3_1b7_p64_g4.yaml` (throughput stress variant)
