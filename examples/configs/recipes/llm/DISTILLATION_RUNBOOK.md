# Scaled Distillation Runbook (Qwen3 1.7B / 4B / 8B / 14B)

This runbook covers launching and verifying the `distill_topk512_*` distillation configs on JUWELS Booster and Berlin (H100) cluster.

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

1. Confirm you are in the `nvidia-rl` repo root.
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

---

## Berlin Cluster (H100)

### Cluster Assumptions

- Berlin cluster: `8` GPUs per node (H100 SXM5 80GB)
- Max `3` nodes (24 GPUs total)
- Partition: `standard`, Account: `hfmi_profound`
- Slurm submission via `ray_bare_berlin.sub`
- Source `env_berlin.sh` before submitting

### Berlin Configs

All configs live in `examples/configs/recipes/llm/berlin/`. Most use 3 nodes with 8 GPUs each, and one (`14b_2node`) is a 2-node split (train+teacher / generation) variant.

| Config | Nodes | Training | Generation | Teacher | Notes |
|---|---:|---|---|---|---|
| `berlin/distill_topk512_qwen3_1b7.yaml` | 3 | 1 node, TP=2, DP=4 | 1 node, vLLM TP=1 (8 engines) | 1 node, TP=2, DP=4 | grad_accum=8 |
| `berlin/distill_topk512_qwen3_4b.yaml` | 3 | 1 node, TP=4, DP=2 | 1 node, vLLM TP=1 (8 engines) | 1 node, TP=2, DP=4 | TP reduced from 8 |
| `berlin/distill_topk512_qwen3_8b.yaml` | 3 | 1 node, TP=4, DP=2 | 1 node, vLLM TP=1 (8 engines) | 1 node, TP=2, DP=4 | vLLM TP=1 (8B fits H100) |
| `berlin/distill_topk512_qwen3_14b.yaml` | 3 | 1 node, TP=8, DP=1 | 1 node, vLLM TP=2 (4 engines) | 1 node, TP=4, DP=2 | grad_accum=32, DP=1 |
| `berlin/distill_topk512_qwen3_1b7_p64_g4.yaml` | 3 | 1 node, TP=2, DP=4 | 1 node, vLLM TP=1 (8 engines) | 1 node, TP=2, DP=4 | 8 engines (vs 16 on JUWELS) |
| `berlin/distill_topk512_qwen3_14b_2node.yaml` | 2 | 1 node, TP=8, DP=1 | 1 node, vLLM TP=2 (4 engines), non-colocated | train node, TP=4, DP=2 | full 20k token budgets (20480) |

### Berlin Preflight Checklist

1. Source the environment: `source env_berlin.sh`
2. Pin UV to this repo venv (avoids stale path from old checkout): `export UV_PROJECT_ENVIRONMENT="$PWD/.venv"`
3. Confirm model snapshots exist under `/fast/project/HFMI_SynergyUnit/ylli/.cache/huggingface/hub/`
4. Confirm dataset exists at `/fast/project/HFMI_SynergyUnit/ylli/openthoughts114k-math-qwen3`
5. Confirm `ray_bare_berlin.sub` is present

Quick parse check:

```bash
uv run python - <<'PY'
import yaml
for p in [
    "examples/configs/recipes/llm/berlin/distill_topk512_qwen3_1b7.yaml",
    "examples/configs/recipes/llm/berlin/distill_topk512_qwen3_4b.yaml",
    "examples/configs/recipes/llm/berlin/distill_topk512_qwen3_8b.yaml",
    "examples/configs/recipes/llm/berlin/distill_topk512_qwen3_14b.yaml",
    "examples/configs/recipes/llm/berlin/distill_topk512_qwen3_1b7_p64_g4.yaml",
]:
    yaml.safe_load(open(p))
    print("OK", p)
PY
```

### Berlin Launch Commands

Run once in the shell before any of the commands below:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
```

#### 3-node configs

#### `distill_topk512_qwen3_1b7.yaml`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_1b7.yaml'
sbatch -J distill_topk512_qwen3_1b7 -N3 -t 24:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

#### `distill_topk512_qwen3_4b.yaml`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_4b.yaml'
sbatch -J distill_topk512_qwen3_4b -N3 -t 24:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

#### `distill_topk512_qwen3_8b.yaml`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_8b.yaml'
sbatch -J distill_topk512_qwen3_8b -N3 -t 24:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

#### `distill_topk512_qwen3_14b.yaml`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_14b.yaml'
sbatch -J distill_topk512_qwen3_14b -N3 -t 24:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

#### `distill_topk512_qwen3_1b7_p64_g4.yaml`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_1b7_p64_g4.yaml'
sbatch -J distill_topk512_qwen3_1b7_p64_g4 -N3 -t 24:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

#### 1-node configs (colocated)

#### `distill_topk512_qwen3_8b_1node.yaml`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_8b_1node.yaml'
sbatch -J distill_topk512_qwen3_8b_1node -N1 -t 24:00:00 --mem=600G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

#### 2-node config (train/generation split)

#### `distill_topk512_qwen3_14b_2node.yaml`

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/berlin/distill_topk512_qwen3_14b_2node.yaml'
sbatch -J distill_topk512_qwen3_14b_2node -N2 -t 24:00:00 --mem=500G --export=ALL,COMMAND="$COMMAND" ray_bare_berlin.sub
```

### Berlin OOM Guidance

H100 80GB has 2x the memory of A100 40GB, so OOM is less likely. If it occurs:

1. `berlin/distill_topk512_qwen3_14b_2node.yaml` baseline is split: `policy TP=8/DP=1` on 1 train node, `teacher TP=4/DP=2` on train node, and non-colocated generation on the other node (`vLLM TP=2`, `gpu_memory_utilization=0.55`).
2. If 14B still OOMs, use this fallback sequence in order:
   a. Reduce sequence-packing token budgets further (keep context/generation lengths unchanged): `policy.sequence_packing.{train_mb_tokens,logprob_mb_tokens}=12288` and `teacher.sequence_packing.{train_mb_tokens,logprob_mb_tokens}=12288`.
   b. Reduce generation concurrency: `policy.generation_batch_size=2`, `teacher.generation_batch_size=2`.
   c. Lower vLLM memory reservation: `policy.generation.vllm_cfg.gpu_memory_utilization=0.50`.
3. For generation OOM on 8B (vLLM TP=1): reduce `gpu_memory_utilization` from 0.80 to 0.70, or switch to vLLM TP=2 (4 engines instead of 8).

### Berlin Bring-Up Order

Same as JUWELS: 1.7B → 4B → 8B → 14B → 1.7B p64_g4
