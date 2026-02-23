# Distillation Runbook (Booster + `ray_bare.sub`)

This README documents:
- smoke tests
- normal on-policy distillation runs (1 node and multi-node)
- small and larger model recipes
- context/self-distillation runs
- exactly where to change student and teacher prompt templates

## 1) Prerequisites

Run from repo root:

```bash
cd /p/project1/envcomp/yll/nemo-rl
```

Use the Slurm launcher:
- `ray_bare.sub`

Main logs after submission:
- `nemo-rl-bare-<jobid>.out`
- `<jobid>-logs/ray-driver.log`
- `<jobid>-logs/ray-head.log`
- `<jobid>-logs/ray-status.log`

Model path note:
- The recipe configs referenced in this runbook are pinned to local Hugging Face snapshot paths
  under `/p/project1/envcomp/yll/.cache/huggingface/hub/...` to work with offline defaults in
  `ray_bare.sub`.

## 2) Smoke Tests

### 2.1 Unit smoke (context-distillation feature tests)

```bash
COMMAND='uv run --group test python -m pytest tests/unit/algorithms/test_context_distillation_utils.py tests/unit/algorithms/test_distillation.py -k "context_distillation"'
sbatch -N1 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

Expected result:
- `ray-driver.log` ends with `5 passed` (or more if you widened `-k`)

### 2.2 Normal distillation smoke (single node, short run)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/my-distillation-1n4g.yaml distillation.max_num_steps=5 logger.wandb_enabled=False'
sbatch -N1 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 2.3 Context/self-distillation smoke (single node, short run)

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/my-context-self-distillation-1n4g.yaml distillation.max_num_steps=5 logger.wandb_enabled=False'
sbatch -N1 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 2.4 Existing smoke/functional scripts already in repo

Useful built-in checks:
- `tests/functional/distillation.sh`
- `tests/test_suites/llm/distillation-qwen3-32b-to-1.7b-base-1n4g-fsdp2tp1.v1.sh`
- `tests/test_suites/llm/distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-long.v1.sh`

Run local functional smoke:

```bash
uv run bash tests/functional/distillation.sh
```

## 3) Normal Distillation Runs

### 3.1 One node, small model baseline

Config:
- `examples/configs/recipes/llm/my-distillation-1n4g.yaml`

Run:

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/my-distillation-1n4g.yaml logger.wandb_enabled=False'
sbatch -N1 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 3.2 One node, larger recipe (if hardware fits)

Config examples:
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-1.7b-base-1n4g-fsdp2tp1.v1.yaml`
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1.yaml`

Run:

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-32b-to-1.7b-base-1n4g-fsdp2tp1.v1.yaml logger.wandb_enabled=False'
sbatch -N1 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 3.3 Multi-node, larger recipe

Config examples:
- `examples/configs/recipes/llm/distillation-qwen3-14b-to-4b-2n4g-fsdp2tp1-long.v1.yaml` (recommended first on 2x4 GPUs)
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-2n4g-fsdp2tp1-long.v1.yaml`
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-long.v1.yaml`
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-seqpack.v1.yaml`

Run (2 nodes):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-2n4g-fsdp2tp1-long.v1.yaml logger.wandb_enabled=False'
sbatch -N2 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 3.4 Multi-node OOM fallback: smaller teacher first

If `32B -> 4B` fails at teacher logprob prep with CUDA OOM, run `14B -> 4B` first:

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-14b-to-4b-2n4g-fsdp2tp1-long.v1.yaml logger.wandb_enabled=False'
sbatch -N2 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

If this still OOMs, override teacher to 8B without creating another config:

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-14b-to-4b-2n4g-fsdp2tp1-long.v1.yaml teacher.model_name=/p/project1/envcomp/yll/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 logger.wandb_enabled=False'
sbatch -N2 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 3.5 Three-pool isolation (train + generation + teacher on separate nodes)

Use this when you want strict role isolation (recommended for larger teachers).

Config:
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-3n4g-isolated.v1.yaml`

Run (3 nodes, one node per role):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-3n4g-isolated.v1.yaml logger.wandb_enabled=False'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

Expected setup log marker in `ray-driver.log`:
- `Isolated clusters created: train=1x4GPUs, inference=1x4GPUs, teacher=1x4GPUs`

Fail-fast validation errors (and what they mean):
- `requires policy.generation.colocated.enabled=false`: disable colocated generation.
- `must exactly partition cluster nodes`: `training.num_nodes + generation.num_nodes + teacher.num_nodes` must equal `cluster.num_nodes`.
- `enforces node-level role isolation`: each role `gpus_per_node` must match `cluster.gpus_per_node`.

### 3.6 Three-pool isolation: smaller-model smoke recipes

These are short smoke runs with smaller teachers/models, still using strict 3-way isolation on `3x4 GPUs`.

Config examples:
- `examples/configs/recipes/llm/distillation-qwen3-14b-to-4b-3n4g-isolated-smoke.v1.yaml`
- `examples/configs/recipes/llm/distillation-qwen3-8b-to-4b-3n4g-isolated-smoke.v1.yaml`
- `examples/configs/recipes/llm/distillation-qwen3-4b-to-1.7b-3n4g-isolated-smoke.v1.yaml`

Run (`14B -> 4B`, smoke):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-14b-to-4b-3n4g-isolated-smoke.v1.yaml logger.wandb_enabled=False'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

Run (`8B -> 4B`, smoke):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-8b-to-4b-3n4g-isolated-smoke.v1.yaml logger.wandb_enabled=False'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

Run (`4B -> 1.7B`, smoke):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-4b-to-1.7b-3n4g-isolated-smoke.v1.yaml logger.wandb_enabled=False'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

## 4) Context/Self-Distillation Runs

### 4.1 Single node

Config:
- `examples/configs/recipes/llm/my-context-self-distillation-1n4g.yaml`

Run:

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/my-context-self-distillation-1n4g.yaml logger.wandb_enabled=False'
sbatch -N1 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 4.2 Multi-node

Start from the same config and override cluster size:

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/my-context-self-distillation-1n4g.yaml cluster.num_nodes=2 cluster.gpus_per_node=4 logger.wandb_enabled=False'
sbatch -N2 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

### 4.3 16k Context/Self-Distillation on OpenThoughts-114k (3x4 GPUs, W&B enabled)

These recipes are preconfigured for:
- student generation `max_new_tokens=16384`
- teacher/student max sequence length `20480`
- train dataset `open-thoughts/OpenThoughts-114k` (`subset=metadata`, `input_key=problem`)
- `wandb_enabled=true`

Run (`1.7B -> 1.7B`):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-1p7b-3n4g.v1.yaml'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

Run (`4B -> 4B`):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-4b-3n4g.v1.yaml'
sbatch -N4 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

Note: 4B self-distillation requires 4 nodes (training cluster gets 2 nodes for DP=2×TP=4).
Config `context-self-distill-qwen3-4b-3n4g.v1.yaml` already encodes `cluster.num_nodes=4` and
`resource_isolation.roles.training.num_nodes=2`, so no overrides needed.

Run (`8B -> 8B`):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-8b-3n4g.v1.yaml'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

Run (`14B -> 14B`):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/context-self-distill-qwen3-14b-3n4g.v1.yaml'
sbatch -N3 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
```

W&B note:
- Ensure `WANDB_API_KEY` is available in the job environment before `sbatch`.

Important V1 constraints for context/self-distillation:
- `distillation.max_rollout_turns` must be `1`
- `policy.model_name` must equal `teacher.model_name`
- `distillation.context_distillation.mode` must be `self_frozen`
- `distillation.context_distillation.alignment` must be `response_token_index`
- `distillation.context_distillation.problem_source` must be `original_user_problem`

## 5) Where To Change Prompts/Templates

### 5.1 Student prefix prompt (problem wrapper)

Change in config:
- `data.default.prompt_file`
- optionally task-specific: `data.train.prompt_file`, `data.validation.prompt_file`

Base location:
- `examples/configs/distillation_math.yaml`

Current default prompt file:
- `examples/prompts/cot.txt`

Important:
- For math processors, the prompt text is applied as `prompt.format(problem)`.
- Your prompt file should contain `{}` where the raw problem should be inserted.

Example `my_student_prompt.txt`:

```text
This is a <problem>, solve it step by step.
{}
```

Then point config to it:

```yaml
data:
  default:
    prompt_file: "examples/prompts/my_student_prompt.txt"
```

### 5.2 Student system prompt

Change in config:
- `data.default.system_prompt_file`

Set to a text file path or `null`.

### 5.3 Teacher prefix template (context-distillation teacher prompt)

Change in config:
- `distillation.context_distillation.teacher_prefix_template`

Recipe location:
- `examples/configs/recipes/llm/my-context-self-distillation-1n4g.yaml`

Default:

```yaml
distillation:
  context_distillation:
    teacher_prefix_template: "You are given a problem and a trace: {problem} + {trace}. Solve it on your own."
```

Important:
- This template uses named placeholders: `{problem}` and `{trace}`.

### 5.4 Trace extraction behavior

Config keys:
- `distillation.context_distillation.trace_extractor.type`
- `distillation.context_distillation.trace_extractor.selection`
- `distillation.context_distillation.trace_extractor.missing_trace_policy`

Current V1 implementation:
- first `<think>...</think>` span is extracted
- code path: `nemo_rl/algorithms/context_distillation_utils.py`
- extractor function: `extract_first_think_span`

## 6) Scale-Up Checklist

Before scaling up, verify all items:
1. Smoke tests pass (unit + short training).
2. `cluster.num_nodes` and Slurm `sbatch -N` match. For isolated runs: `cluster.num_nodes` = sum of all role `num_nodes` (e.g. 4B self-distill: training=2 + generation=1 + teacher=1 = 4).
3. `cluster.gpus_per_node` matches actual GPUs per node.
4. `policy.max_total_sequence_length` and `teacher.max_total_sequence_length` fit memory.
5. `policy.dtensor_cfg.tensor_parallel_size` and `context_parallel_size` are valid for your node topology.
6. `policy.generation.vllm_cfg.max_model_len` is consistent with sequence length.
7. For context/self-distillation, student and teacher model names are identical.
8. `distillation.topk_logits_k` is set as desired (default in base config is `64`).

## 7) Debugging Quick Commands

```bash
tail -n 200 nemo-rl-bare-<jobid>.out
tail -n 200 <jobid>-logs/ray-driver.log
tail -n 200 <jobid>-logs/ray-head.log
tail -n 200 <jobid>-logs/ray-status.log
```

For context-distillation runs, look for these metrics in logs:
- `context_distillation_trace_coverage`
- `context_distillation_prefix_truncation_rate`
- `context_distillation_dropped_samples`
