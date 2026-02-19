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
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-2n4g-fsdp2tp1-long.v1.yaml`
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-long.v1.yaml`
- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-seqpack.v1.yaml`

Run (2 nodes):

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-2n4g-fsdp2tp1-long.v1.yaml logger.wandb_enabled=False'
sbatch -N2 --export=ALL,COMMAND="$COMMAND" ray_bare.sub
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
2. `cluster.num_nodes` and Slurm `sbatch -N` match.
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
