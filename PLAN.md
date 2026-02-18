# Plan: Bare-Metal On-Policy Distillation on JUWELS (NeMo-RL)

## Context

NeMo-RL is available at `/p/project1/envcomp/yll/nemo-rl/`. Prior failed jobs and script history exist in `archive/`:

- `/p/project1/envcomp/yll/archive/multi-node-distillation.sh`
- `/p/project1/envcomp/yll/archive/nemo-rl-distill-13276129.out`
- `/p/project1/envcomp/yll/archive/nemo-rl-distill-13276216.out`

This plan uses a new bare-metal launcher, `ray_bare.sub`, and keeps the flow aligned with current NeMo-RL behavior.

---

## Step 1: Identify your Slurm partition and account

Run once on the login node:

```bash
sinfo -o "%P %G %D %N" | head -20
sacctmgr show assoc user=$USER format=account,partition -P | head -10
```

Use these values when submitting with `sbatch --partition=<PARTITION> --account=<ACCOUNT>`.

---

## Step 2: Confirm environment setup

From repo root:

```bash
cd /p/project1/envcomp/yll/nemo-rl

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Build/update env with distillation extras
NRL_FORCE_REBUILD_VENVS=true uv sync --extra fsdp --extra vllm
```

Quick check:

```bash
uv run python -c "import torch, ray, vllm; print('OK')"
```

Notes:
- If `deep_ep` fails due to ibverbs headers: load/install `libibverbs` on your system first.
- On JUWELS, run long setup commands in `tmux`/`screen`.

---

## Step 3: Use the bare-metal launcher

**File:** `/p/project1/envcomp/yll/nemo-rl/ray_bare.sub`

This launcher is now part of the repo and handles:

1. Head IP discovery via `srun hostname -I` before Ray startup.
2. Head/worker Ray startup using `--block` so Slurm cgroups keep processes alive.
3. Cluster readiness polling using `worker_units` from `ray status`.
4. Driver launch on head node with `srun --overlap` (prevents deadlock with blocked head step).
5. Absolute `.venv` paths for `ray`/`python` (no PATH dependency in `srun` shells).
6. Optional `COMMAND` override for workload; default distillation command if `COMMAND` is unset.
7. Logs in `$BASE_LOG_DIR/$SLURM_JOB_ID-logs/` (default base: `SLURM_SUBMIT_DIR`).

### Launcher interface

- `COMMAND` (optional): command executed after Ray cluster is ready.
- `BASE_LOG_DIR` (optional): base directory for logs.
- `GPUS_PER_NODE` (optional, default `4`): used for Ray resource tags and default distillation overrides.
- `DISTILL_CONFIG` (optional): config path for default distillation mode.
- `CKPT_DIR` (optional): checkpoint output dir for default distillation mode.
- `RAY_PORT`, `RAY_DASHBOARD_PORT` (optional): override auto-derived ports.

Example dry run:

```bash
bash -n ray_bare.sub
sbatch --test-only ray_bare.sub
```

---

## Step 4: Create a custom distillation config

Create:

`/p/project1/envcomp/yll/nemo-rl/examples/configs/recipes/llm/my-distillation-1n4g.yaml`

Template:

```yaml
defaults: ../../distillation_math.yaml

policy:
  model_name: "<YOUR_STUDENT_MODEL>"
  max_total_sequence_length: 4096
  train_global_batch_size: 16
  generation_batch_size: 16
  dtensor_cfg:
    tensor_parallel_size: 1
    context_parallel_size: 1
    activation_checkpointing: true
  generation:
    vllm_cfg:
      tensor_parallel_size: 2
    colocated:
      enabled: false
      resources:
        gpus_per_node: 2

teacher:
  model_name: "<YOUR_TEACHER_MODEL>"
  dtensor_cfg:
    tensor_parallel_size: 1
    context_parallel_size: 1

data:
  train:
    dataset_name: DeepScaler
  validation:
    dataset_name: AIME2024

distillation:
  num_prompts_per_step: 16
  max_num_steps: 100

cluster:
  gpus_per_node: 4
  num_nodes: 1

logger:
  log_dir: logs/my-distillation
  wandb:
    project: nemo-distillation
    name: my-distillation-1n4g
```

---

## Step 5: Single-node smoke test

```bash
cd /p/project1/envcomp/yll/nemo-rl

COMMAND="uv run python examples/run_distillation.py \
  --config examples/configs/recipes/llm/my-distillation-1n4g.yaml \
  distillation.max_num_steps=5 \
  logger.wandb_enabled=False" \
sbatch \
  --nodes=1 \
  --gres=gpu:4 \
  --partition=<PARTITION> \
  --account=<ACCOUNT> \
  ray_bare.sub
```

---

## Step 6: Scale to multi-node

```bash
cd /p/project1/envcomp/yll/nemo-rl

COMMAND="uv run python examples/run_distillation.py \
  --config examples/configs/recipes/llm/my-distillation-1n4g.yaml \
  cluster.num_nodes=2 \
  cluster.gpus_per_node=4 \
  teacher.dtensor_cfg.tensor_parallel_size=2 \
  distillation.num_prompts_per_step=32" \
sbatch \
  --nodes=2 \
  --gres=gpu:4 \
  --partition=<PARTITION> \
  --account=<ACCOUNT> \
  ray_bare.sub
```

For larger teachers on 2x4 GPUs, use this as reference:

- `examples/configs/recipes/llm/distillation-qwen3-32b-to-4b-base-2n4g-fsdp2tp1-long.v1.yaml`

---

## Monitoring and verification

Assuming default `BASE_LOG_DIR=$SLURM_SUBMIT_DIR`:

```bash
tail -f <SLURM_JOB_ID>-logs/ray-head.log
tail -f <SLURM_JOB_ID>-logs/ray-driver.log
```

Expected milestones:

1. Launcher prints `All workers connected!`.
2. Driver log shows training startup without Ray connection fallback errors.
3. Single-node smoke completes (`max_num_steps=5`) without OOM.
4. Multi-node run shows ongoing training progress.

---

## Important behavior note: `init_ray()` and resource tags

Current `init_ray()` behavior in `nemo_rl/distributed/virtual_cluster.py`:

- It attaches with `ray.init(address="auto")`.
- If cluster resources include a NeMo local tag (`nrl_tag_*`), it enforces CVD-tag matching.
- Otherwise it reuses the connected cluster as externally managed.

Implication:

- `slurm_managed_ray_cluster` is **not** the only condition for reusing a multi-node cluster.
- We still set it in `ray_bare.sub` for compatibility, cluster visibility, and consistency with `ray.sub`.

---

## Critical files

| File | Purpose |
|------|---------|
| `/p/project1/envcomp/yll/nemo-rl/ray_bare.sub` | Bare-metal Slurm launcher used by this plan |
| `/p/project1/envcomp/yll/nemo-rl/ray.sub` | Container/pyxis reference launcher |
| `/p/project1/envcomp/yll/nemo-rl/examples/run_distillation.py` | Distillation entrypoint |
| `/p/project1/envcomp/yll/nemo-rl/nemo_rl/distributed/virtual_cluster.py` | `init_ray()` reuse logic |
| `/p/project1/envcomp/yll/nemo-rl/examples/configs/distillation_math.yaml` | Base distillation config |
| `/p/project1/envcomp/yll/archive/multi-node-distillation.sh` | Historical script that exposed previous failures |

---

## Common gotchas

| Issue | Fix |
|-------|-----|
| `ray: command not found` inside `srun` | Use `.venv/bin/ray` and `.venv/bin/python` absolute paths |
| Workers never connect | Inspect `<JOB_ID>-logs/ray-worker-*.log` and `<JOB_ID>-logs/ray-head.log` |
| Driver never starts | Ensure launcher uses `srun --overlap` for driver step (already in `ray_bare.sub`) |
| `ray status` connection errors early in startup | Wait for readiness loop; do not launch workload before `All workers connected!` |
| OOM during distillation | Reduce batch sizes, sequence length, or teacher TP/model size |
| ibverbs/deep_ep build errors | Provide required ibverbs libs before `uv sync` |
| `/tmp` pressure from Ray | Set `RAY_TMPDIR` to scratch storage before submit |
| `undefined symbol ... libcudnn_graph.so.9` from TE/cuDNN | Avoid mixed cuDNN stacks: do not load cluster `cuDNN` module in this launcher; use wheel-provided cuDNN from NeMo-RL worker venvs |
