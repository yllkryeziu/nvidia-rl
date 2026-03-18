# Jupiter Distillation Runbook

All commands are run from `nvidia-rl/` directory.

**Cluster**: Jupiter — 4x NVIDIA GH200 (96GB HBM3) per node

---

## Prerequisites

1. Download the models into local HF cache:
   ```bash
   for model in Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B; do
     HF_HOME=/e/project1/scifi/kryeziu1/.cache/huggingface huggingface-cli download Qwen/$model
   done
   ```
2. Datasets at `/e/project1/scifi/kryeziu1/data/`:
   - `openthoughts114k-math-qwen3` — 89k math problems, columns: `qwen3_{size}_original_answer`
   - `dapo-math-17k-qwen3` — 17k math problems, columns: `qwen3_{size}_answer`
   - `deepmath-103k-qwen3` — 103k math problems, columns: `qwen3_{size}_answer` (no 32B)
3. Ensure the `.venv` is set up (see `env_jupiter.sh`).

---

## Directory layout

```
jupiter/
├── openthoughts114k-math-qwen3/   # 5 configs
├── dapo-math-17k-qwen3/           # 5 configs
├── deepmath-103k-qwen3/           # 5 configs
└── runbook.md
```

All configs use Megatron backend, colocated vLLM, batch 256, val every 10 steps (256 samples), checkpointing every 10 steps.

---

## Scaling summary

| Model | Nodes | Policy TP/PP/DP | Teacher TP/PP | vLLM TP | Grad accum | LR   |
|-------|-------|-----------------|---------------|---------|------------|------|
| 1.7B  | 1     | 2/2/1           | 2/1           | 1       | 256        | 1e-5 |
| 4B    | 1     | 2/2/1           | 2/1           | 1       | 256        | 1e-5 |
| 8B    | 2     | 2/2/2           | 2/1           | 1       | 128        | 5e-6 |
| 14B   | 2     | 4/2/1           | 4/1           | 2       | 256        | 3e-6 |
| 32B   | 4     | 4/2/2           | 4/1           | 4       | 128        | 3e-6 |

**Note**: 32B configs use `qwen3_14b` traces across all datasets (32B traces not yet generated).

---

## 1. openthoughts114k-math-qwen3

Dataset: 89k math problems filtered by `domain=math` | `input_key=problem` | `output_key=ground_truth_solution`

### 1.7B — 1 node

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/openthoughts114k-math-qwen3/distill_topk512_qwen3_1b7_1node_p256_v256.yaml'
sbatch -J distill-ot-qwen3-1b7 -N1 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 4B — 1 node

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/openthoughts114k-math-qwen3/distill_topk512_qwen3_4b_1node_p256_v256.yaml'
sbatch -J distill-ot-qwen3-4b -N1 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 8B — 2 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/openthoughts114k-math-qwen3/distill_topk512_qwen3_8b_2node_p256_v256.yaml'
sbatch -J distill-ot-qwen3-8b -N2 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 14B — 2 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/openthoughts114k-math-qwen3/distill_topk512_qwen3_14b_2node_p256_v256.yaml'
sbatch -J distill-ot-qwen3-14b -N2 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 32B — 4 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/openthoughts114k-math-qwen3/distill_topk512_qwen3_32b_4node_p256_v256.yaml'
sbatch -J distill-ot-qwen3-32b -N4 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

---

## 2. dapo-math-17k-qwen3

Dataset: 17k math problems, no filter | `input_key=problem` | `output_key=answer`

### 1.7B — 1 node

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/dapo-math-17k-qwen3/distill_topk512_qwen3_1b7_1node_p256_v256.yaml'
sbatch -J distill-dapo-qwen3-1b7 -N1 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 4B — 1 node

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/dapo-math-17k-qwen3/distill_topk512_qwen3_4b_1node_p256_v256.yaml'
sbatch -J distill-dapo-qwen3-4b -N1 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 8B — 2 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/dapo-math-17k-qwen3/distill_topk512_qwen3_8b_2node_p256_v256.yaml'
sbatch -J distill-dapo-qwen3-8b -N2 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 14B — 2 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/dapo-math-17k-qwen3/distill_topk512_qwen3_14b_2node_p256_v256.yaml'
sbatch -J distill-dapo-qwen3-14b -N2 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 32B — 4 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/dapo-math-17k-qwen3/distill_topk512_qwen3_32b_4node_p256_v256.yaml'
sbatch -J distill-dapo-qwen3-32b -N4 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

---

## 3. deepmath-103k-qwen3

Dataset: 103k math problems, no filter | `input_key=question` | `output_key=final_answer`

### 1.7B — 1 node

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/deepmath-103k-qwen3/distill_topk512_qwen3_1b7_1node_p256_v256.yaml'
sbatch -J distill-dm-qwen3-1b7 -N1 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 4B — 1 node

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/deepmath-103k-qwen3/distill_topk512_qwen3_4b_1node_p256_v256.yaml'
sbatch -J distill-dm-qwen3-4b -N1 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 8B — 2 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/deepmath-103k-qwen3/distill_topk512_qwen3_8b_2node_p256_v256.yaml'
sbatch -J distill-dm-qwen3-8b -N2 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 14B — 2 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/deepmath-103k-qwen3/distill_topk512_qwen3_14b_2node_p256_v256.yaml'
sbatch -J distill-dm-qwen3-14b -N2 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```

### 32B — 4 nodes

```bash
COMMAND='uv run python examples/run_distillation.py --config examples/configs/recipes/llm/jupiter/deepmath-103k-qwen3/distill_topk512_qwen3_32b_4node_p256_v256.yaml'
sbatch -J distill-dm-qwen3-32b -N4 -t 12:00:00 --export=ALL,COMMAND="$COMMAND" ray_bare_jupiter.sub
```
