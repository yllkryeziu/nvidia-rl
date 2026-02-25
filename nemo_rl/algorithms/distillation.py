# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
# limitations under the License.
import os
import warnings
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.context_distillation_utils import (
    ContextDistillationBuildResult,
    align_teacher_topk_to_student_positions,
    build_context_distillation_teacher_batch,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class ContextDistillationTraceExtractorConfig(TypedDict):
    type: str
    selection: str
    missing_trace_policy: str


class ContextDistillationStaticTraceConfig(TypedDict):
    answer_column: str


class ContextDistillationMetricsConfig(TypedDict):
    enabled: bool


class ContextDistillationConfig(TypedDict):
    enabled: bool
    mode: str
    problem_source: str
    trace_source: NotRequired[str]
    static_trace: NotRequired[ContextDistillationStaticTraceConfig]
    teacher_prefix_template: str
    trace_extractor: ContextDistillationTraceExtractorConfig
    alignment: str
    overflow_policy: str
    frozen_teacher_source: str
    metrics: ContextDistillationMetricsConfig


class ContextDistillationRuntimeConfig(TypedDict):
    enabled: bool
    problem_source: str
    trace_source: str
    static_trace_answer_column: str
    teacher_prefix_template: str
    trace_type: str
    trace_selection: str
    missing_trace_policy: str
    overflow_policy: str
    metrics_enabled: bool
    teacher_max_sequence_length: int


class DistillationRoleResourcesConfig(TypedDict):
    num_nodes: int
    gpus_per_node: int


class DistillationResourceIsolationRolesConfig(TypedDict):
    training: DistillationRoleResourcesConfig
    generation: DistillationRoleResourcesConfig
    teacher: DistillationRoleResourcesConfig


class DistillationResourceIsolationConfig(TypedDict):
    enabled: bool
    roles: DistillationResourceIsolationRolesConfig
    teacher_resident_on_gpu: NotRequired[bool]


class DistillationConfig(TypedDict):
    # Training configuration
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int  # for multi-turn rollouts. Math Environments just have 1 turn (answering the question)
    max_num_steps: int  # maximum number of steps to train for
    max_num_epochs: int  # maximum number of epochs to train for
    val_batch_size: int
    val_period: int
    val_at_start: bool
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool
    max_val_samples: int
    val_track_train_metrics: NotRequired[bool]
    topk_logits_k: int
    seed: int
    context_distillation: NotRequired[ContextDistillationConfig]
    resource_isolation: NotRequired[DistillationResourceIsolationConfig]


class DistillationSaveState(TypedDict):
    total_steps: int  # Track total number of steps across all epochs
    current_epoch: int  # Track current epoch
    current_step: int  # Track step within current epoch
    val_reward: NotRequired[
        float
    ]  # Can be any metric. Setted to 'accuracy' by default in validation.
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _get_context_trace_source_and_static_column(
    context_distillation_cfg: dict[str, Any],
) -> tuple[str, str]:
    trace_source = str(context_distillation_cfg.get("trace_source", "dynamic"))
    if trace_source not in {"dynamic", "static_dataset"}:
        raise ValueError(
            "distillation.context_distillation.trace_source must be one of "
            "{'dynamic', 'static_dataset'}. "
            f"Got {trace_source!r}."
        )

    static_trace_cfg = context_distillation_cfg.get("static_trace", {})
    if static_trace_cfg is None:
        static_trace_cfg = {}
    if not isinstance(static_trace_cfg, dict):
        raise ValueError(
            "distillation.context_distillation.static_trace must be a mapping "
            "when provided."
        )
    answer_column = static_trace_cfg.get("answer_column", "")
    if answer_column is None:
        answer_column = ""
    if not isinstance(answer_column, str):
        raise ValueError(
            "distillation.context_distillation.static_trace.answer_column must be a string."
        )

    if trace_source == "static_dataset" and not answer_column.strip():
        raise ValueError(
            "distillation.context_distillation.static_trace.answer_column must be set "
            "when trace_source='static_dataset'."
        )

    return trace_source, answer_column


def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,  # Aligned with GRPO
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


def _get_data_parallel_size(policy: ColocatablePolicyInterface) -> int:
    """Best-effort retrieval of policy data parallel size."""
    sharding_annotations = getattr(policy, "sharding_annotations", None)
    if sharding_annotations is None:
        return 1
    try:
        return int(sharding_annotations.get_axis_size("data_parallel"))
    except Exception:
        return 1


def _make_context_teacher_batch_dp_compatible(
    *,
    context_batch: ContextDistillationBuildResult,
    teacher_dp_size: int,
) -> int:
    """Trim context teacher batch so its size is divisible by teacher DP size.

    Returns number of samples dropped from the context teacher batch.
    """
    if teacher_dp_size <= 1 or context_batch.teacher_data is None:
        return 0

    teacher_batch_size = int(context_batch.teacher_data["input_ids"].shape[0])
    remainder = teacher_batch_size % teacher_dp_size
    if remainder == 0:
        return 0

    keep_count = teacher_batch_size - remainder
    dropped_sample_indices = context_batch.valid_sample_indices[keep_count:]
    for sample_idx in dropped_sample_indices:
        context_batch.sample_mask[sample_idx] = 0.0

    if keep_count <= 0:
        context_batch.teacher_data = None
        context_batch.valid_sample_indices = []
        context_batch.alignments = []
        return teacher_batch_size

    context_batch.teacher_data = BatchedDataDict(
        {
            "input_ids": context_batch.teacher_data["input_ids"][:keep_count],
            "input_lengths": context_batch.teacher_data["input_lengths"][:keep_count],
        }
    )
    context_batch.valid_sample_indices = context_batch.valid_sample_indices[:keep_count]
    context_batch.alignments = context_batch.alignments[:keep_count]
    return remainder


_MEAN_REDUCED_METRICS = {
    "lr",
    "wd",
    "global_valid_seqs",
    "global_valid_toks",
    "mean_prompt_length",
}


def _to_numpy_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _reduce_metric(value: Any, *, use_mean: bool) -> float:
    arr = _to_numpy_array(value)
    if arr.size == 0:
        return 0.0
    if use_mean:
        return float(np.mean(arr))
    return float(np.sum(arr))


def _build_train_data_from_rollout(
    *,
    repeated_batch: BatchedDataDict[DatumSpec],
    tokenizer,
    make_sequence_length_divisible_by: int,
) -> tuple[BatchedDataDict[DistillationLossDataDict], BatchedDataDict[Any], torch.Tensor]:
    # Mark assistant tokens for loss and non-assistant tokens as masked out.
    for message_log in repeated_batch["message_log"]:
        for message in message_log:
            if message["role"] == "assistant":
                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
            else:
                message["token_loss_mask"] = torch.zeros_like(message["token_ids"])

    flat_messages, input_lengths = batched_message_log_to_flat_message(
        repeated_batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    train_data = BatchedDataDict[DistillationLossDataDict](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat_messages["token_loss_mask"],
            "sample_mask": repeated_batch["loss_multiplier"],
        }
    )
    # Keep packed multimodal entries so policy microbatching can handle them.
    train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
    train_data.to("cpu")
    return train_data, flat_messages, input_lengths


def _get_required_extra_env_infos(
    repeated_batch: BatchedDataDict[DatumSpec],
) -> list[dict[str, Any] | None]:
    extra_env_infos = repeated_batch.get("extra_env_info")
    if not isinstance(extra_env_infos, list):
        raise ValueError(
            "Context distillation requires batched 'extra_env_info' with raw 'problem'."
        )
    if len(extra_env_infos) != len(repeated_batch["message_log"]):
        raise ValueError(
            "extra_env_info length must match message_log length for context distillation."
        )
    return cast(list[dict[str, Any] | None], extra_env_infos)


def _populate_teacher_topk_for_train_data(
    *,
    train_data: BatchedDataDict[DistillationLossDataDict],
    repeated_batch: BatchedDataDict[DatumSpec],
    extra_env_infos: list[dict[str, Any] | None],
    teacher_policy: ColocatablePolicyInterface,
    tokenizer,
    topk_k: int,
    timer: Optional[Timer],
    context_runtime_cfg: ContextDistillationRuntimeConfig,
    teacher_dp_size: int,
    debug_print_first_sample: bool,
) -> tuple[dict[str, float], bool]:
    context_step_metrics: dict[str, float] = {}
    debug_dump_printed = False
    if context_runtime_cfg["enabled"]:
        context_batch = build_context_distillation_teacher_batch(
            message_logs=repeated_batch["message_log"],
            sample_mask=train_data["sample_mask"],
            student_input_lengths=train_data["input_lengths"],
            extra_env_infos=extra_env_infos,
            tokenizer=tokenizer,
            teacher_prefix_template=context_runtime_cfg["teacher_prefix_template"],
            max_teacher_sequence_length=context_runtime_cfg[
                "teacher_max_sequence_length"
            ],
            pad_token_id=tokenizer.pad_token_id,
            problem_source=context_runtime_cfg["problem_source"],
            trace_source=context_runtime_cfg["trace_source"],
            static_trace_answer_column=context_runtime_cfg[
                "static_trace_answer_column"
            ],
            trace_extractor_type=context_runtime_cfg["trace_type"],
            trace_extractor_selection=context_runtime_cfg["trace_selection"],
            missing_trace_policy=context_runtime_cfg["missing_trace_policy"],
            overflow_policy=context_runtime_cfg["overflow_policy"],
            metrics_enabled=context_runtime_cfg["metrics_enabled"],
            debug_print_first_sample=debug_print_first_sample,
        )
        debug_dump_printed = context_batch.debug_dump_printed
        train_data["sample_mask"] = context_batch.sample_mask
        context_step_metrics = context_batch.metrics
        dropped_for_dp_divisibility = _make_context_teacher_batch_dp_compatible(
            context_batch=context_batch,
            teacher_dp_size=teacher_dp_size,
        )
        if dropped_for_dp_divisibility > 0:
            train_data["sample_mask"] = context_batch.sample_mask
            context_step_metrics["context_distillation_dropped_for_dp_divisibility"] = (
                float(dropped_for_dp_divisibility)
            )

        batch_size, student_seq_len = train_data["input_ids"].shape[:2]
        if context_batch.teacher_data is None:
            train_data["teacher_topk_logits"] = torch.zeros(
                (batch_size, student_seq_len, topk_k),
                dtype=torch.float32,
            )
            train_data["teacher_topk_indices"] = torch.zeros(
                (batch_size, student_seq_len, topk_k),
                dtype=torch.long,
            )
        else:
            teacher_topk = teacher_policy.get_topk_logits(
                context_batch.teacher_data,
                k=topk_k,
                timer=timer,
            )
            aligned_logits, aligned_indices = align_teacher_topk_to_student_positions(
                teacher_topk_logits=teacher_topk["topk_logits"],
                teacher_topk_indices=teacher_topk["topk_indices"],
                alignments=context_batch.alignments,
                valid_sample_indices=context_batch.valid_sample_indices,
                student_batch_size=batch_size,
                student_sequence_length=student_seq_len,
                topk=topk_k,
            )
            train_data["teacher_topk_logits"] = aligned_logits
            train_data["teacher_topk_indices"] = aligned_indices

            if debug_print_first_sample:
                # One-time sanity check: does teacher top-k support the student's first response token
                # (ideally the first token piece of "<think>") at the first scored assistant position?
                sample0_alignment = None
                for alignment in context_batch.alignments:
                    if alignment.sample_idx == 0:
                        sample0_alignment = alignment
                        break
                if sample0_alignment is not None and sample0_alignment.num_response_tokens > 0:
                    student_pred_row = sample0_alignment.student_pred_start
                    first_response_col = student_pred_row + 1
                    if (
                        0 <= student_pred_row < aligned_indices.shape[1]
                        and 0 <= first_response_col < train_data["input_ids"].shape[1]
                    ):
                        student_first_token_id = int(
                            train_data["input_ids"][0, first_response_col].item()
                        )
                        topk_ids = (
                            aligned_indices[0, student_pred_row, :]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        topk_logits_vals = (
                            aligned_logits[0, student_pred_row, :]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        student_rank = next(
                            (
                                rank
                                for rank, tok_id in enumerate(topk_ids)
                                if int(tok_id) == student_first_token_id
                            ),
                            None,
                        )
                        think_token_ids = tokenizer(
                            "<think>",
                            add_special_tokens=False,
                            return_tensors="pt",
                        )["input_ids"][0].tolist()
                        think_first_token_id = (
                            int(think_token_ids[0]) if len(think_token_ids) > 0 else None
                        )
                        think_first_rank = (
                            next(
                                (
                                    rank
                                    for rank, tok_id in enumerate(topk_ids)
                                    if int(tok_id) == think_first_token_id
                                ),
                                None,
                            )
                            if think_first_token_id is not None
                            else None
                        )
                        preview_n = min(10, len(topk_ids))
                        topk_preview = [
                            {
                                "rank": i + 1,
                                "id": int(topk_ids[i]),
                                "tok": _token_debug_str(tokenizer, int(topk_ids[i])),
                                "logit": float(topk_logits_vals[i]),
                            }
                            for i in range(preview_n)
                        ]

                        preview_num_positions = min(
                            5, int(sample0_alignment.num_response_tokens)
                        )
                        first_scored_positions_preview: list[dict[str, Any]] = []
                        for pos_offset in range(preview_num_positions):
                            pos_student_pred_row = student_pred_row + pos_offset
                            pos_response_col = first_response_col + pos_offset
                            if (
                                pos_student_pred_row < 0
                                or pos_student_pred_row >= aligned_indices.shape[1]
                                or pos_response_col < 0
                                or pos_response_col >= train_data["input_ids"].shape[1]
                            ):
                                continue

                            pos_student_token_id = int(
                                train_data["input_ids"][0, pos_response_col].item()
                            )
                            pos_topk_ids = (
                                aligned_indices[0, pos_student_pred_row, :]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            pos_topk_logits_vals = (
                                aligned_logits[0, pos_student_pred_row, :]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            pos_student_rank = next(
                                (
                                    rank
                                    for rank, tok_id in enumerate(pos_topk_ids)
                                    if int(tok_id) == pos_student_token_id
                                ),
                                None,
                            )
                            pos_preview_n = min(10, len(pos_topk_ids))
                            pos_topk_preview = [
                                {
                                    "rank": i + 1,
                                    "id": int(pos_topk_ids[i]),
                                    "tok": _token_debug_str(
                                        tokenizer, int(pos_topk_ids[i])
                                    ),
                                    "logit": float(pos_topk_logits_vals[i]),
                                }
                                for i in range(pos_preview_n)
                            ]
                            first_scored_positions_preview.append(
                                {
                                    "scored_pos_offset": pos_offset,
                                    "student_pred_row": int(pos_student_pred_row),
                                    "student_token_id": pos_student_token_id,
                                    "student_token": _token_debug_str(
                                        tokenizer, pos_student_token_id
                                    ),
                                    "student_token_rank_in_teacher_topk": (
                                        (pos_student_rank + 1)
                                        if pos_student_rank is not None
                                        else "not_in_topk"
                                    ),
                                    "teacher_topk_preview": pos_topk_preview,
                                }
                            )
                        print(
                            (
                                "\n===== CONTEXT DISTILLATION TEACHER TOP-K CHECK =====\n"
                                f"sample_idx: 0\n"
                                f"student_first_response_token_id: {student_first_token_id}\n"
                                f"student_first_response_token: {_token_debug_str(tokenizer, student_first_token_id)}\n"
                                f"student_first_response_token_rank_in_teacher_topk: "
                                f"{(student_rank + 1) if student_rank is not None else 'not_in_topk'}\n"
                                f"tokenization('<think>'): {think_token_ids}\n"
                                f"tokenization('<think>') pieces: "
                                f"{[_token_debug_str(tokenizer, int(t)) for t in think_token_ids]}\n"
                                f"first_token_of_<think>_rank_in_teacher_topk: "
                                f"{(think_first_rank + 1) if think_first_rank is not None else 'not_in_topk'}\n"
                                f"teacher_topk_preview_first_scored_pos: {topk_preview}\n"
                                f"teacher_topk_preview_first_{preview_num_positions}_scored_positions: "
                                f"{first_scored_positions_preview}\n"
                                "===== END CONTEXT DISTILLATION TEACHER TOP-K CHECK =====\n"
                            ),
                            flush=True,
                        )
    else:
        teacher_topk = teacher_policy.get_topk_logits(
            train_data,
            k=topk_k,
            timer=timer,
        )
        train_data["teacher_topk_logits"] = teacher_topk["topk_logits"]
        train_data["teacher_topk_indices"] = teacher_topk["topk_indices"]

    return context_step_metrics, debug_dump_printed


def _finalize_distillation_step_metrics(
    *,
    train_results: dict[str, Any],
    repeated_batch: BatchedDataDict[DatumSpec],
    input_lengths: torch.Tensor,
    rollout_metrics: dict[str, Any],
    context_step_metrics: dict[str, float],
) -> dict[str, Any]:
    raw_metrics: dict[str, Any] = {
        "loss": train_results["loss"],
        "mean_prompt_length": repeated_batch["length"],
        "total_num_tokens": input_lengths,
    }
    grad_norm_value = train_results.get("grad_norm")
    raw_metrics["grad_norm"] = (
        np.asarray([np.nan], dtype=np.float32)
        if grad_norm_value is None
        else grad_norm_value
    )
    raw_metrics.update(train_results["all_mb_metrics"])

    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        metrics[key] = _reduce_metric(value, use_mean=key in _MEAN_REDUCED_METRICS)

    metrics.update(rollout_metrics)
    metrics.update(context_step_metrics)
    return metrics


def _mean_aggregate_metric_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, float]:
    if not metric_dicts:
        return {}

    per_key_values: dict[str, list[float]] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if not isinstance(value, (int, float, np.number)):
                continue
            per_key_values.setdefault(key, []).append(float(value))

    return {
        key: float(np.mean(values)) if len(values) > 0 else 0.0
        for key, values in per_key_values.items()
    }


def _token_debug_str(tokenizer, token_id: int) -> str:
    try:
        piece = tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:
        piece = None
    if piece is None:
        try:
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            piece = "<decode_error>"
    return str(piece)


def _validate_resource_isolation_cfg(
    resource_isolation_cfg: DistillationResourceIsolationConfig,
    cluster_config: ClusterConfig,
    generation_config: dict[str, Any],
) -> DistillationResourceIsolationRolesConfig:
    assert generation_config["backend"] == "vllm", (
        "distillation.resource_isolation currently supports vLLM generation only. "
        f"Got policy.generation.backend={generation_config['backend']}."
    )
    assert not generation_config["colocated"]["enabled"], (
        "distillation.resource_isolation requires policy.generation.colocated.enabled=false."
    )

    roles = resource_isolation_cfg.get("roles")
    assert roles is not None, (
        "distillation.resource_isolation.roles must be set with training, generation, and teacher entries."
    )

    cluster_nodes = cluster_config["num_nodes"]
    cluster_gpus_per_node = cluster_config["gpus_per_node"]

    role_names = ("training", "generation", "teacher")
    total_role_nodes = 0
    for role_name in role_names:
        role_cfg = roles.get(role_name)
        assert role_cfg is not None, (
            f"distillation.resource_isolation.roles.{role_name} must be provided."
        )
        role_nodes = role_cfg.get("num_nodes")
        role_gpus = role_cfg.get("gpus_per_node")
        assert isinstance(role_nodes, int) and role_nodes > 0, (
            "distillation.resource_isolation.roles."
            f"{role_name}.num_nodes must be a positive integer, got {role_nodes}."
        )
        assert isinstance(role_gpus, int) and role_gpus > 0, (
            "distillation.resource_isolation.roles."
            f"{role_name}.gpus_per_node must be a positive integer, got {role_gpus}."
        )
        assert role_gpus == cluster_gpus_per_node, (
            "distillation.resource_isolation enforces node-level role isolation. "
            "Set distillation.resource_isolation.roles."
            f"{role_name}.gpus_per_node={cluster_gpus_per_node} to match cluster.gpus_per_node, got {role_gpus}."
        )
        total_role_nodes += role_nodes

    assert total_role_nodes == cluster_nodes, (
        "distillation.resource_isolation role nodes must exactly partition cluster nodes. "
        f"Expected sum(role.num_nodes)={cluster_nodes}, got {total_role_nodes}. "
        "Check distillation.resource_isolation.roles.*.num_nodes and cluster.num_nodes."
    )
    return roles


class MasterConfig(TypedDict):
    """Main configuration structure."""

    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]  # Environment configuration
    data: DataConfig  # Data configuration
    distillation: DistillationConfig  # Distillation configuration
    logger: LoggerConfig  # Logger configuration
    cluster: ClusterConfig  # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration


# ===============================================================================
# Setup & Initialization
# ===============================================================================
def check_vocab_equality(
    tokenizer: TokenizerType, student_model_name: str, teacher_model_name: str
) -> None:
    """Check if the vocab of the tokenizer (student) and the teacher tokenizer are equal."""
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    skip_hint = "Set NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true to skip this check."

    # 1) Exact token->id mapping equality
    vocab_a = tokenizer.get_vocab()
    vocab_b = teacher_tokenizer.get_vocab()
    assert vocab_a == vocab_b, (
        f"Token->ID mapping differs between student and teacher. {skip_hint}"
    )

    # 2) Size consistency (sanity checks)
    assert len(tokenizer) == len(teacher_tokenizer), (
        f"Effective vocab sizes differ between student and teacher. {skip_hint}"
    )

    # 3) Chech model.config.vocab_size to guarantee the last dimension of the logits is the same
    student_config = AutoConfig.from_pretrained(student_model_name)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name)
    assert student_config.vocab_size == teacher_config.vocab_size, (
        f"Model config vocab sizes differ between student and teacher. {skip_hint}"
    )


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,  # student_policy
    ColocatablePolicyInterface,  # teacher_policy
    Optional[GenerationInterface],  # student_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DistillationLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """Main entry point for distillation algorithm.

    Returns:
        tuple of student_policy, teacher_policy, student_generation,
        train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # Extract configuration
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    resource_isolation_cfg = distillation_config.get("resource_isolation")
    resource_isolation_enabled = bool(
        resource_isolation_cfg and resource_isolation_cfg.get("enabled", False)
    )
    teacher_resident_on_gpu = bool(
        resource_isolation_cfg.get("teacher_resident_on_gpu", True)
    ) if resource_isolation_enabled else False

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for distillation"
    )

    # Disallow SP + packing for dtensor path
    for cfg, who in ((policy_config, "student"), (teacher_config, "teacher")):
        # DTensor sequence parallel is supported; ensure CP and SP are not enabled together
        # This incompatibility is enforced in DTensor workers during initialization.
        # Additionally, SP may not be compatible with sequence packing for some models.
        # Refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details.
        # Therefore, we disable SP + packing for distillation.
        dtensor_enabled = cfg["dtensor_cfg"]["enabled"]
        sequence_packing_enabled = (
            "sequence_packing" in cfg and cfg["sequence_packing"]["enabled"]
        )
        sequence_parallel_enabled = (
            "sequence_parallel" in cfg["dtensor_cfg"]
            and cfg["dtensor_cfg"]["sequence_parallel"]
        )

        if dtensor_enabled and sequence_packing_enabled and sequence_parallel_enabled:
            raise AssertionError(
                f"Distillation does not support DTensor sequence parallel + sequence packing ({who} policy). "
                "Please refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details."
            )

    # Set random seed
    set_seed(distillation_config["seed"])

    context_distillation_cfg = distillation_config.get("context_distillation")
    if context_distillation_cfg and context_distillation_cfg.get("enabled", False):
        if context_distillation_cfg.get("mode", "self_frozen") != "self_frozen":
            raise AssertionError(
                "distillation.context_distillation.mode must be 'self_frozen' for V1."
            )
        if (
            context_distillation_cfg.get("frozen_teacher_source", "base_model")
            != "base_model"
        ):
            raise AssertionError(
                "distillation.context_distillation.frozen_teacher_source must be 'base_model' for V1."
            )
        if (
            context_distillation_cfg.get("alignment", "response_token_index")
            != "response_token_index"
        ):
            raise AssertionError(
                "distillation.context_distillation.alignment must be 'response_token_index' for V1."
            )
        if (
            context_distillation_cfg.get("problem_source", "original_user_problem")
            != "original_user_problem"
        ):
            raise AssertionError(
                "distillation.context_distillation.problem_source must be 'original_user_problem' for V1."
            )
        _get_context_trace_source_and_static_column(context_distillation_cfg)
        assert policy_config["model_name"] == teacher_config["model_name"], (
            "For self_frozen context distillation V1, policy.model_name and teacher.model_name must be identical."
        )
        assert distillation_config["max_rollout_turns"] == 1, (
            "Context distillation V1 supports single-turn rollouts only. Set distillation.max_rollout_turns=1."
        )

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(
        f"  ✓ Training dataloader loaded with {len(train_dataset)} samples", flush=True
    )

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if (
        distillation_config["val_period"] > 0
        or distillation_config["val_at_start"]
        or distillation_config["val_at_end"]
    ):
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]
    inference_nodes = 0
    inference_gpus_per_node = 0

    if resource_isolation_enabled:
        assert resource_isolation_cfg is not None
        roles = _validate_resource_isolation_cfg(
            resource_isolation_cfg=resource_isolation_cfg,
            cluster_config=cluster_config,
            generation_config=generation_config,
        )
        training_role = roles["training"]
        generation_role = roles["generation"]
        teacher_role = roles["teacher"]
        inference_nodes = generation_role["num_nodes"]
        inference_gpus_per_node = generation_role["gpus_per_node"]

        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[training_role["gpus_per_node"]]
            * training_role["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=training_role["gpus_per_node"],
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[generation_role["gpus_per_node"]]
            * generation_role["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=generation_role["gpus_per_node"],
            max_colocated_worker_groups=3,
        )
        teacher_cluster = RayVirtualCluster(
            name="distillation_teacher_cluster",
            bundle_ct_per_node_list=[teacher_role["gpus_per_node"]]
            * teacher_role["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=teacher_role["gpus_per_node"],
            max_colocated_worker_groups=1,
        )
        print(
            "  ✓ Isolated clusters created: "
            f"train={training_role['num_nodes']}x{training_role['gpus_per_node']}GPUs, "
            f"inference={generation_role['num_nodes']}x{generation_role['gpus_per_node']}GPUs, "
            f"teacher={teacher_role['num_nodes']}x{teacher_role['gpus_per_node']}GPUs, "
            f"teacher_resident_on_gpu={teacher_resident_on_gpu}",
            flush=True,
        )
    elif colocated_inference:
        cluster = RayVirtualCluster(
            name="distillation_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        teacher_cluster = cluster
        print(
            f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes",
            flush=True,
        )
    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set to a value > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set and equal to cluster.gpus_per_node "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got inference_gpus_per_node={inference_gpus_per_node}, cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        # create clusters
        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        teacher_cluster = train_cluster
        print(
            f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs",
            flush=True,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)
    # Checkpoint paths
    weights_path = None
    optimizer_path = None

    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    if "megatron_cfg" in teacher_config and teacher_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        teacher_config["megatron_cfg"]["train_iters"] = total_train_iters

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=teacher_cluster,
        config=teacher_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=False,
        init_reference_model=False,
    )
    if not teacher_resident_on_gpu:
        teacher_policy.offload_after_refit()
    else:
        print(
            "  ✓ Teacher will remain resident on GPU between distillation steps.",
            flush=True,
        )

    # ==========================
    #    Student Generation Interface
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "vllm_cfg" in generation_config:
            ## make vllm hf overrides match the training policy
            generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up student policy...", flush=True)

    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = train_cluster.world_size()
        # inference cluster + head node of the train cluster
        world_size = train_world_size + inference_nodes * inference_gpus_per_node
        # init collective
        futures_train = student_policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = student_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    loss_fn = DistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )


# ===============================================================================
# Training & Validation
# ===============================================================================


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Run Distillation training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If student_generation is None, use the student_policy as the generation interface (megatron framework backend)
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check

    # common config/state items
    current_epoch = distillation_save_state["current_epoch"]  # current epoch
    current_step = distillation_save_state[
        "current_step"
    ]  # current step within current epoch
    total_steps = distillation_save_state[
        "total_steps"
    ]  # total number of steps across all epochs
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    val_period = master_config["distillation"]["val_period"]
    val_at_start = master_config["distillation"]["val_at_start"]
    val_at_end = master_config["distillation"]["val_at_end"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = master_config["distillation"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    max_steps = master_config["distillation"][
        "max_num_steps"
    ]  # max number of steps to train for
    context_distillation_cfg = master_config["distillation"].get(
        "context_distillation"
    )
    context_distillation_enabled = bool(
        context_distillation_cfg and context_distillation_cfg.get("enabled", False)
    )
    context_metrics_enabled = True
    context_teacher_prefix_template = ""
    context_problem_source = "original_user_problem"
    context_trace_source = "dynamic"
    context_static_trace_answer_column = ""
    context_trace_type = "think_tag"
    context_trace_selection = "first"
    context_missing_trace_policy = "empty"
    context_overflow_policy = "truncate_prefix_only"
    context_first_sample_dump_pending = context_distillation_enabled
    context_teacher_max_sequence_length = master_config["teacher"].get(
        "max_total_sequence_length",
        master_config["policy"]["max_total_sequence_length"],
    )
    resource_isolation_cfg = master_config["distillation"].get("resource_isolation")
    teacher_should_stay_resident = bool(
        resource_isolation_cfg
        and resource_isolation_cfg.get("enabled", False)
        and resource_isolation_cfg.get("teacher_resident_on_gpu", True)
    )
    student_dp_size = _get_data_parallel_size(student_policy)
    teacher_dp_size = _get_data_parallel_size(teacher_policy)
    expected_step_batch_size = (
        master_config["distillation"]["num_prompts_per_step"]
        * master_config["distillation"]["num_generations_per_prompt"]
    )
    configured_train_gbs = master_config["policy"]["train_global_batch_size"]
    if configured_train_gbs % max(student_dp_size, 1) != 0:
        raise ValueError(
            "policy.train_global_batch_size must be divisible by student data_parallel size. "
            f"Got train_global_batch_size={configured_train_gbs}, data_parallel={student_dp_size}."
        )
    if expected_step_batch_size % max(student_dp_size, 1) != 0:
        raise ValueError(
            "distillation step batch must be divisible by student data_parallel size. "
            "Set distillation.num_prompts_per_step * distillation.num_generations_per_prompt "
            "to a multiple of DP. "
            f"Got step_batch={expected_step_batch_size}, data_parallel={student_dp_size}."
        )

    if context_distillation_enabled:
        assert context_distillation_cfg is not None
        if context_distillation_cfg.get("mode", "self_frozen") != "self_frozen":
            raise ValueError(
                "distillation.context_distillation.mode must be 'self_frozen' for V1."
            )
        if (
            context_distillation_cfg.get("alignment", "response_token_index")
            != "response_token_index"
        ):
            raise ValueError(
                "distillation.context_distillation.alignment must be 'response_token_index' for V1."
            )
        context_problem_source = context_distillation_cfg.get(
            "problem_source", context_problem_source
        )
        (
            context_trace_source,
            context_static_trace_answer_column,
        ) = _get_context_trace_source_and_static_column(context_distillation_cfg)
        context_teacher_prefix_template = context_distillation_cfg.get(
            "teacher_prefix_template",
            "You are given a problem and a trace: {problem} + {trace}. Solve it on your own.",
        )
        trace_cfg = context_distillation_cfg.get("trace_extractor", {})
        context_trace_type = trace_cfg.get("type", context_trace_type)
        context_trace_selection = trace_cfg.get("selection", context_trace_selection)
        context_missing_trace_policy = trace_cfg.get(
            "missing_trace_policy", context_missing_trace_policy
        )
        context_overflow_policy = context_distillation_cfg.get(
            "overflow_policy", context_overflow_policy
        )
        context_metrics_cfg = context_distillation_cfg.get("metrics", {})
        context_metrics_enabled = context_metrics_cfg.get("enabled", True)

    context_runtime_cfg: ContextDistillationRuntimeConfig = {
        "enabled": context_distillation_enabled,
        "problem_source": context_problem_source,
        "trace_source": context_trace_source,
        "static_trace_answer_column": context_static_trace_answer_column,
        "teacher_prefix_template": context_teacher_prefix_template,
        "trace_type": context_trace_type,
        "trace_selection": context_trace_selection,
        "missing_trace_policy": context_missing_trace_policy,
        "overflow_policy": context_overflow_policy,
        "metrics_enabled": context_metrics_enabled,
        "teacher_max_sequence_length": context_teacher_max_sequence_length,
    }

    if teacher_should_stay_resident:
        print(
            "▶ Preparing teacher logprob inference once (resident mode)...",
            flush=True,
        )
        teacher_policy.prepare_for_lp_inference()

    # Run distillation training (multi-epoch until reaching max_num_steps or max_num_epochs)
    batch: BatchedDataDict[DatumSpec]

    try:
        # Run validation at the start if configured
        if val_at_start and total_steps == 0:
            print("\n🔍 Running initial validation...", flush=True)
            if NEED_REFIT and POLICY_GENERATION_STALE:
                refit_policy_generation(
                    student_policy, student_generation, colocated_inference
                )
                POLICY_GENERATION_STALE = False
            else:
                student_generation.prepare_for_generation()
            val_metrics, validation_timings = validate(
                student_generation,
                val_dataloader,
                tokenizer,
                val_task_to_env,
                step=total_steps,
                master_config=master_config,
                logger=logger,
                student_policy=student_policy,
                teacher_policy=teacher_policy,
                loss_fn=loss_fn,
                teacher_should_stay_resident=teacher_should_stay_resident,
                teacher_dp_size=teacher_dp_size,
                context_runtime_cfg=context_runtime_cfg,
            )
            student_generation.finish_generation()
            logger.log_metrics(val_metrics, total_steps, prefix="validation")
            logger.log_metrics(
                validation_timings, total_steps, prefix="timing/validation"
            )

        while total_steps < max_steps and current_epoch < max_epochs:
            print(
                f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
                flush=True,
            )

            for batch in dataloader:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                    flush=True,
                )
                maybe_gpu_profile_step(student_policy, total_steps + 1)
                if student_policy != student_generation:
                    maybe_gpu_profile_step(student_generation, total_steps + 1)
                val_metrics, validation_timings = None, None

                with timer.time("total_step_time"):
                    # Prepare batch
                    print("▶ Preparing batch...", flush=True)
                    with timer.time("data_processing"):
                        # Repeat batch items
                        repeated_batch: BatchedDataDict[DatumSpec] = (
                            batch.repeat_interleave(
                                master_config["distillation"][
                                    "num_generations_per_prompt"
                                ]
                            )
                        )
                    # Generate responses - this updates the LLMMessageLogType in repeated_batch
                    print(
                        f"▶ Generating responses for batch of size {repeated_batch.size}...",
                        flush=True,
                    )
                    with timer.time("prepare_for_generation"):
                        if NEED_REFIT and POLICY_GENERATION_STALE:
                            refit_policy_generation(
                                student_policy,
                                student_generation,
                                colocated_inference,
                                timer=timer,
                            )
                            POLICY_GENERATION_STALE = False
                        else:
                            student_generation.prepare_for_generation()

                    with timer.time("generation"):
                        # Use async rollouts if vLLM async engine is enabled
                        if _should_use_async_rollouts(master_config):
                            (
                                repeated_batch,
                                rollout_metrics,
                            ) = run_async_multi_turn_rollout(
                                policy_generation=student_generation,
                                input_batch=repeated_batch,
                                tokenizer=tokenizer,
                                task_to_env=task_to_env,
                                max_seq_len=master_config["policy"][
                                    "max_total_sequence_length"
                                ],
                                max_rollout_turns=master_config["distillation"][
                                    "max_rollout_turns"
                                ],
                                greedy=False,
                            )
                        else:
                            repeated_batch, rollout_metrics = run_multi_turn_rollout(
                                policy_generation=student_generation,
                                input_batch=repeated_batch,
                                tokenizer=tokenizer,
                                task_to_env=task_to_env,
                                max_seq_len=master_config["policy"][
                                    "max_total_sequence_length"
                                ],
                                max_rollout_turns=master_config["distillation"][
                                    "max_rollout_turns"
                                ],
                                greedy=False,
                            )
                        student_generation.finish_generation()

                    with timer.time("data_processing"):
                        train_data, flat_messages, input_lengths = (
                            _build_train_data_from_rollout(
                                repeated_batch=repeated_batch,
                                tokenizer=tokenizer,
                                make_sequence_length_divisible_by=master_config[
                                    "policy"
                                ]["make_sequence_length_divisible_by"],
                            )
                        )

                    print("▶ Preparing for teacher logprob inference...", flush=True)
                    with timer.time("teacher_logprob_inference_prep"):
                        if not teacher_should_stay_resident:
                            teacher_policy.prepare_for_lp_inference()

                    print("▶ Computing teacher logprobs...", flush=True)
                    with timer.time("teacher_logprob_inference"):
                        extra_env_infos = _get_required_extra_env_infos(repeated_batch)
                        context_step_metrics, debug_dump_printed = (
                            _populate_teacher_topk_for_train_data(
                                train_data=train_data,
                                repeated_batch=repeated_batch,
                                extra_env_infos=extra_env_infos,
                                teacher_policy=teacher_policy,
                                tokenizer=tokenizer,
                                topk_k=master_config["distillation"]["topk_logits_k"],
                                timer=timer,
                                context_runtime_cfg=context_runtime_cfg,
                                teacher_dp_size=teacher_dp_size,
                                debug_print_first_sample=context_first_sample_dump_pending,
                            )
                        )
                        if debug_dump_printed:
                            context_first_sample_dump_pending = False

                    print("▶ Preparing for training...", flush=True)
                    with timer.time("training_prep"):
                        if not teacher_should_stay_resident:
                            teacher_policy.offload_after_refit()
                        # set model train and reload optim to GPU
                        student_policy.prepare_for_training()
                        POLICY_GENERATION_STALE = True

                    print("▶ Training policy...", flush=True)
                    with timer.time("policy_training"):
                        train_results = student_policy.train(
                            train_data,
                            loss_fn,
                            timer=timer,
                        )

                    is_last_step = (total_steps + 1 >= max_steps) or (
                        (current_epoch + 1 == max_epochs)
                        and (current_step + 1 == len(dataloader))
                    )

                    # Run validation if it's a validation step or last step with val_at_end
                    if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                        val_at_end and is_last_step
                    ):
                        if NEED_REFIT and POLICY_GENERATION_STALE:
                            refit_policy_generation(
                                student_policy, student_generation, colocated_inference
                            )
                            POLICY_GENERATION_STALE = False
                        else:
                            student_generation.prepare_for_generation()
                        val_metrics, validation_timings = validate(
                            student_generation,
                            val_dataloader,
                            tokenizer,
                            val_task_to_env,
                            step=total_steps + 1,
                            master_config=master_config,
                            logger=logger,
                            student_policy=student_policy,
                            teacher_policy=teacher_policy,
                            loss_fn=loss_fn,
                            teacher_should_stay_resident=teacher_should_stay_resident,
                            teacher_dp_size=teacher_dp_size,
                            context_runtime_cfg=context_runtime_cfg,
                        )
                        student_generation.finish_generation()
                        logger.log_metrics(
                            validation_timings, total_steps + 1, prefix="timing/validation"
                        )
                        logger.log_metrics(
                            val_metrics, total_steps + 1, prefix="validation"
                        )

                    metrics = _finalize_distillation_step_metrics(
                        train_results=train_results,
                        repeated_batch=repeated_batch,
                        input_lengths=input_lengths,
                        rollout_metrics=rollout_metrics,
                        context_step_metrics=context_step_metrics,
                    )
                    total_valid_tokens += metrics.get("global_valid_toks", 0.0)

                    ## Checkpointing
                    consumed_samples += master_config["distillation"][
                        "num_prompts_per_step"
                    ]
                    timeout.mark_iteration()

                    should_save_by_step = (
                        is_last_step
                        or (total_steps + 1)
                        % master_config["checkpointing"]["save_period"]
                        == 0
                    )
                    # +1 because total_steps is 0-indexed
                    # Check if timeout-based checkpointing is enabled in config.
                    should_save_by_timeout = timeout.check_save()

                    if master_config["checkpointing"]["enabled"] and (
                        should_save_by_step or should_save_by_timeout
                    ):
                        student_policy.prepare_for_training()

                        distillation_save_state["current_epoch"] = current_epoch
                        distillation_save_state["current_step"] = current_step + 1
                        distillation_save_state["total_steps"] = total_steps + 1
                        distillation_save_state["total_valid_tokens"] = total_valid_tokens
                        if val_metrics is not None:
                            distillation_save_state["val_reward"] = val_metrics[
                                "accuracy"
                            ]
                        elif "val_reward" in distillation_save_state:
                            del distillation_save_state["val_reward"]
                        distillation_save_state["consumed_samples"] = consumed_samples

                        full_metric_name = master_config["checkpointing"]["metric_name"]
                        if full_metric_name is not None:
                            assert full_metric_name.startswith(
                                "train:"
                            ) or full_metric_name.startswith("val:"), (
                                f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                                f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                                f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                                f" e.g. 'val_reward --> 'val:accuracy'"
                            )
                            prefix, metric_name = full_metric_name.split(":", 1)
                            metrics_source = metrics if prefix == "train" else val_metrics
                            if not metrics_source:
                                warnings.warn(
                                    f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                    "This checkpoint will not be saved as top-k.",
                                    stacklevel=2,
                                )
                                if full_metric_name in distillation_save_state:
                                    del distillation_save_state[full_metric_name]
                            elif metric_name not in metrics_source:
                                raise ValueError(
                                    f"Metric {metric_name} not found in {prefix} metrics"
                                )
                            else:
                                distillation_save_state[full_metric_name] = (
                                    metrics_source[metric_name]
                                )

                        with timer.time("checkpointing"):
                            print(
                                f"Saving checkpoint for step {total_steps + 1}...",
                                flush=True,
                            )
                            checkpoint_path = checkpointer.init_tmp_checkpoint(
                                total_steps + 1, distillation_save_state, master_config
                            )
                            student_policy.save_checkpoint(
                                weights_path=os.path.join(
                                    checkpoint_path, "policy", "weights"
                                ),
                                optimizer_path=os.path.join(
                                    checkpoint_path, "policy", "optimizer"
                                ),
                                tokenizer_path=os.path.join(
                                    checkpoint_path, "policy", "tokenizer"
                                ),
                                checkpointing_cfg=master_config["checkpointing"],
                            )
                            torch.save(
                                dataloader.state_dict(),
                                os.path.join(checkpoint_path, "train_dataloader.pt"),
                            )
                            checkpointer.finalize_checkpoint(checkpoint_path)

                # Logging
                # Log training data
                log_data = {"content": flat_messages["content"]}
                log_data["input_lengths"] = input_lengths.tolist()
                logger.log_batched_dict_as_jsonl(
                    log_data, f"train_data_step{total_steps + 1}.jsonl"
                )

                timing_metrics: dict[str, float] = timer.get_timing_metrics(
                    reduction_op="sum"
                )  # type: ignore

                print("\n📊 Training Results:")

                print(f"  • Loss: {metrics['loss']:.4f}")
                print(
                    f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
                )
                if "total_flops" in train_results:
                    total_tflops = (
                        train_results["total_flops"]
                        / timing_metrics["policy_training"]
                        / 1e12
                    )
                    num_ranks = train_results["num_ranks"]
                    print(
                        f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                        flush=True,
                    )
                    if "theoretical_tflops" in train_results:
                        theoretical_tflops = train_results["theoretical_tflops"]
                        print(
                            f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                            flush=True,
                        )
                        metrics["train_fp_utilization"] = (
                            total_tflops / theoretical_tflops
                        )

                print("\n⏱️  Timing:", flush=True)
                # Display total time first, separately
                total_time = timing_metrics.get("total_step_time", 0)

                total_num_gpus = (
                    master_config["cluster"]["num_nodes"]
                    * master_config["cluster"]["gpus_per_node"]
                )
                metrics.update(
                    {
                        "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                        / total_time
                        / total_num_gpus
                    }
                )

                print(f"  • Total step time: {total_time:.2f}s", flush=True)

                # Display all other timing metrics
                for k, v in sorted(
                    timing_metrics.items(), key=lambda item: item[1], reverse=True
                ):
                    if k != "total_step_time":
                        percent = (v / total_time * 100) if total_time > 0 else 0
                        print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    metrics["global_valid_toks"] / total_time / total_num_gpus
                )
                logger.log_metrics(metrics, total_steps + 1, prefix="train")
                logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

                timer.reset()
                current_step += 1
                total_steps += 1
                if should_save_by_timeout:
                    print("Timeout has been reached, stopping training early", flush=True)
                    return
                if total_steps >= max_steps:
                    print(
                        "Max number of steps has been reached, stopping training early",
                        flush=True,
                    )
                    return

        # End of epoch
        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
    finally:
        if teacher_should_stay_resident:
            print(
                "▶ Offloading teacher policy after training loop teardown...",
                flush=True,
            )
            teacher_policy.offload_after_refit()


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
    student_policy: Optional[ColocatablePolicyInterface] = None,
    teacher_policy: Optional[ColocatablePolicyInterface] = None,
    loss_fn: Optional[DistillationLossFn] = None,
    teacher_should_stay_resident: bool = False,
    teacher_dp_size: int = 1,
    context_runtime_cfg: Optional[ContextDistillationRuntimeConfig] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    if val_task_to_env is None:
        print(
            "  ⚠️ No validation task to environment mapping provided, skipping validation",
            flush=True,
        )
        return {}, {}

    track_train_metrics = bool(
        master_config["distillation"].get("val_track_train_metrics", False)
    )
    if track_train_metrics:
        if student_policy is None or teacher_policy is None or loss_fn is None:
            raise ValueError(
                "Validation train-metric tracking requires student_policy, "
                "teacher_policy, and loss_fn."
            )
        if context_runtime_cfg is None:
            context_distillation_cfg = master_config["distillation"].get(
                "context_distillation"
            )
            trace_cfg = (
                context_distillation_cfg.get("trace_extractor", {})
                if context_distillation_cfg
                else {}
            )
            metrics_cfg = (
                context_distillation_cfg.get("metrics", {})
                if context_distillation_cfg
                else {}
            )
            if context_distillation_cfg:
                (
                    context_trace_source,
                    context_static_trace_answer_column,
                ) = _get_context_trace_source_and_static_column(context_distillation_cfg)
            else:
                context_trace_source = "dynamic"
                context_static_trace_answer_column = ""
            context_runtime_cfg = {
                "enabled": bool(
                    context_distillation_cfg
                    and context_distillation_cfg.get("enabled", False)
                ),
                "problem_source": (
                    context_distillation_cfg.get(
                        "problem_source", "original_user_problem"
                    )
                    if context_distillation_cfg
                    else "original_user_problem"
                ),
                "trace_source": context_trace_source,
                "static_trace_answer_column": context_static_trace_answer_column,
                "teacher_prefix_template": (
                    context_distillation_cfg.get(
                        "teacher_prefix_template",
                        "You are given a problem and a trace: {problem} + {trace}. Solve it on your own.",
                    )
                    if context_distillation_cfg
                    else "You are given a problem and a trace: {problem} + {trace}. Solve it on your own."
                ),
                "trace_type": trace_cfg.get("type", "think_tag"),
                "trace_selection": trace_cfg.get("selection", "first"),
                "missing_trace_policy": trace_cfg.get("missing_trace_policy", "empty"),
                "overflow_policy": (
                    context_distillation_cfg.get(
                        "overflow_policy", "truncate_prefix_only"
                    )
                    if context_distillation_cfg
                    else "truncate_prefix_only"
                ),
                "metrics_enabled": metrics_cfg.get("enabled", True),
                "teacher_max_sequence_length": master_config["teacher"].get(
                    "max_total_sequence_length",
                    master_config["policy"]["max_total_sequence_length"],
                ),
            }

    max_val_samples = master_config["distillation"]["max_val_samples"]
    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards = []  # Can be any metric. Setted to 'accuracy' by default.
        total_generated_tokens = 0.0
        all_message_logs = []  # Collect all message logs
        distill_validation_metrics_by_batch: list[dict[str, Any]] = []

        if track_train_metrics and not teacher_should_stay_resident:
            assert teacher_policy is not None
            with timer.time("validation_teacher_logprob_inference_prep"):
                teacher_policy.prepare_for_lp_inference()

        for val_batch in val_dataloader:
            remaining_samples = max_val_samples - len(total_rewards)
            if remaining_samples <= 0:
                break
            if val_batch.size > remaining_samples:
                val_batch = val_batch.slice(0, remaining_samples)

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            if _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            rewards = val_batch["total_reward"]
            batch_size = len(rewards)

            total_rewards.extend(rewards.tolist())
            total_generated_tokens += (
                float(gen_metrics["mean_gen_tokens_per_sample"]) * batch_size
            )

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)

            if track_train_metrics:
                assert student_policy is not None
                assert teacher_policy is not None
                assert loss_fn is not None
                assert context_runtime_cfg is not None

                with timer.time("validation_data_processing"):
                    val_train_data, _, val_input_lengths = _build_train_data_from_rollout(
                        repeated_batch=val_batch,
                        tokenizer=tokenizer,
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                with timer.time("validation_teacher_logprob_inference"):
                    val_extra_env_infos = _get_required_extra_env_infos(val_batch)
                    val_context_step_metrics, _ = _populate_teacher_topk_for_train_data(
                        train_data=val_train_data,
                        repeated_batch=val_batch,
                        extra_env_infos=val_extra_env_infos,
                        teacher_policy=teacher_policy,
                        tokenizer=tokenizer,
                        topk_k=master_config["distillation"]["topk_logits_k"],
                        timer=timer,
                        context_runtime_cfg=context_runtime_cfg,
                        teacher_dp_size=teacher_dp_size,
                        debug_print_first_sample=False,
                    )

                with timer.time("validation_policy_training"):
                    student_policy.prepare_for_training()
                    val_train_results = student_policy.train(
                        val_train_data,
                        loss_fn,
                        eval_mode=True,
                        timer=timer,
                    )

                val_step_metrics = _finalize_distillation_step_metrics(
                    train_results=val_train_results,
                    repeated_batch=val_batch,
                    input_lengths=val_input_lengths,
                    rollout_metrics=gen_metrics,
                    context_step_metrics=val_context_step_metrics,
                )
                distill_validation_metrics_by_batch.append(val_step_metrics)

                if policy_generation is student_policy:
                    with timer.time("validation_prepare_for_generation"):
                        policy_generation.prepare_for_generation()

        if track_train_metrics and not teacher_should_stay_resident:
            assert teacher_policy is not None
            with timer.time("validation_teacher_offload"):
                teacher_policy.offload_after_refit()

        # Calculate validation metrics
        accuracy = (
            sum(total_rewards) / len(total_rewards) if len(total_rewards) > 0 else 0
        )
        avg_length = total_generated_tokens / len(total_rewards) if total_rewards else 0

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }
        if track_train_metrics:
            val_metrics.update(
                _mean_aggregate_metric_dicts(distill_validation_metrics_by_batch)
            )

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)
    if track_train_metrics and "loss" in val_metrics:
        print(f"    • Distillation loss: {val_metrics['loss']:.4f}", flush=True)

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    # Persist validation samples similarly to training samples for later inspection.
    if logger is not None and all_message_logs:
        val_log_data = {
            "content": all_message_logs,
            "rewards": total_rewards,
        }
        logger.log_batched_dict_as_jsonl(val_log_data, f"val_data_step{step}.jsonl")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
