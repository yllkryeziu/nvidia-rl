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
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


@dataclass
class ContextDistillationAlignment:
    sample_idx: int
    # Index in teacher top-k tensor that predicts first response token.
    teacher_pred_start: int
    # Index in student top-k tensor that predicts first response token.
    student_pred_start: int
    # Number of response tokens to align.
    num_response_tokens: int


@dataclass
class ContextDistillationBuildResult:
    teacher_data: BatchedDataDict[GenerationDatumSpec] | None
    valid_sample_indices: list[int]
    alignments: list[ContextDistillationAlignment]
    sample_mask: torch.Tensor
    metrics: dict[str, float]


def extract_first_think_span(text: str) -> str:
    match = THINK_PATTERN.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _tokenize_text(
    text: str, tokenizer: PreTrainedTokenizerBase, *, add_special_tokens: bool = False
) -> torch.Tensor:
    tokenized = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"][0]
    return input_ids.to(dtype=torch.long, device="cpu")


def _extract_problem_text(message_log: LLMMessageLogType) -> str:
    for message in message_log:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("No user message with string content found in message log.")


def _extract_last_assistant_payload(
    message_log: LLMMessageLogType,
) -> tuple[str, torch.Tensor, int]:
    response_content: str | None = None
    response_tokens: torch.Tensor | None = None
    response_start: int | None = None

    running_length = 0
    for message in message_log:
        token_ids = message.get("token_ids")
        if not isinstance(token_ids, torch.Tensor):
            raise ValueError("All messages must contain tensor token_ids.")
        token_ids = token_ids.to(dtype=torch.long, device="cpu").flatten()
        token_count = int(token_ids.numel())

        if message.get("role") == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Assistant content must be a string for V1.")
            response_content = content
            response_tokens = token_ids
            response_start = running_length

        running_length += token_count

    if response_content is None or response_tokens is None or response_start is None:
        raise ValueError("No assistant response found in message log.")
    if int(response_tokens.numel()) == 0:
        raise ValueError("Assistant response is empty.")
    return response_content, response_tokens, response_start


def build_context_distillation_teacher_batch(
    *,
    message_logs: list[LLMMessageLogType],
    sample_mask: torch.Tensor,
    student_input_lengths: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    teacher_prefix_template: str,
    max_teacher_sequence_length: int,
    pad_token_id: int,
    problem_source: str = "original_user_problem",
    trace_extractor_type: str = "think_tag",
    trace_extractor_selection: str = "first",
    missing_trace_policy: str = "empty",
    overflow_policy: str = "truncate_prefix_only",
    metrics_enabled: bool = True,
) -> ContextDistillationBuildResult:
    if problem_source != "original_user_problem":
        raise ValueError(f"Unsupported problem_source for V1: {problem_source}")
    if trace_extractor_type != "think_tag":
        raise ValueError(f"Unsupported trace extractor type for V1: {trace_extractor_type}")
    if trace_extractor_selection != "first":
        raise ValueError(
            f"Unsupported trace extractor selection for V1: {trace_extractor_selection}"
        )
    if missing_trace_policy != "empty":
        raise ValueError(f"Unsupported missing_trace_policy for V1: {missing_trace_policy}")
    if overflow_policy != "truncate_prefix_only":
        raise ValueError(f"Unsupported overflow_policy for V1: {overflow_policy}")
    if max_teacher_sequence_length <= 0:
        raise ValueError("max_teacher_sequence_length must be positive.")

    sample_mask_out = sample_mask.clone()

    teacher_sequences: list[torch.Tensor] = []
    teacher_lengths: list[int] = []
    valid_sample_indices: list[int] = []
    alignments: list[ContextDistillationAlignment] = []

    num_samples = len(message_logs)
    trace_hits = 0
    trace_char_total = 0
    trace_token_total = 0
    prefix_truncation_count = 0
    dropped_samples = 0
    dropped_too_long_response = 0
    dropped_invalid_message = 0
    dropped_no_prefix_budget = 0

    for sample_idx, message_log in enumerate(message_logs):
        if float(sample_mask_out[sample_idx].item()) == 0.0:
            continue

        try:
            problem = _extract_problem_text(message_log)
            response_text, response_tokens, response_start = _extract_last_assistant_payload(
                message_log
            )
        except ValueError:
            sample_mask_out[sample_idx] = 0.0
            dropped_samples += 1
            dropped_invalid_message += 1
            continue

        trace = extract_first_think_span(response_text)
        if trace:
            trace_hits += 1
            trace_char_total += len(trace)
            trace_token_total += int(
                _tokenize_text(trace, tokenizer, add_special_tokens=False).numel()
            )

        prefix_text = teacher_prefix_template.format(problem=problem, trace=trace)
        prefix_tokens = _tokenize_text(prefix_text, tokenizer, add_special_tokens=False)

        response_length = int(response_tokens.numel())
        if response_length > max_teacher_sequence_length:
            sample_mask_out[sample_idx] = 0.0
            dropped_samples += 1
            dropped_too_long_response += 1
            continue

        prefix_budget = max_teacher_sequence_length - response_length
        if prefix_budget <= 0:
            sample_mask_out[sample_idx] = 0.0
            dropped_samples += 1
            dropped_no_prefix_budget += 1
            continue

        if int(prefix_tokens.numel()) > prefix_budget:
            prefix_tokens = prefix_tokens[-prefix_budget:]
            prefix_truncation_count += 1

        if int(prefix_tokens.numel()) == 0:
            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            if bos_token_id is None:
                sample_mask_out[sample_idx] = 0.0
                dropped_samples += 1
                dropped_no_prefix_budget += 1
                continue
            prefix_tokens = torch.tensor([bos_token_id], dtype=torch.long)

        teacher_sequence = torch.cat([prefix_tokens, response_tokens], dim=0)
        teacher_seq_len = int(teacher_sequence.numel())
        student_seq_len = int(student_input_lengths[sample_idx].item())
        student_pred_start = response_start - 1

        if student_pred_start < 0 or response_start + response_length > student_seq_len:
            sample_mask_out[sample_idx] = 0.0
            dropped_samples += 1
            dropped_invalid_message += 1
            continue

        teacher_sequences.append(teacher_sequence)
        teacher_lengths.append(teacher_seq_len)
        valid_sample_indices.append(sample_idx)
        alignments.append(
            ContextDistillationAlignment(
                sample_idx=sample_idx,
                teacher_pred_start=int(prefix_tokens.numel()) - 1,
                student_pred_start=student_pred_start,
                num_response_tokens=response_length,
            )
        )

    teacher_data: BatchedDataDict[GenerationDatumSpec] | None = None
    if teacher_sequences:
        max_len = max(teacher_lengths)
        padded = torch.full(
            (len(teacher_sequences), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        for row_idx, sequence in enumerate(teacher_sequences):
            padded[row_idx, : sequence.numel()] = sequence
        teacher_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": padded,
                "input_lengths": torch.tensor(teacher_lengths, dtype=torch.long),
            }
        )

    metrics: dict[str, float] = {}
    if metrics_enabled and num_samples > 0:
        valid_samples = len(valid_sample_indices)
        trace_denom = max(trace_hits, 1)
        metrics = {
            "context_distillation_trace_coverage": trace_hits / num_samples,
            "context_distillation_trace_mean_chars": trace_char_total / trace_denom,
            "context_distillation_trace_mean_tokens": trace_token_total / trace_denom,
            "context_distillation_prefix_truncation_count": float(
                prefix_truncation_count
            ),
            "context_distillation_prefix_truncation_rate": (
                prefix_truncation_count / max(valid_samples, 1)
            ),
            "context_distillation_dropped_samples": float(dropped_samples),
            "context_distillation_dropped_too_long_response": float(
                dropped_too_long_response
            ),
            "context_distillation_dropped_invalid_message": float(
                dropped_invalid_message
            ),
            "context_distillation_dropped_no_prefix_budget": float(
                dropped_no_prefix_budget
            ),
            "context_distillation_valid_samples": float(valid_samples),
        }

    return ContextDistillationBuildResult(
        teacher_data=teacher_data,
        valid_sample_indices=valid_sample_indices,
        alignments=alignments,
        sample_mask=sample_mask_out,
        metrics=metrics,
    )


def align_teacher_topk_to_student_positions(
    *,
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    alignments: Sequence[ContextDistillationAlignment],
    valid_sample_indices: Sequence[int],
    student_batch_size: int,
    student_sequence_length: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(alignments) != len(valid_sample_indices):
        raise ValueError("alignments and valid_sample_indices must have the same length.")
    if teacher_topk_logits.shape[-1] != topk or teacher_topk_indices.shape[-1] != topk:
        raise ValueError("Teacher top-k tensors do not match configured top-k size.")

    aligned_logits = torch.zeros(
        (student_batch_size, student_sequence_length, topk),
        dtype=teacher_topk_logits.dtype,
        device=teacher_topk_logits.device,
    )
    aligned_indices = torch.zeros(
        (student_batch_size, student_sequence_length, topk),
        dtype=teacher_topk_indices.dtype,
        device=teacher_topk_indices.device,
    )

    for local_idx, sample_idx in enumerate(valid_sample_indices):
        alignment = alignments[local_idx]
        if sample_idx != alignment.sample_idx:
            raise ValueError("Alignment ordering mismatch with valid sample indices.")

        for token_offset in range(alignment.num_response_tokens):
            teacher_row = alignment.teacher_pred_start + token_offset
            student_row = alignment.student_pred_start + token_offset

            if teacher_row < 0 or teacher_row >= teacher_topk_logits.shape[1]:
                raise ValueError(
                    f"Teacher row {teacher_row} out of range for sample {sample_idx}."
                )
            if student_row < 0 or student_row >= student_sequence_length:
                raise ValueError(
                    f"Student row {student_row} out of range for sample {sample_idx}."
                )

            aligned_logits[sample_idx, student_row, :] = teacher_topk_logits[
                local_idx, teacher_row, :
            ]
            aligned_indices[sample_idx, student_row, :] = teacher_topk_indices[
                local_idx, teacher_row, :
            ]

    return aligned_logits, aligned_indices
