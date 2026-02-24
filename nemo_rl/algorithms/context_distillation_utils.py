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
    debug_dump_printed: bool


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


def _build_teacher_chat_prefix_text(
    *, tokenizer: PreTrainedTokenizerBase, teacher_prefix_payload: str
) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None or not callable(apply_chat_template):
        raise ValueError(
            "Context distillation requires tokenizer.apply_chat_template to build the "
            "teacher scoring context."
        )

    try:
        formatted = apply_chat_template(
            [{"role": "user", "content": teacher_prefix_payload}],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
    except TypeError as e:
        raise ValueError(
            "Context distillation teacher chat prefix creation failed. The tokenizer "
            "must support apply_chat_template(..., tokenize=False, "
            "add_generation_prompt=True, add_special_tokens=False)."
        ) from e

    if isinstance(formatted, list):
        if len(formatted) != 1 or not isinstance(formatted[0], str):
            raise ValueError(
                "Context distillation expected a single chat-formatted prefix string "
                f"but got list output of length {len(formatted)}."
            )
        formatted = formatted[0]

    if not isinstance(formatted, str):
        raise ValueError(
            "Context distillation expected tokenizer.apply_chat_template(..., "
            f"tokenize=False) to return str, got {type(formatted).__name__}."
        )

    return formatted


def _extract_problem_from_extra_env_info(extra_env_info: Any) -> str:
    if not isinstance(extra_env_info, dict):
        raise ValueError(
            "Context distillation requires extra_env_info to be a dict containing 'problem'."
        )
    problem = extra_env_info.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(
            "Context distillation requires extra_env_info['problem'] to be a non-empty string."
        )
    return problem


def _extract_string_from_extra_env_info(
    extra_env_info: Any, key: str, *, require_non_empty: bool = True
) -> str:
    if not isinstance(extra_env_info, dict):
        raise ValueError(
            "Context distillation requires extra_env_info to be a dict when "
            f"reading '{key}'."
        )
    value = extra_env_info.get(key)
    if not isinstance(value, str):
        return ""
    if require_non_empty and not value.strip():
        return ""
    return value


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
    extra_env_infos: list[dict[str, Any] | None],
    problem_source: str = "original_user_problem",
    trace_source: str = "dynamic",
    static_trace_answer_column: str = "",
    trace_extractor_type: str = "think_tag",
    trace_extractor_selection: str = "first",
    missing_trace_policy: str = "empty",
    overflow_policy: str = "truncate_prefix_only",
    metrics_enabled: bool = True,
    debug_print_first_sample: bool = False,
) -> ContextDistillationBuildResult:
    if problem_source != "original_user_problem":
        raise ValueError(f"Unsupported problem_source for V1: {problem_source}")
    if trace_source not in {"dynamic", "static_dataset"}:
        raise ValueError(f"Unsupported trace_source for context distillation: {trace_source}")
    if trace_source == "static_dataset" and not static_trace_answer_column.strip():
        raise ValueError(
            "Context distillation trace_source='static_dataset' requires a non-empty "
            "static_trace_answer_column."
        )
    if trace_extractor_type != "think_tag":
        raise ValueError(f"Unsupported trace extractor type for V1: {trace_extractor_type}")
    if trace_extractor_selection != "first":
        raise ValueError(
            f"Unsupported trace extractor selection for V1: {trace_extractor_selection}"
        )
    if missing_trace_policy not in {"empty", "drop_sample"}:
        raise ValueError(
            "Unsupported missing_trace_policy for context distillation: "
            f"{missing_trace_policy}"
        )
    if overflow_policy != "truncate_prefix_only":
        raise ValueError(f"Unsupported overflow_policy for V1: {overflow_policy}")
    if max_teacher_sequence_length <= 0:
        raise ValueError("max_teacher_sequence_length must be positive.")
    if len(extra_env_infos) != len(message_logs):
        raise ValueError(
            "extra_env_infos length must match message_logs length for context distillation."
        )

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
    dropped_missing_trace = 0
    debug_dump_printed = False

    if debug_print_first_sample and num_samples > 0:
        sample0_mask = float(sample_mask[0].item()) if sample_mask.numel() > 0 else float("nan")
        try:
            sample0_problem = _extract_problem_from_extra_env_info(extra_env_infos[0])
            sample0_response_text, _, _ = _extract_last_assistant_payload(message_logs[0])
            sample0_trace_source_text = sample0_response_text
            sample0_trace_source_label = "STUDENT TRACE SOURCE TEXT (LIVE RESPONSE)"
            if trace_source == "static_dataset":
                sample0_trace_source_text = _extract_string_from_extra_env_info(
                    extra_env_infos[0],
                    static_trace_answer_column,
                )
                sample0_trace_source_label = (
                    "STATIC TRACE SOURCE TEXT "
                    f"(extra_env_info['{static_trace_answer_column}'])"
                )
            sample0_trace = extract_first_think_span(sample0_trace_source_text)
            sample0_teacher_payload = teacher_prefix_template.format(
                problem=sample0_problem, trace=sample0_trace
            )
            sample0_teacher_chat_prefix = _build_teacher_chat_prefix_text(
                tokenizer=tokenizer,
                teacher_prefix_payload=sample0_teacher_payload,
            )
            print(
                (
                    "\n===== CONTEXT DISTILLATION FIRST-SAMPLE DUMP =====\n"
                    "sample_idx: 0\n"
                    f"sample_mask: {sample0_mask}\n\n"
                    "1) WHOLE STUDENT GENERATION\n"
                    f"{sample0_response_text}\n\n"
                    f"2) {sample0_trace_source_label}\n"
                    f"{sample0_trace_source_text}\n\n"
                    "3) EXTRACTED TRACE\n"
                    f"{sample0_trace}\n\n"
                    "4) RAW TEACHER PROMPT PAYLOAD (PROBLEM + EXTRACTED TRACE)\n"
                    f"{sample0_teacher_payload}\n\n"
                    "5) CHAT-FORMATTED TEACHER PREFIX (ACTUAL CONTEXT TOKENIZED)\n"
                    f"{sample0_teacher_chat_prefix}\n\n"
                    "6) TEXT THE TEACHER SCORES\n"
                    f"{sample0_response_text}\n"
                    "===== END CONTEXT DISTILLATION FIRST-SAMPLE DUMP =====\n"
                ),
                flush=True,
            )
        except Exception as e:
            print(
                (
                    "\n===== CONTEXT DISTILLATION FIRST-SAMPLE DUMP =====\n"
                    "Failed to build first-sample dump.\n"
                    f"error: {e}\n"
                    "===== END CONTEXT DISTILLATION FIRST-SAMPLE DUMP =====\n"
                ),
                flush=True,
            )
        debug_dump_printed = True

    for sample_idx, message_log in enumerate(message_logs):
        if float(sample_mask_out[sample_idx].item()) == 0.0:
            continue

        try:
            problem = _extract_problem_from_extra_env_info(extra_env_infos[sample_idx])
            response_text, response_tokens, response_start = _extract_last_assistant_payload(
                message_log
            )
        except ValueError:
            sample_mask_out[sample_idx] = 0.0
            dropped_samples += 1
            dropped_invalid_message += 1
            continue

        trace_source_text = response_text
        if trace_source == "static_dataset":
            trace_source_text = _extract_string_from_extra_env_info(
                extra_env_infos[sample_idx],
                static_trace_answer_column,
            )
        trace = extract_first_think_span(trace_source_text)
        if not trace and missing_trace_policy == "drop_sample":
            sample_mask_out[sample_idx] = 0.0
            dropped_samples += 1
            dropped_missing_trace += 1
            continue
        if trace:
            trace_hits += 1
            trace_char_total += len(trace)
            trace_token_total += int(
                _tokenize_text(trace, tokenizer, add_special_tokens=False).numel()
            )

        teacher_prefix_payload = teacher_prefix_template.format(problem=problem, trace=trace)
        teacher_chat_prefix = _build_teacher_chat_prefix_text(
            tokenizer=tokenizer,
            teacher_prefix_payload=teacher_prefix_payload,
        )
        prefix_tokens = _tokenize_text(
            teacher_chat_prefix, tokenizer, add_special_tokens=False
        )

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
            "context_distillation_dropped_missing_trace": float(dropped_missing_trace),
            "context_distillation_valid_samples": float(valid_samples),
        }

    return ContextDistillationBuildResult(
        teacher_data=teacher_data,
        valid_sample_indices=valid_sample_indices,
        alignments=alignments,
        sample_mask=sample_mask_out,
        metrics=metrics,
        debug_dump_printed=debug_dump_printed,
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
