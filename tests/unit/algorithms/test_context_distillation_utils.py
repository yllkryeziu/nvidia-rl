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

import torch

from nemo_rl.algorithms.context_distillation_utils import (
    align_teacher_topk_to_student_positions,
    build_context_distillation_teacher_batch,
    extract_first_think_span,
)


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1

    def __init__(self):
        self.last_chat_payload = None

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        tokens = [tok for tok in text.split(" ") if tok]
        token_ids = torch.arange(1, len(tokens) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": token_ids}

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert add_special_tokens is False
        self.last_chat_payload = messages[0]["content"]
        return f"USER {messages[0]['content']} ASSISTANT"


def _message(role: str, content: str, token_ids: list[int]) -> dict[str, object]:
    return {
        "role": role,
        "content": content,
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
    }


def test_extract_first_think_span():
    text = "<think>first trace</think> answer <think>second</think>"
    assert extract_first_think_span(text) == "first trace"
    assert extract_first_think_span("no think tags here") == ""


def test_build_teacher_batch_and_align_topk():
    tokenizer = DummyTokenizer()
    message_logs = [
        [
            _message("user", "solve x", [10, 11]),
            _message("assistant", "<think>trace</think> final", [20, 21, 22]),
        ]
    ]
    sample_mask = torch.tensor([1.0], dtype=torch.float32)
    student_input_lengths = torch.tensor([5], dtype=torch.long)

    result = build_context_distillation_teacher_batch(
        message_logs=message_logs,  # type: ignore[arg-type]
        sample_mask=sample_mask,
        student_input_lengths=student_input_lengths,
        tokenizer=tokenizer,
        teacher_prefix_template="P {problem} T {trace}",
        max_teacher_sequence_length=32,
        pad_token_id=tokenizer.pad_token_id,
        extra_env_infos=[{"problem": "solve x"}],
    )

    assert result.teacher_data is not None
    assert result.valid_sample_indices == [0]
    assert len(result.alignments) == 1
    alignment = result.alignments[0]
    assert alignment.student_pred_start == 1  # response starts at token index 2
    assert alignment.num_response_tokens == 3
    assert result.metrics["context_distillation_trace_coverage"] == 1.0

    teacher_lengths = result.teacher_data["input_lengths"]
    teacher_seq_len = int(teacher_lengths[0].item())
    topk = 2
    teacher_topk_logits = torch.zeros((1, teacher_seq_len, topk), dtype=torch.float32)
    teacher_topk_indices = torch.zeros((1, teacher_seq_len, topk), dtype=torch.long)
    for row in range(teacher_seq_len):
        teacher_topk_logits[0, row, :] = torch.tensor([row, row + 0.5])
        teacher_topk_indices[0, row, :] = torch.tensor([100 + row, 200 + row])

    aligned_logits, aligned_indices = align_teacher_topk_to_student_positions(
        teacher_topk_logits=teacher_topk_logits,
        teacher_topk_indices=teacher_topk_indices,
        alignments=result.alignments,
        valid_sample_indices=result.valid_sample_indices,
        student_batch_size=1,
        student_sequence_length=5,
        topk=topk,
    )

    # Response has 3 tokens, so student prediction rows 1..3 should be filled.
    for offset in range(3):
        teacher_row = alignment.teacher_pred_start + offset
        student_row = alignment.student_pred_start + offset
        assert torch.equal(
            aligned_logits[0, student_row, :], teacher_topk_logits[0, teacher_row, :]
        )
        assert torch.equal(
            aligned_indices[0, student_row, :], teacher_topk_indices[0, teacher_row, :]
        )

    assert torch.equal(aligned_logits[0, 0, :], torch.zeros(topk))
    assert torch.equal(aligned_indices[0, 0, :], torch.zeros(topk, dtype=torch.long))


def test_build_teacher_batch_truncation_and_drop_too_long_response():
    tokenizer = DummyTokenizer()
    message_logs = [
        [
            _message("user", "u v", [1, 2]),
            _message("assistant", "<think>a b c d e</think> ans", [3, 4, 5]),
        ],
        [
            _message("user", "q", [1]),
            _message("assistant", "<think>x</think> y", [2, 3, 4, 5, 6, 7]),
        ],
    ]

    result = build_context_distillation_teacher_batch(
        message_logs=message_logs,  # type: ignore[arg-type]
        sample_mask=torch.tensor([1.0, 1.0], dtype=torch.float32),
        student_input_lengths=torch.tensor([5, 7], dtype=torch.long),
        tokenizer=tokenizer,
        teacher_prefix_template="tok0 tok1 tok2 tok3 tok4 {problem} {trace}",
        max_teacher_sequence_length=5,
        pad_token_id=tokenizer.pad_token_id,
        extra_env_infos=[{"problem": "u v"}, {"problem": "q"}],
    )

    assert result.valid_sample_indices == [0]
    assert result.sample_mask.tolist() == [1.0, 0.0]
    assert result.metrics["context_distillation_prefix_truncation_count"] == 1.0
    assert result.metrics["context_distillation_dropped_too_long_response"] == 1.0


def test_build_teacher_batch_static_trace_source_uses_dataset_answer():
    tokenizer = DummyTokenizer()
    message_logs = [
        [
            _message("user", "solve x", [10, 11]),
            _message(
                "assistant",
                "<think>live trace</think> final",
                [20, 21, 22],
            ),
        ]
    ]

    result = build_context_distillation_teacher_batch(
        message_logs=message_logs,  # type: ignore[arg-type]
        sample_mask=torch.tensor([1.0], dtype=torch.float32),
        student_input_lengths=torch.tensor([5], dtype=torch.long),
        tokenizer=tokenizer,
        teacher_prefix_template="P {problem} T {trace}",
        max_teacher_sequence_length=64,
        pad_token_id=tokenizer.pad_token_id,
        extra_env_infos=[
            {
                "problem": "solve x",
                "qwen3_1b7_original_answer": "<think>dataset trace words</think> ans",
            }
        ],
        trace_source="static_dataset",
        static_trace_answer_column="qwen3_1b7_original_answer",
    )

    assert result.teacher_data is not None
    assert tokenizer.last_chat_payload is not None
    assert "dataset trace words" in tokenizer.last_chat_payload
    assert "live trace" not in tokenizer.last_chat_payload


def test_build_teacher_batch_static_trace_missing_drops_sample():
    tokenizer = DummyTokenizer()
    message_logs = [
        [
            _message("user", "solve x", [10, 11]),
            _message("assistant", "<think>live trace</think> final", [20, 21, 22]),
        ]
    ]

    result = build_context_distillation_teacher_batch(
        message_logs=message_logs,  # type: ignore[arg-type]
        sample_mask=torch.tensor([1.0], dtype=torch.float32),
        student_input_lengths=torch.tensor([5], dtype=torch.long),
        tokenizer=tokenizer,
        teacher_prefix_template="P {problem} T {trace}",
        max_teacher_sequence_length=64,
        pad_token_id=tokenizer.pad_token_id,
        extra_env_infos=[{"problem": "solve x"}],
        trace_source="static_dataset",
        static_trace_answer_column="qwen3_1b7_original_answer",
        missing_trace_policy="drop_sample",
    )

    assert result.teacher_data is None
    assert result.valid_sample_indices == []
    assert result.sample_mask.tolist() == [0.0]
    assert result.metrics["context_distillation_dropped_missing_trace"] == 1.0


def test_build_teacher_batch_static_trace_no_think_drops_sample():
    tokenizer = DummyTokenizer()
    message_logs = [
        [
            _message("user", "solve x", [10, 11]),
            _message("assistant", "<think>live trace</think> final", [20, 21, 22]),
        ]
    ]

    result = build_context_distillation_teacher_batch(
        message_logs=message_logs,  # type: ignore[arg-type]
        sample_mask=torch.tensor([1.0], dtype=torch.float32),
        student_input_lengths=torch.tensor([5], dtype=torch.long),
        tokenizer=tokenizer,
        teacher_prefix_template="P {problem} T {trace}",
        max_teacher_sequence_length=64,
        pad_token_id=tokenizer.pad_token_id,
        extra_env_infos=[
            {"problem": "solve x", "qwen3_1b7_original_answer": "no think tags here"}
        ],
        trace_source="static_dataset",
        static_trace_answer_column="qwen3_1b7_original_answer",
        missing_trace_policy="drop_sample",
    )

    assert result.teacher_data is None
    assert result.valid_sample_indices == []
    assert result.sample_mask.tolist() == [0.0]
    assert result.metrics["context_distillation_dropped_missing_trace"] == 1.0
