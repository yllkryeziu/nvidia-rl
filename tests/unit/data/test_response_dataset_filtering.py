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

from datasets import Dataset
import pytest

import nemo_rl.data.datasets.response_datasets.response_dataset as response_dataset_module
from nemo_rl.data.datasets.response_datasets.response_dataset import ResponseDataset


def test_filter_column_value_keeps_only_matching_rows(monkeypatch):
    raw = Dataset.from_list(
        [
            {"input": "math q1", "output": "math a1", "domain": "math"},
            {"input": "code q1", "output": "code a1", "domain": "code"},
            {"input": "math q2", "output": "math a2", "domain": "math"},
        ]
    )
    monkeypatch.setattr(response_dataset_module, "load_dataset_from_path", lambda *_: raw)

    dataset = ResponseDataset(
        data_path="open-thoughts/OpenThoughts-114k",
        input_key="input",
        output_key="output",
        subset="metadata",
        split="train",
        filter_column="domain",
        filter_value="math",
    )

    assert len(dataset.dataset) == 2
    first = dataset.dataset[0]
    second = dataset.dataset[1]
    assert first["messages"][0]["content"] == "math q1"
    assert second["messages"][0]["content"] == "math q2"


@pytest.mark.parametrize(
    ("filter_column", "filter_value"),
    [("domain", None), (None, "math")],
)
def test_filter_requires_both_column_and_value(
    monkeypatch, filter_column, filter_value
):
    raw = Dataset.from_list([{"input": "q", "output": "a", "domain": "math"}])
    monkeypatch.setattr(response_dataset_module, "load_dataset_from_path", lambda *_: raw)

    with pytest.raises(
        ValueError, match="Both filter_column and filter_value must be provided together."
    ):
        ResponseDataset(
            data_path="open-thoughts/OpenThoughts-114k",
            input_key="input",
            output_key="output",
            subset="metadata",
            split="train",
            filter_column=filter_column,
            filter_value=filter_value,
        )


def test_filter_column_missing_raises(monkeypatch):
    raw = Dataset.from_list([{"input": "q", "output": "a", "source": "numina_math"}])
    monkeypatch.setattr(response_dataset_module, "load_dataset_from_path", lambda *_: raw)

    with pytest.raises(ValueError, match="filter_column='domain' not found in dataset columns"):
        ResponseDataset(
            data_path="open-thoughts/OpenThoughts-114k",
            input_key="input",
            output_key="output",
            subset="metadata",
            split="train",
            filter_column="domain",
            filter_value="math",
        )


def test_filter_empty_result_raises(monkeypatch):
    raw = Dataset.from_list([{"input": "q", "output": "a", "domain": "code"}])
    monkeypatch.setattr(response_dataset_module, "load_dataset_from_path", lambda *_: raw)

    with pytest.raises(
        ValueError, match="Filtering resulted in an empty dataset for domain == 'math'."
    ):
        ResponseDataset(
            data_path="open-thoughts/OpenThoughts-114k",
            input_key="input",
            output_key="output",
            subset="metadata",
            split="train",
            filter_column="domain",
            filter_value="math",
        )
