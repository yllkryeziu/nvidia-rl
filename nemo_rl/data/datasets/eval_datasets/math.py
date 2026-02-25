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

"""Math dataset and its variants."""

import os
from pathlib import Path
from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class MathDataset:
    def __init__(
        self,
        variant: Literal["math_test", "math_500_test"] = "math_test",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        url = f"https://openaipublic.blob.core.windows.net/simple-evals/{variant}.csv"
        local_candidates = [
            # Preferred explicit override for offline clusters.
            Path(os.environ["NEMO_RL_SIMPLE_EVALS_DIR"]) / f"{variant}.csv"
            if "NEMO_RL_SIMPLE_EVALS_DIR" in os.environ
            else None,
            # Typical when running from repo root: /.../nemo-rl/.cache/simple-evals
            Path.cwd() / "nemo-rl" / ".cache" / "simple-evals" / f"{variant}.csv",
            # Typical when running from nemo-rl cwd: /.../nemo-rl/.cache/simple-evals
            Path.cwd() / ".cache" / "simple-evals" / f"{variant}.csv",
        ]
        data_file = next((str(p) for p in local_candidates if p is not None and p.exists()), url)
        ds = load_dataset(
            "csv",
            data_files=data_file,
            split="train",
        )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=f"{variant}",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_data_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "problem": data["Question"],
            "expected_answer": data["Answer"],
        }
