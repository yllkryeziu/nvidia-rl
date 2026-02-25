#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def summarize_file(path: Path, tokenizer, field: str) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected top-level list, got {type(data).__name__}")

    problem_count = len(data)
    sample_counts = []
    per_problem_totals = []
    non_empty = 0

    for idx, row in enumerate(data):
        outputs = row.get(field)
        if not isinstance(outputs, list):
            raise ValueError(
                f"{path}: row {idx} field {field!r} expected list, got {type(outputs).__name__}"
            )
        row_total = 0
        for out in outputs:
            if not isinstance(out, str):
                raise ValueError(
                    f"{path}: row {idx} field {field!r} contains non-string item {type(out).__name__}"
                )
            t = count_tokens(out, tokenizer)
            sample_counts.append(t)
            row_total += t
            if out:
                non_empty += 1
        per_problem_totals.append(row_total)

    total_samples = len(sample_counts)
    total_tokens = sum(sample_counts)
    return {
        "path": str(path),
        "field": field,
        "problems": problem_count,
        "samples": total_samples,
        "non_empty_samples": non_empty,
        "total_tokens": total_tokens,
        "avg_tokens_per_sample": (total_tokens / total_samples) if total_samples else 0.0,
        "avg_tokens_per_problem": (total_tokens / problem_count) if problem_count else 0.0,
        "min_tokens_per_sample": min(sample_counts) if sample_counts else 0,
        "max_tokens_per_sample": max(sample_counts) if sample_counts else 0,
        "median_tokens_per_sample": (
            sorted(sample_counts)[total_samples // 2] if sample_counts else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count completion tokens in LiveCodeBench output JSON files."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        help="LCB output JSON files, e.g. Scenario.codegeneration_10_0.2.json",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer path (HF local path/checkpoint) used for token counting.",
    )
    parser.add_argument(
        "--field",
        default="output_list",
        help="JSON field containing generated answers (default: output_list).",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    grand_total = 0
    print(
        "file\tproblems\tsamples\tnon_empty_samples\ttotal_tokens\tavg_tokens_per_sample\tavg_tokens_per_problem\tmin\tmedian\tmax"
    )
    for json_file in args.json_files:
        summary = summarize_file(Path(json_file), tokenizer, args.field)
        grand_total += summary["total_tokens"]
        print(
            f"{summary['path']}\t{summary['problems']}\t{summary['samples']}\t"
            f"{summary['non_empty_samples']}\t{summary['total_tokens']}\t"
            f"{summary['avg_tokens_per_sample']:.2f}\t{summary['avg_tokens_per_problem']:.2f}\t"
            f"{summary['min_tokens_per_sample']}\t{summary['median_tokens_per_sample']}\t"
            f"{summary['max_tokens_per_sample']}"
        )

    if len(args.json_files) > 1:
        print(f"TOTAL\t-\t-\t-\t{grand_total}")


if __name__ == "__main__":
    main()
