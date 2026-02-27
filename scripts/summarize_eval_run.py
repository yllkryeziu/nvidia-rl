#!/usr/bin/env python3
"""Normalize post-training eval outputs into a single summary table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_EXPECTED_BENCHMARKS = ("aime2025_avg16", "math500_avg16", "livecodebench")
BENCHMARK_ORDER = {"aime2025_avg16": 0, "math500_avg16": 1, "livecodebench": 2}


def parse_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:  # noqa: BLE001
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def latest_match(root: Path, pattern: str) -> Path | None:
    matches = [p for p in root.rglob(pattern) if p.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def latest_lcb_eval_json(workspace_dir: Path) -> Path | None:
    matches = [
        p
        for p in workspace_dir.rglob("*_eval.json")
        if p.is_file() and not p.name.endswith("_eval_all.json")
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def parse_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass
class SummaryRow:
    run_tag: str
    model_label: str
    model_path: str | None
    benchmark: str
    variant: str
    primary_metric_name: str | None
    primary_metric_value: float | None
    pass_at_5: float | None
    pass_at_10: float | None
    problems: int | None
    samples: int | None
    samples_per_problem: float | None
    avg_tokens_per_sample: float | None
    avg_tokens_per_problem: float | None
    total_tokens: int | None
    raw_outputs_path: str | None
    metrics_json_path: str | None
    status: str
    source: str | None
    notes: str | None = None


class TokenizerCache:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def count_tokens(self, model_path: str, texts: list[str]) -> list[int]:
        if model_path not in self._cache:
            self._cache[model_path] = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        tok = self._cache[model_path]
        return [len(tok.encode(t or "", add_special_tokens=False)) for t in texts]


def summarize_math_like(
    run_root: Path,
    model_label: str,
    bench_dir: Path,
    benchmark_name: str,
    variant: str,
    write_metrics: bool,
) -> SummaryRow:
    eval_data_path = bench_dir / "evaluation_data.json"
    config_path = bench_dir / "config.json"
    metrics_path = bench_dir / "metrics.json"

    if not eval_data_path.exists():
        return SummaryRow(
            run_tag=run_root.name,
            model_label=model_label,
            model_path=None,
            benchmark=benchmark_name,
            variant=variant,
            primary_metric_name="avg@16_pass@1",
            primary_metric_value=None,
            pass_at_5=None,
            pass_at_10=None,
            problems=None,
            samples=None,
            samples_per_problem=None,
            avg_tokens_per_sample=None,
            avg_tokens_per_problem=None,
            total_tokens=None,
            raw_outputs_path=None,
            metrics_json_path=rel(metrics_path, run_root),
            status="missing",
            source="nemo_eval",
            notes="missing evaluation_data.json",
        )

    try:
        data = load_json(eval_data_path)
        entries = data.get("evaluation_data", [])
        cfg = load_json(config_path) if config_path.exists() else {}
    except Exception as e:  # noqa: BLE001
        return SummaryRow(
            run_tag=run_root.name,
            model_label=model_label,
            model_path=None,
            benchmark=benchmark_name,
            variant=variant,
            primary_metric_name="avg@16_pass@1",
            primary_metric_value=None,
            pass_at_5=None,
            pass_at_10=None,
            problems=None,
            samples=None,
            samples_per_problem=None,
            avg_tokens_per_sample=None,
            avg_tokens_per_problem=None,
            total_tokens=None,
            raw_outputs_path=rel(eval_data_path, run_root),
            metrics_json_path=rel(metrics_path, run_root),
            status="failed",
            source="nemo_eval",
            notes=f"parse_error:{type(e).__name__}",
        )

    rewards = []
    completion_tokens: list[int] = []
    for row in entries:
        reward = row.get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        ct = row.get("completion_tokens")
        if isinstance(ct, int):
            completion_tokens.append(ct)
    samples = len(entries)

    num_tests = cfg.get("num_tests_per_prompt")
    problems: int | None = None
    spp: float | None = None
    if isinstance(num_tests, int) and num_tests > 0:
        spp = float(num_tests)
        if samples > 0 and samples % num_tests == 0:
            problems = samples // num_tests
    if problems is None and samples > 0:
        # best effort fallback for nonstandard outputs
        problems = None

    primary = (sum(rewards) / len(rewards)) if rewards else None
    total_tokens = sum(completion_tokens) if completion_tokens else None
    avg_tokens_per_sample = (total_tokens / samples) if (total_tokens is not None and samples) else None
    avg_tokens_per_problem = (
        total_tokens / problems
        if (total_tokens is not None and isinstance(problems, int) and problems > 0)
        else None
    )
    status = "success"
    notes = None
    if not config_path.exists():
        status = "partial"
        notes = "missing config.json"
    elif primary is None:
        status = "partial"
        notes = "missing reward values"

    metrics_doc = {
        "benchmark": benchmark_name,
        "variant": variant,
        "model_label": model_label,
        "model_path": cfg.get("model_name"),
        "status": status,
        "primary_metric": {"name": "avg@16_pass@1", "value": primary},
        "metrics": {
            "avg@16_pass@1": primary,
            "problems": problems,
            "samples": samples,
            "samples_per_problem": spp,
        },
        "token_usage": {
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": avg_tokens_per_sample,
            "avg_tokens_per_problem": avg_tokens_per_problem,
        },
        "artifacts": {
            "evaluation_data_json": rel(eval_data_path, run_root),
            "config_json": rel(config_path, run_root) if config_path.exists() else None,
        },
        "source": "nemo_eval",
        "notes": notes,
    }
    if write_metrics:
        metrics_path.write_text(json.dumps(metrics_doc, indent=2))

    return SummaryRow(
        run_tag=run_root.name,
        model_label=model_label,
        model_path=cfg.get("model_name"),
        benchmark=benchmark_name,
        variant=variant,
        primary_metric_name="avg@16_pass@1",
        primary_metric_value=primary,
        pass_at_5=None,
        pass_at_10=None,
        problems=problems,
        samples=samples,
        samples_per_problem=spp,
        avg_tokens_per_sample=avg_tokens_per_sample,
        avg_tokens_per_problem=avg_tokens_per_problem,
        total_tokens=total_tokens,
        raw_outputs_path=rel(eval_data_path, run_root),
        metrics_json_path=rel(metrics_path, run_root),
        status=status,
        source="nemo_eval",
        notes=notes,
    )


def _extract_eval_summary(eval_json: Any) -> dict[str, Any]:
    if isinstance(eval_json, list) and eval_json and isinstance(eval_json[0], dict):
        return eval_json[0]
    if isinstance(eval_json, dict):
        return eval_json
    return {}


def summarize_lcb(
    run_root: Path,
    model_label: str,
    bench_dir: Path,
    write_metrics: bool,
    tok_cache: TokenizerCache,
) -> SummaryRow:
    metrics_path = bench_dir / "metrics.json"
    run_info_path = bench_dir / "run_info.txt"
    workspace_dir = bench_dir / "workspace"

    eval_json_path = latest_lcb_eval_json(workspace_dir) if workspace_dir.exists() else None
    eval_all_json_path = latest_match(workspace_dir, "*_eval_all.json") if workspace_dir.exists() else None

    if not eval_json_path and not eval_all_json_path and not run_info_path.exists():
        return SummaryRow(
            run_tag=run_root.name,
            model_label=model_label,
            model_path=None,
            benchmark="livecodebench",
            variant="official_lcb",
            primary_metric_name="pass@1",
            primary_metric_value=None,
            pass_at_5=None,
            pass_at_10=None,
            problems=None,
            samples=None,
            samples_per_problem=None,
            avg_tokens_per_sample=None,
            avg_tokens_per_problem=None,
            total_tokens=None,
            raw_outputs_path=None,
            metrics_json_path=rel(metrics_path, run_root),
            status="missing",
            source="livecodebench_official",
            notes="missing LCB artifacts",
        )

    run_info = parse_kv_file(run_info_path) if run_info_path.exists() else {}
    model_path = run_info.get("model_path")
    pass1 = pass5 = pass10 = None
    problems = samples = None
    spp = None
    total_tokens = None
    avg_tokens_per_sample = None
    avg_tokens_per_problem = None
    token_notes = None
    status = "success"

    if eval_json_path:
        try:
            eval_summary = _extract_eval_summary(load_json(eval_json_path))
            for key, attr in (("pass@1", "pass1"), ("pass@5", "pass5"), ("pass@10", "pass10")):
                val = eval_summary.get(key)
                if isinstance(val, (int, float)):
                    if attr == "pass1":
                        pass1 = float(val)
                    elif attr == "pass5":
                        pass5 = float(val)
                    else:
                        pass10 = float(val)
        except Exception as e:  # noqa: BLE001
            status = "partial"
            token_notes = f"eval_json_parse_error:{type(e).__name__}"
    else:
        status = "partial"
        token_notes = "missing *_eval.json"

    if eval_all_json_path:
        try:
            rows = load_json(eval_all_json_path)
            if isinstance(rows, list):
                problems = len(rows)
                outputs_flat: list[str] = []
                sample_counts: list[int] = []
                for row in rows:
                    output_list = row.get("output_list", []) if isinstance(row, dict) else []
                    if isinstance(output_list, list):
                        sample_counts.append(len(output_list))
                        for out in output_list:
                            outputs_flat.append(out if isinstance(out, str) else "")
                samples = len(outputs_flat)
                if problems and sample_counts:
                    spp = sum(sample_counts) / len(sample_counts)
                if model_path:
                    try:
                        token_counts = tok_cache.count_tokens(model_path, outputs_flat)
                        total_tokens = sum(token_counts)
                        if samples:
                            avg_tokens_per_sample = total_tokens / samples
                        if problems:
                            avg_tokens_per_problem = total_tokens / problems
                        token_extra = {
                            "min_tokens_per_sample": min(token_counts) if token_counts else None,
                            "median_tokens_per_sample": (
                                statistics.median(token_counts) if token_counts else None
                            ),
                            "max_tokens_per_sample": max(token_counts) if token_counts else None,
                        }
                    except Exception as e:  # noqa: BLE001
                        status = "partial"
                        token_extra = {
                            "min_tokens_per_sample": None,
                            "median_tokens_per_sample": None,
                            "max_tokens_per_sample": None,
                        }
                        msg = f"lcb_token_count_error:{type(e).__name__}"
                        token_notes = msg if token_notes is None else f"{token_notes};{msg}"
                else:
                    status = "partial"
                    token_extra = {
                        "min_tokens_per_sample": None,
                        "median_tokens_per_sample": None,
                        "max_tokens_per_sample": None,
                    }
                    msg = "missing model_path in run_info.txt for token counting"
                    token_notes = msg if token_notes is None else f"{token_notes};{msg}"
            else:
                status = "partial"
                token_extra = {
                    "min_tokens_per_sample": None,
                    "median_tokens_per_sample": None,
                    "max_tokens_per_sample": None,
                }
                msg = "unexpected *_eval_all.json format"
                token_notes = msg if token_notes is None else f"{token_notes};{msg}"
        except Exception as e:  # noqa: BLE001
            status = "partial"
            token_extra = {
                "min_tokens_per_sample": None,
                "median_tokens_per_sample": None,
                "max_tokens_per_sample": None,
            }
            msg = f"eval_all_parse_error:{type(e).__name__}"
            token_notes = msg if token_notes is None else f"{token_notes};{msg}"
    else:
        status = "partial"
        token_extra = {
            "min_tokens_per_sample": None,
            "median_tokens_per_sample": None,
            "max_tokens_per_sample": None,
        }
        msg = "missing *_eval_all.json"
        token_notes = msg if token_notes is None else f"{token_notes};{msg}"

    metrics_doc = {
        "benchmark": "livecodebench",
        "variant": "official_lcb",
        "model_label": model_label,
        "model_path": model_path,
        "status": status,
        "primary_metric": {"name": "pass@1", "value": pass1},
        "metrics": {
            "pass@1": pass1,
            "pass@5": pass5,
            "pass@10": pass10,
            "problems": problems,
            "samples": samples,
            "samples_per_problem": spp,
        },
        "token_usage": {
            "counted_on": "output_list",
            "includes_reasoning": True,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": avg_tokens_per_sample,
            "avg_tokens_per_problem": avg_tokens_per_problem,
            **token_extra,
        },
        "artifacts": {
            "run_info_txt": rel(run_info_path, run_root) if run_info_path.exists() else None,
            "eval_json": rel(eval_json_path, run_root) if eval_json_path else None,
            "eval_all_json": rel(eval_all_json_path, run_root) if eval_all_json_path else None,
        },
        "source": "livecodebench_official",
        "notes": token_notes,
    }
    if write_metrics:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_doc, indent=2))

    return SummaryRow(
        run_tag=run_root.name,
        model_label=model_label,
        model_path=model_path,
        benchmark="livecodebench",
        variant="official_lcb",
        primary_metric_name="pass@1",
        primary_metric_value=pass1,
        pass_at_5=pass5,
        pass_at_10=pass10,
        problems=problems,
        samples=samples,
        samples_per_problem=spp,
        avg_tokens_per_sample=avg_tokens_per_sample,
        avg_tokens_per_problem=avg_tokens_per_problem,
        total_tokens=total_tokens,
        raw_outputs_path=rel(eval_all_json_path, run_root) if eval_all_json_path else None,
        metrics_json_path=rel(metrics_path, run_root),
        status=status,
        source="livecodebench_official",
        notes=token_notes,
    )


def expected_rows_for_model(
    run_root: Path,
    model_label: str,
    model_dir: Path,
    write_metrics: bool,
    tok_cache: TokenizerCache,
    expected_benchmarks: tuple[str, ...],
) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    mapping = {
        "aime2025_avg16": model_dir / "aime2025_avg16",
        "math500_avg16": model_dir / "math500_avg16",
        "livecodebench": model_dir / "livecodebench" / "official_lcb",
    }
    for benchmark in expected_benchmarks:
        path = mapping[benchmark]
        if benchmark == "livecodebench":
            rows.append(summarize_lcb(run_root, model_label, path, write_metrics, tok_cache))
        else:
            variant = benchmark
            rows.append(
                summarize_math_like(
                    run_root=run_root,
                    model_label=model_label,
                    bench_dir=path,
                    benchmark_name=benchmark.split("_avg16")[0],
                    variant=variant,
                    write_metrics=write_metrics,
                )
            )
    return rows


def write_outputs(run_root: Path, rows: list[SummaryRow]) -> None:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    row_dicts = [asdict(r) for r in rows]

    leaderboard_json = summary_dir / "leaderboard.json"
    leaderboard_json.write_text(json.dumps(row_dicts, indent=2))

    tsv_path = summary_dir / "leaderboard.tsv"
    fieldnames = [
        "run_tag",
        "model_label",
        "model_path",
        "benchmark",
        "variant",
        "primary_metric_name",
        "primary_metric_value",
        "pass@5",
        "pass@10",
        "problems",
        "samples",
        "samples_per_problem",
        "avg_tokens_per_sample",
        "avg_tokens_per_problem",
        "total_tokens",
        "raw_outputs_path",
        "metrics_json_path",
        "status",
        "source",
        "notes",
    ]
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in row_dicts:
            out = dict(r)
            out["pass@5"] = out.pop("pass_at_5")
            out["pass@10"] = out.pop("pass_at_10")
            writer.writerow({k: ("" if out.get(k) is None else out.get(k)) for k in fieldnames})

    statuses = [r.status for r in rows]
    model_labels = sorted({r.model_label for r in rows})
    step_models = [m for m in model_labels if m.startswith("step_")]
    if not rows:
        overall = "failed"
    elif any(s == "failed" for s in statuses):
        overall = "failed"
    elif any(s in {"partial", "missing"} for s in statuses):
        overall = "partial_no_checkpoints" if not step_models else "partial"
    else:
        overall = "success"

    run_status = {
        "run_tag": run_root.name,
        "overall_status": overall,
        "row_count": len(rows),
        "models": model_labels,
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
    }
    (summary_dir / "run_status.json").write_text(json.dumps(run_status, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a post-training eval run.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--fail-on-missing", type=parse_bool, default=False)
    ap.add_argument("--write-per-benchmark-metrics", type=parse_bool, default=True)
    ap.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_EXPECTED_BENCHMARKS),
        help="Comma-separated expected benchmarks (aime2025_avg16,math500_avg16,livecodebench)",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root)
    models_root = run_root / "models" if (run_root / "models").is_dir() else run_root
    if not models_root.exists():
        raise SystemExit(f"Run root not found: {run_root}")

    model_dirs = [
        p
        for p in models_root.iterdir()
        if p.is_dir() and (p.name == "base" or p.name.startswith("step_"))
    ]
    model_dirs.sort(key=lambda p: (0 if p.name == "base" else 1, p.name))

    expected_benchmarks = tuple(
        b.strip() for b in args.benchmarks.split(",") if b.strip()
    )

    tok_cache = TokenizerCache()
    rows: list[SummaryRow] = []
    for model_dir in model_dirs:
        rows.extend(
            expected_rows_for_model(
                run_root=run_root,
                model_label=model_dir.name,
                model_dir=model_dir,
                write_metrics=args.write_per_benchmark_metrics,
                tok_cache=tok_cache,
                expected_benchmarks=expected_benchmarks,
            )
        )

    rows.sort(key=lambda r: (0 if r.model_label == "base" else 1, r.model_label, BENCHMARK_ORDER.get(r.benchmark if r.benchmark != "aime2025" and r.benchmark != "math500" else f"{r.benchmark}_avg16", 99)))
    write_outputs(run_root, rows)

    if args.fail_on_missing and any(r.status in {"missing", "partial", "failed"} for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
