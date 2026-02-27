#!/usr/bin/env python3
"""Select top checkpoints from a NeMo-RL checkpoint directory.

Reads ``step_*/training_info.json`` files and ranks checkpoints by a metric.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"^step_(\d+)$")


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value!r}")


def parse_force_steps(value: str) -> list[int]:
    tokens = re.split(r"[,\s]+", value.strip())
    out: list[int] = []
    for tok in tokens:
        if not tok:
            continue
        m = STEP_RE.match(tok)
        if m:
            out.append(int(m.group(1)))
            continue
        if tok.isdigit():
            out.append(int(tok))
            continue
        raise argparse.ArgumentTypeError(
            f"invalid step token {tok!r}; expected 100 or step_100"
        )
    return out


@dataclass
class StepRecord:
    step: str
    step_num: int
    path: str
    training_info_path: str
    exists: bool
    metric_name_used: str | None = None
    metric_value: float | None = None
    rankable: bool = False
    reason: str | None = None


def _safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def discover_steps(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    steps: list[tuple[int, Path]] = []
    if not checkpoint_dir.exists():
        return steps
    for child in checkpoint_dir.iterdir():
        m = STEP_RE.match(child.name)
        if m and child.is_dir():
            steps.append((int(m.group(1)), child))
    steps.sort(key=lambda x: x[0])
    return steps


def build_record(
    step_num: int,
    step_dir: Path,
    metric_name: str,
    allow_unranked_steps: bool,
) -> StepRecord:
    training_info_path = step_dir / "training_info.json"
    rec = StepRecord(
        step=f"step_{step_num}",
        step_num=step_num,
        path=str(step_dir),
        training_info_path=str(training_info_path),
        exists=step_dir.exists(),
    )
    if not training_info_path.exists():
        rec.reason = "missing_training_info"
        rec.rankable = False
        return rec

    try:
        data = json.loads(training_info_path.read_text())
    except Exception as e:  # noqa: BLE001
        rec.reason = f"training_info_parse_error:{type(e).__name__}"
        rec.rankable = False
        return rec

    metric_val = _safe_float(data.get(metric_name))
    metric_name_used = metric_name
    if metric_val is None and metric_name != "val:accuracy":
        fallback = _safe_float(data.get("val:accuracy"))
        if fallback is not None:
            metric_val = fallback
            metric_name_used = "val:accuracy"

    if metric_val is None:
        rec.reason = "metric_missing"
        rec.rankable = False
        if allow_unranked_steps:
            rec.reason = "metric_missing_unranked_allowed"
        return rec

    rec.metric_name_used = metric_name_used
    rec.metric_value = metric_val
    rec.rankable = True
    rec.reason = "ok"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best checkpoints from step_* dirs.")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--metric-name", required=True)
    ap.add_argument("--higher-is-better", type=parse_bool, default=True)
    ap.add_argument("--num-ckpts", type=int, default=3)
    ap.add_argument("--allow-unranked-steps", type=parse_bool, default=False)
    ap.add_argument(
        "--force-steps",
        type=parse_force_steps,
        default=None,
        help="Comma/space-separated list like '100,200,300' or 'step_100 step_200'",
    )
    ap.add_argument("--steps-only", action="store_true")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    report: dict[str, Any] = {
        "checkpoint_dir": str(ckpt_dir),
        "metric_name_requested": args.metric_name,
        "higher_is_better": bool(args.higher_is_better),
        "num_requested": int(args.num_ckpts),
        "allow_unranked_steps": bool(args.allow_unranked_steps),
        "force_steps": args.force_steps,
        "discovered": [],
        "selected": [],
        "skipped": [],
    }

    discovered_pairs = discover_steps(ckpt_dir)
    pair_by_num = {n: p for n, p in discovered_pairs}

    records: list[StepRecord] = []
    if args.force_steps:
        for step_num in args.force_steps:
            step_dir = pair_by_num.get(step_num, ckpt_dir / f"step_{step_num}")
            rec = build_record(
                step_num,
                step_dir,
                args.metric_name,
                allow_unranked_steps=args.allow_unranked_steps,
            )
            if not step_dir.exists():
                rec.exists = False
                rec.reason = "missing_step_dir"
            records.append(rec)
        selected_records = [r for r in records if r.exists]
    else:
        for step_num, step_dir in discovered_pairs:
            rec = build_record(
                step_num,
                step_dir,
                args.metric_name,
                allow_unranked_steps=args.allow_unranked_steps,
            )
            records.append(rec)

        rankable = [r for r in records if r.rankable and r.metric_value is not None]
        unranked = [r for r in records if not r.rankable]

        rankable.sort(
            key=lambda r: (
                r.metric_value if args.higher_is_better else -r.metric_value,  # type: ignore[arg-type]
                r.step_num,
            ),
            reverse=True,
        )

        selected_records = rankable[: args.num_ckpts]
        if len(selected_records) < args.num_ckpts and args.allow_unranked_steps:
            need = args.num_ckpts - len(selected_records)
            fill = sorted(unranked, key=lambda r: r.step_num, reverse=True)[:need]
            selected_records.extend(fill)

    selected_nums = {r.step_num for r in selected_records}

    report["discovered"] = [asdict(r) for r in records]
    report["selected"] = [asdict(r) for r in selected_records]
    report["skipped"] = [asdict(r) for r in records if r.step_num not in selected_nums]

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    if args.steps_only:
        for rec in selected_records:
            if rec.exists:
                print(rec.step_num)
        return

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
