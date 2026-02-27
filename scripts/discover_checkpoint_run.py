#!/usr/bin/env python3
"""Discover checkpoint-run metadata from a checkpoint directory.

This script is intentionally strict for submission preflight:
- fails if no step_* directories
- fails if no step config metadata source (config.yaml)
- fails if base model cannot be found
- fails if any selected step is missing sharded model payload
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

STEP_RE = re.compile(r"^step_(\d+)$")


def parse_force_steps(value: str | None) -> list[int] | None:
    if value is None:
        return None
    out: list[int] = []
    for tok in re.split(r"[,\s]+", value.strip()):
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


def discover_steps(checkpoint_dir: Path) -> list[int]:
    steps: list[int] = []
    if not checkpoint_dir.is_dir():
        return steps
    for child in checkpoint_dir.iterdir():
        m = STEP_RE.match(child.name)
        if m and child.is_dir():
            steps.append(int(m.group(1)))
    steps.sort()
    return steps


def load_metadata_from_config(config_path: Path) -> tuple[str, str, bool]:
    cfg = yaml.safe_load(config_path.read_text())
    base_model = cfg.get("policy", {}).get("model_name")
    if not base_model:
        raise ValueError(f"missing policy.model_name in {config_path}")
    metric_name = cfg.get("checkpointing", {}).get("metric_name", "val:accuracy")
    higher = bool(cfg.get("checkpointing", {}).get("higher_is_better", True))
    return str(base_model), str(metric_name), higher


def main() -> None:
    ap = argparse.ArgumentParser(description="Discover checkpoint-run metadata.")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--force-steps", default=None)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    force_steps = parse_force_steps(args.force_steps)

    result: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "experiment_name": checkpoint_dir.name,
        "steps_discovered": [],
        "steps_selected": [],
        "steps_missing_training_info": [],
        "steps_missing_model_payload": [],
        "config_source": None,
        "base_model": None,
        "metric_name": "val:accuracy",
        "higher_is_better": True,
    }

    steps_discovered = discover_steps(checkpoint_dir)
    result["steps_discovered"] = steps_discovered
    if not steps_discovered:
        raise SystemExit(f"No step_* directories found in {checkpoint_dir}")

    if force_steps is None:
        steps_selected = list(steps_discovered)
    else:
        missing_forced = [s for s in force_steps if s not in set(steps_discovered)]
        if missing_forced:
            raise SystemExit(
                f"Forced steps missing in checkpoint dir {checkpoint_dir}: {missing_forced}"
            )
        steps_selected = list(force_steps)
    result["steps_selected"] = steps_selected

    config_source: Path | None = None
    for step in steps_discovered:
        cand = checkpoint_dir / f"step_{step}" / "config.yaml"
        if cand.is_file():
            config_source = cand
            break
    if config_source is None:
        raise SystemExit(
            f"No step_*/config.yaml metadata source found under {checkpoint_dir}"
        )
    result["config_source"] = str(config_source)

    try:
        base_model, metric_name, higher = load_metadata_from_config(config_source)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(str(e)) from e

    result["base_model"] = base_model
    result["metric_name"] = metric_name
    result["higher_is_better"] = higher

    missing_info: list[int] = []
    missing_payload: list[int] = []
    for step in steps_selected:
        step_dir = checkpoint_dir / f"step_{step}"
        if not (step_dir / "training_info.json").is_file():
            missing_info.append(step)
        if not (step_dir / "policy" / "weights" / "model").is_dir():
            missing_payload.append(step)
    result["steps_missing_training_info"] = missing_info
    result["steps_missing_model_payload"] = missing_payload

    if missing_payload:
        msg = (
            "Selected steps are missing sharded model payload "
            f"(step_*/policy/weights/model): {missing_payload}"
        )
        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
        raise SystemExit(msg)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
