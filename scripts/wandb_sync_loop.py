#!/usr/bin/env python3
"""Periodically sync offline W&B runs from a login node."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WANDB_SUFFIX = ".wandb"
SYNCED_SUFFIX = ".synced"
RUN_DIR_PREFIXES = ("offline-run-", "run-")

_stop_requested = False


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def request_stop(signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True
    signame = signal.Signals(signum).name
    log(f"Received {signame}; stopping after the current sync attempt.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Loop on a login node and periodically run `wandb sync` for offline runs."
        )
    )
    parser.add_argument(
        "target",
        help=(
            "Path to a base log dir, experiment dir, wandb dir, offline-run dir, or "
            ".wandb file."
        ),
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=15.0,
        help="Polling interval between sync passes (default: 15).",
    )
    parser.add_argument(
        "--wandb-python",
        default=sys.executable,
        help=(
            "Python interpreter used for `python -m wandb sync` "
            f"(default: {sys.executable})."
        ),
    )
    parser.add_argument(
        "--include-synced",
        action="store_true",
        help="Also retry runs already marked with `.wandb.synced`.",
    )
    parser.add_argument(
        "--skip-console",
        action="store_true",
        help="Pass `--skip-console` to `wandb sync`.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sync pass and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which runs would be synced; do not call `wandb sync`.",
    )
    return parser.parse_args()


def find_run_dirs(target: Path) -> list[Path]:
    if not target.exists():
        raise SystemExit(f"Target does not exist: {target}")

    if target.is_file():
        if target.name.endswith(WANDB_SUFFIX):
            return [target.parent.resolve()]
        raise SystemExit(f"Unsupported file target: {target}")

    if target.is_dir() and target.name.startswith(RUN_DIR_PREFIXES):
        return [target.resolve()]

    run_dirs: set[Path] = set()
    for pattern in RUN_DIR_PREFIXES:
        for path in target.rglob(f"{pattern}*"):
            if path.is_dir():
                run_dirs.add(path.resolve())

    return sorted(run_dirs)


def get_wandb_file(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob(f"*{WANDB_SUFFIX}"))
    if len(candidates) != 1:
        return None
    return candidates[0]


def select_runs(run_dirs: list[Path], include_synced: bool) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    warnings: list[str] = []

    for run_dir in run_dirs:
        wandb_file = get_wandb_file(run_dir)
        if wandb_file is None:
            warnings.append(f"Skipping {run_dir}: expected exactly one *{WANDB_SUFFIX} file.")
            continue

        synced_marker = Path(f"{wandb_file}{SYNCED_SUFFIX}")
        if synced_marker.exists() and not include_synced:
            continue
        selected.append(run_dir)

    return selected, warnings


def sync_run(run_dir: Path, args: argparse.Namespace) -> int:
    cmd = [args.wandb_python, "-m", "wandb", "sync", "--mark-synced"]
    if args.skip_console:
        cmd.append("--skip-console")
    cmd.append(str(run_dir))

    env = os.environ.copy()
    env.pop("WANDB_MODE", None)
    env.setdefault("PYTHONWARNINGS", "ignore")

    log(f"Syncing {run_dir}")
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode == 0:
        log(f"Finished {run_dir}")
    else:
        log(f"`wandb sync` failed for {run_dir} with exit code {completed.returncode}")
    return completed.returncode


def sleep_with_stop(interval_seconds: float) -> None:
    deadline = time.time() + interval_seconds
    while not _stop_requested:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5.0))


def main() -> None:
    args = parse_args()
    target = Path(args.target).expanduser().resolve()
    interval_seconds = max(args.interval_minutes * 60.0, 0.0)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    log(f"Watching {target}")
    log(f"Using interpreter {args.wandb_python} for `python -m wandb sync`")

    pass_index = 0
    while not _stop_requested:
        pass_index += 1
        run_dirs = find_run_dirs(target)
        selected_runs, warnings = select_runs(run_dirs, args.include_synced)

        for warning in warnings:
            log(warning)

        log(
            f"Pass {pass_index}: discovered {len(run_dirs)} run dirs, "
            f"{len(selected_runs)} pending sync."
        )

        failures = 0
        for run_dir in selected_runs:
            if _stop_requested:
                break
            if args.dry_run:
                log(f"Would sync {run_dir}")
                continue
            failures += int(sync_run(run_dir, args) != 0)

        if args.once or _stop_requested:
            break

        if failures:
            log(
                f"Pass {pass_index} completed with {failures} failed sync attempts; "
                f"retrying after {args.interval_minutes:g} minutes."
            )
        else:
            log(
                f"Pass {pass_index} completed; sleeping for "
                f"{args.interval_minutes:g} minutes."
            )
        sleep_with_stop(interval_seconds)

    log("Exited sync loop.")


if __name__ == "__main__":
    main()
