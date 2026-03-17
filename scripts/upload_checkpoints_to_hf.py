#!/usr/bin/env python3
"""Upload NeMo-RL checkpoint runs to Hugging Face.

Default behavior uploads these runs into separate repos:
  - distill-topk512-qwen3-14b-2node -> <hf-username>/distill-topk512-qwen3-14b-2node
  - distill-topk512-qwen3-8b-1node  -> <hf-username>/distill-topk512-qwen3-8b-1node

Authentication is taken from your existing local Hugging Face login session.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
from pathlib import Path
from typing import Any, Callable

DEFAULT_ROOT = Path("/fast/project/HFMI_SynergyUnit/ylli/checkpoints")
DEFAULT_RUNS = [
    "distill-topk512-qwen3-14b-2node",
    "distill-topk512-qwen3-8b-1node",
]
STEP_RE = re.compile(r"^step_(\d+)$")
ROOT_CKPT_FILES = {"config.yaml", "training_info.json", "train_dataloader.pt"}


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in sig.parameters and value is not None
    }
    return fn(**supported)


def _parse_step_token(value: str) -> str:
    token = value.strip()
    if not token:
        raise argparse.ArgumentTypeError("step token cannot be empty")
    if token.isdigit():
        return f"step_{int(token)}"
    if STEP_RE.match(token):
        return token
    raise argparse.ArgumentTypeError(
        f"invalid step token {value!r}; expected 100 or step_100"
    )


def _join_repo_path(*parts: str) -> str:
    norm = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join(norm)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload checkpoint run directories to Hugging Face repos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Checkpoint root directory containing run folders.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run folder name under --root (repeatable). If omitted, uploads defaults.",
    )
    parser.add_argument(
        "--step",
        action="append",
        type=_parse_step_token,
        default=[],
        help=(
            "Only upload selected steps (repeatable), e.g. --step 150 --step step_200. "
            "If omitted, uploads the entire run folder."
        ),
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="HF namespace (user/org). If omitted, uses logged-in username.",
    )
    parser.add_argument(
        "--path-prefix",
        default="",
        help=(
            "Optional destination prefix inside repo, e.g. checkpoints/2026-03-01. "
            "Useful for additive uploads without touching prior paths."
        ),
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target revision/branch in destination repos.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private if they do not already exist.",
    )
    parser.add_argument(
        "--num-workers",
        type=_non_negative_int,
        default=8,
        help="Worker count for upload_large_folder when available.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Glob pattern(s) to include (repeatable).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Glob pattern(s) to exclude (repeatable).",
    )
    parser.add_argument(
        "--cleanup-root",
        action="store_true",
        help=(
            "Delete accidental root-level checkpoint files in repo "
            "(policy/* plus root config/training_info/train_dataloader)."
        ),
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only run root cleanup; skip uploads.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _extract_username(whoami: dict[str, Any]) -> str:
    for key in ("name", "username", "user"):
        value = whoami.get(key)
        if isinstance(value, str) and value:
            return value
    raise SystemExit(f"Could not determine username from whoami(): {whoami}")


def _root_cleanup_paths(repo_files: list[str]) -> list[str]:
    to_delete = [
        f for f in repo_files if f.startswith("policy/") or f in ROOT_CKPT_FILES
    ]
    return sorted(to_delete)


def main() -> None:
    args = _build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Checkpoint root does not exist: {root}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required in your venv.\n"
            "Install with: pip install -U huggingface_hub[hf_transfer]"
        ) from exc

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    api = HfApi()

    try:
        whoami = api.whoami()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Hugging Face login not found. Run `huggingface-cli login` in this venv."
        ) from exc

    namespace = args.namespace or _extract_username(whoami)
    run_names: list[str] = args.run if args.run else list(DEFAULT_RUNS)
    allow_patterns = args.allow_pattern or None
    ignore_patterns = args.ignore_pattern or None

    use_large_upload = hasattr(api, "upload_large_folder")
    large_upload_supports_path = False
    if use_large_upload:
        large_upload_supports_path = (
            "path_in_repo" in inspect.signature(api.upload_large_folder).parameters
        )

    for run_name in run_names:
        run_dir = root / run_name
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory does not exist: {run_dir}")

        repo_id = f"{namespace}/{run_name}"
        upload_units: list[tuple[Path, str, str]] = []
        if args.step:
            for step_name in args.step:
                step_dir = run_dir / step_name
                if not step_dir.is_dir():
                    raise SystemExit(f"Step directory does not exist: {step_dir}")
                repo_path = _join_repo_path(args.path_prefix, step_name)
                upload_units.append((step_dir, repo_path, step_name))
        else:
            repo_path = _join_repo_path(args.path_prefix)
            upload_units.append((run_dir, repo_path, run_name))

        # If the local huggingface_hub cannot target path_in_repo for large uploads,
        # force upload_folder to preserve step_* directories in the repo.
        need_non_root_paths = any(repo_path for _, repo_path, _ in upload_units)
        if use_large_upload and (need_non_root_paths and not large_upload_supports_path):
            upload_fn = api.upload_folder
            upload_method = "upload_folder"
        else:
            upload_fn = api.upload_large_folder if use_large_upload else api.upload_folder
            upload_method = "upload_large_folder" if use_large_upload else "upload_folder"

        print(f"\nRun             : {run_name}")
        print(f"Repo            : {args.repo_type}:{repo_id}")
        print(f"Revision        : {args.revision}")
        print(f"Upload method   : {upload_method}")
        print(f"Path prefix     : {args.path_prefix or '<repo-root>'}")
        print(f"Selected steps  : {args.step if args.step else '<all>'}")
        print(f"Cleanup root    : {args.cleanup_root}")
        print(f"Cleanup only    : {args.cleanup_only}")
        print(f"Dry run         : {args.dry_run}")

        _call_with_supported_kwargs(
            api.create_repo,
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=bool(args.private),
            exist_ok=True,
        )

        if not args.cleanup_only:
            if args.dry_run:
                for local_dir, repo_path, label in upload_units:
                    print(
                        f"  [DRY-RUN] upload {label}: {local_dir} -> "
                        f"{repo_path or '<repo-root>'}"
                    )
            else:
                for local_dir, repo_path, label in upload_units:
                    print(f"  Uploading      : {label}")
                    print(f"  Local dir      : {local_dir}")
                    print(f"  Repo path      : {repo_path or '<repo-root>'}")

                    upload_kwargs: dict[str, Any] = {
                        "repo_id": repo_id,
                        "repo_type": args.repo_type,
                        "folder_path": str(local_dir),
                        "path_in_repo": repo_path,
                        "revision": args.revision,
                        "allow_patterns": allow_patterns,
                        "ignore_patterns": ignore_patterns,
                    }

                    if upload_method == "upload_large_folder":
                        upload_kwargs["num_workers"] = args.num_workers
                        if "commit_message" in inspect.signature(upload_fn).parameters:
                            upload_kwargs["commit_message"] = (
                                f"Add {run_name}/{label} checkpoints"
                            )
                    else:
                        upload_kwargs["commit_message"] = (
                            f"Add {run_name}/{label} checkpoints"
                        )
                        upload_kwargs["multi_commits"] = True
                        upload_kwargs["multi_commits_verbose"] = True

                    _call_with_supported_kwargs(upload_fn, **upload_kwargs)

        if args.cleanup_root:
            repo_files = api.list_repo_files(
                repo_id=repo_id, repo_type=args.repo_type, revision=args.revision
            )
            delete_paths = _root_cleanup_paths(repo_files)
            print(f"  Root cleanup   : {len(delete_paths)} file(s) to delete")
            for p in delete_paths[:20]:
                print(f"    - {p}")
            if len(delete_paths) > 20:
                print(f"    ... ({len(delete_paths) - 20} more)")

            if delete_paths and not args.dry_run:
                from huggingface_hub import CommitOperationDelete

                operations = [CommitOperationDelete(path_in_repo=p) for p in delete_paths]
                _call_with_supported_kwargs(
                    api.create_commit,
                    repo_id=repo_id,
                    repo_type=args.repo_type,
                    revision=args.revision,
                    operations=operations,
                    commit_message="Cleanup accidental root-level checkpoint files",
                )
                print("  Root cleanup   : done")

        print(f"Done            : {repo_id}")

    print("\nAll uploads completed.")


if __name__ == "__main__":
    main()
