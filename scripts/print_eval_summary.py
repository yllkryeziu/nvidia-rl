#!/usr/bin/env python3
"""Aggregate and pretty-print all eval results across all experiments/models/steps.

Token columns always show avg_tokens_per_sample = tokens for ONE generation,
NOT the total across all generations for a problem.

Usage:
    python scripts/print_eval_summary.py
    python scripts/print_eval_summary.py --eval-root /p/scratch/envcomp/yll/eval_results
    python scripts/print_eval_summary.py --rerun-root /p/scratch/envcomp/yll/eval_results_develbooster_reruns_avg4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EVAL_ROOT = Path("/p/scratch/envcomp/yll/eval_results")
DEFAULT_RERUN_ROOTS = [
    Path("/p/scratch/envcomp/yll/eval_results_develbooster_reruns_avg4"),
]
SKIP_DIRS = {"slurm-logs", "submission_manifests"}

EXPERIMENT_ORDER = [
    "distill-topk512-qwen3-1b7",
    "distill-topk512-qwen3-1b7-p64-g4",
    "distill-topk512-qwen3-4b",
    "distill-topk512-qwen3-8b-1node",
    "distill-topk512-qwen3-14b-2node",
]


# ── helpers ──────────────────────────────────────────────────────────────────────

def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0


def _parse_kv(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def _latest_file(root: Path, pattern: str) -> Path | None:
    matches = [f for f in root.rglob(pattern) if f.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


# ── record building ───────────────────────────────────────────────────────────────

def _bench_key(name: str) -> str:
    if "aime2025" in name:
        return "aime2025"
    if "math500" in name:
        return "math500"
    return "lcb"


def _record_from_metrics_json(
    m: dict,
    experiment: str,
    run_tag: str,
    model_label: str,
    mtime: float,
) -> dict:
    """Build a record from a pre-computed metrics.json."""
    benchmark = m.get("benchmark", "")
    bkey = _bench_key(benchmark)
    pm = m.get("primary_metric", {})
    mm = m.get("metrics", {})
    tok = m.get("token_usage", {})

    # math/aime: accuracy = primary_metric.value
    # lcb: accuracy left None; use pass@1/pass@4
    accuracy = pm.get("value") if bkey != "lcb" else None

    # LCB pass metrics — prefer pass@4, fall back to pass@5
    pass1 = mm.get("pass@1") if bkey == "lcb" else None
    pass4 = mm.get("pass@4") or mm.get("pass@5") if bkey == "lcb" else None
    # label so caller knows which k was used
    pass4_k = None
    if bkey == "lcb":
        if mm.get("pass@4") is not None:
            pass4_k = 4
        elif mm.get("pass@5") is not None:
            pass4_k = 5

    return {
        "experiment": experiment,
        "run_tag": run_tag,
        "model_label": model_label,
        "benchmark": bkey,
        "variant": m.get("variant", ""),
        "accuracy": accuracy,
        "pass1": pass1,
        "pass4": pass4,
        "pass4_k": pass4_k,
        "tok_per_gen": tok.get("avg_tokens_per_sample"),  # per single generation
        "samples_per_problem": mm.get("samples_per_problem"),
        "problems": mm.get("problems"),
        "metric_name": pm.get("name", ""),
        "status": m.get("status", "unknown"),
        "notes": m.get("notes"),
        "mtime": mtime,
        "source": "metrics_json",
    }


def _record_from_leaderboard_row(
    row: dict,
    experiment: str,
    run_tag: str,
    lb_mtime: float,
) -> dict:
    """Build a record from a leaderboard.json row (less detail than metrics.json)."""
    benchmark = row.get("benchmark", "")
    bkey = _bench_key(benchmark)
    model_label = row.get("model_label", "")

    accuracy = row.get("primary_metric_value") if bkey != "lcb" else None
    pass1 = row.get("primary_metric_value") if bkey == "lcb" else None
    pass4 = row.get("pass_at_5") if bkey == "lcb" else None  # leaderboard only has pass_at_5
    pass4_k = 5 if (bkey == "lcb" and pass4 is not None) else None

    return {
        "experiment": experiment,
        "run_tag": run_tag,
        "model_label": model_label,
        "benchmark": bkey,
        "variant": row.get("variant", ""),
        "accuracy": accuracy,
        "pass1": pass1,
        "pass4": pass4,
        "pass4_k": pass4_k,
        "tok_per_gen": row.get("avg_tokens_per_sample"),
        "samples_per_problem": row.get("samples_per_problem"),
        "problems": row.get("problems"),
        "metric_name": row.get("primary_metric_name", ""),
        "status": row.get("status", "unknown"),
        "notes": row.get("notes"),
        "mtime": lb_mtime,
        "source": "leaderboard",
    }


def _record_from_math_fallback(
    eval_data_path: Path,
    config_path: Path,
    experiment: str,
    run_tag: str,
    model_label: str,
    variant: str,
) -> dict | None:
    """Read evaluation_data.json + config.json directly (no metrics.json)."""
    cfg = _load_json(config_path) if config_path.exists() else {}
    data = _load_json(eval_data_path)
    if not isinstance(data, dict):
        return None
    entries = data.get("evaluation_data", [])
    if not entries:
        return None

    rewards = [float(e["reward"]) for e in entries if isinstance(e.get("reward"), (int, float))]
    tokens = [e["completion_tokens"] for e in entries if isinstance(e.get("completion_tokens"), int)]

    accuracy = sum(rewards) / len(rewards) if rewards else None
    tok_per_gen = sum(tokens) / len(tokens) if tokens else None  # avg per single generation
    num_tests = cfg.get("num_tests_per_prompt")
    spp = float(num_tests) if isinstance(num_tests, int) else None
    problems = len(entries) // num_tests if (isinstance(num_tests, int) and num_tests and entries) else None

    bkey = _bench_key(variant)
    metric_name = f"avg@{num_tests}_pass@1" if isinstance(num_tests, int) else "pass@1"

    return {
        "experiment": experiment,
        "run_tag": run_tag,
        "model_label": model_label,
        "benchmark": bkey,
        "variant": variant,
        "accuracy": accuracy,
        "pass1": None,
        "pass4": None,
        "pass4_k": None,
        "tok_per_gen": tok_per_gen,
        "samples_per_problem": spp,
        "problems": problems,
        "metric_name": metric_name,
        "status": "success" if accuracy is not None else "partial",
        "notes": None,
        "mtime": _mtime(eval_data_path),
        "source": "eval_data_fallback",
    }


def _record_from_lcb_fallback(
    bench_dir: Path,
    experiment: str,
    run_tag: str,
    model_label: str,
) -> dict | None:
    """Read run_info.txt + *_eval.json directly (no metrics.json)."""
    run_info = _parse_kv(bench_dir / "run_info.txt")
    workspace = bench_dir / "workspace"
    if not workspace.exists():
        return None

    eval_json_path = _latest_file(workspace, "*_eval.json")
    if eval_json_path and eval_json_path.name.endswith("_eval_all.json"):
        # skip eval_all files, only want *_eval.json
        matches = [
            f for f in workspace.rglob("*_eval.json")
            if f.is_file() and not f.name.endswith("_eval_all.json")
        ]
        eval_json_path = max(matches, key=lambda p: p.stat().st_mtime) if matches else None

    if not eval_json_path:
        return None

    raw = _load_json(eval_json_path)
    summary = (raw[0] if isinstance(raw, list) and raw else raw) if raw else {}
    if not isinstance(summary, dict):
        return None

    pass1 = summary.get("pass@1")
    pass4 = summary.get("pass@4") or summary.get("pass@5")
    pass4_k = 4 if summary.get("pass@4") is not None else (5 if summary.get("pass@5") is not None else None)
    n_raw = run_info.get("n")
    try:
        samples_per_problem = float(n_raw) if n_raw is not None else None
    except Exception:
        samples_per_problem = None

    return {
        "experiment": experiment,
        "run_tag": run_tag,
        "model_label": model_label,
        "benchmark": "lcb",
        "variant": "official_lcb",
        "accuracy": None,
        "pass1": pass1,
        "pass4": pass4,
        "pass4_k": pass4_k,
        "tok_per_gen": None,  # can't count tokens without tokenizer
        "samples_per_problem": samples_per_problem,
        "problems": None,
        "metric_name": "pass@1",
        "status": "success" if pass1 is not None else "partial",
        "notes": "tok_per_gen unavailable (no metrics.json)",
        "mtime": _mtime(eval_json_path),
        "source": "lcb_eval_fallback",
    }


# ── collection ────────────────────────────────────────────────────────────────────

def _with_origin(rec: dict, eval_root: Path, root_priority: int) -> dict:
    rec = dict(rec)
    rec["eval_root"] = str(eval_root)
    rec["root_priority"] = root_priority
    return rec


def collect_records(eval_root: Path, root_priority: int = 0) -> list[dict]:
    records: list[dict] = []

    for exp_dir in sorted(eval_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        experiment = exp_dir.name
        runs_dir = exp_dir / "runs"
        if not runs_dir.is_dir():
            continue

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_tag = run_dir.name
            models_dir = run_dir / "models"
            if not models_dir.is_dir():
                continue

            # Track which (model, bench) already have a metrics.json record so we
            # don't fall back to leaderboard for the same entry.
            seen_from_metrics: set[tuple[str, str]] = set()

            # ── primary: per-model metrics.json ──────────────────────────────────
            for model_dir in sorted(models_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_label = model_dir.name

                for variant_dir in sorted(model_dir.iterdir()):
                    if not variant_dir.is_dir():
                        continue

                    if variant_dir.name == "livecodebench":
                        # LCB: livecodebench/official_lcb/metrics.json
                        lcb_metrics = variant_dir / "official_lcb" / "metrics.json"
                        if lcb_metrics.exists():
                            m = _load_json(lcb_metrics)
                            if isinstance(m, dict):
                                rec = _record_from_metrics_json(m, experiment, run_tag, model_label, _mtime(lcb_metrics))
                                records.append(_with_origin(rec, eval_root, root_priority))
                                seen_from_metrics.add((model_label, "lcb"))
                        else:
                            # LCB fallback: read eval_json directly
                            lcb_dir = variant_dir / "official_lcb"
                            if lcb_dir.is_dir():
                                rec = _record_from_lcb_fallback(lcb_dir, experiment, run_tag, model_label)
                                if rec:
                                    records.append(_with_origin(rec, eval_root, root_priority))
                                    seen_from_metrics.add((model_label, "lcb"))
                    else:
                        # Math/AIME: {variant}/metrics.json
                        metrics_path = variant_dir / "metrics.json"
                        bkey = _bench_key(variant_dir.name)
                        if metrics_path.exists():
                            m = _load_json(metrics_path)
                            if isinstance(m, dict):
                                rec = _record_from_metrics_json(m, experiment, run_tag, model_label, _mtime(metrics_path))
                                records.append(_with_origin(rec, eval_root, root_priority))
                                seen_from_metrics.add((model_label, bkey))
                        else:
                            # Math fallback: read evaluation_data.json directly
                            eval_data = variant_dir / "evaluation_data.json"
                            config_json = variant_dir / "config.json"
                            if eval_data.exists():
                                rec = _record_from_math_fallback(
                                    eval_data, config_json,
                                    experiment, run_tag, model_label, variant_dir.name,
                                )
                                if rec:
                                    records.append(_with_origin(rec, eval_root, root_priority))
                                    seen_from_metrics.add((model_label, bkey))

            # ── secondary: leaderboard.json (fills gaps) ──────────────────────────
            lb_path = run_dir / "summary" / "leaderboard.json"
            if lb_path.exists():
                lb = _load_json(lb_path)
                if isinstance(lb, list):
                    lb_mtime = _mtime(lb_path)
                    for row in lb:
                        if not isinstance(row, dict):
                            continue
                        ml = row.get("model_label", "")
                        bkey = _bench_key(row.get("benchmark", ""))
                        if (ml, bkey) not in seen_from_metrics:
                            rec = _record_from_leaderboard_row(row, experiment, run_tag, lb_mtime)
                            records.append(_with_origin(rec, eval_root, root_priority))

    return records


# ── deduplication ─────────────────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> dict[tuple, dict]:
    """Keep preferred record per (experiment, model_label, benchmark).

    Later/preferred roots win over earlier roots. Within the same root, newer
    records win.
    """
    best: dict[tuple, dict] = {}
    for r in records:
        key = (r["experiment"], r["model_label"], r["benchmark"])
        r_rank = (r.get("root_priority", 0), r["mtime"])
        best_rank = (
            best[key].get("root_priority", 0),
            best[key]["mtime"],
        ) if key in best else None
        if best_rank is None or r_rank > best_rank:
            best[key] = r
    return best


# ── formatting ────────────────────────────────────────────────────────────────────

def _pct(v: float | None, warn: bool = False) -> str:
    if v is None:
        return "   —  "
    s = f"{v * 100:5.1f}%"
    return s + "(!)" if warn else s


def _tok(v: float | None) -> str:
    if v is None:
        return "     —  "
    return f"{int(v):6,}"


def _step_sort(label: str) -> tuple:
    if label == "base":
        return (0, 0)
    try:
        return (1, int(label.split("_")[1]))
    except Exception:
        return (2, label)


def _math_subheader(records: list[dict]) -> str:
    for r in records:
        spp = r.get("samples_per_problem")
        mn = r.get("metric_name", "")
        avg = f"avg{int(spp)}" if spp else "avg?"
        # metric_name looks like "avg@4_pass@1"
        if "_pass@" in mn:
            k = mn.split("_pass@")[1]
            return f"{avg} · pass@{k}"
        return avg
    return "—"


def _lcb_pass4_label(records: list[dict]) -> str:
    """Return 'p@4', 'p@5', or 'p@4/5' depending on what's in the data."""
    ks = {r["pass4_k"] for r in records if r.get("pass4_k") is not None}
    if not ks:
        ns = {
            int(spp) for r in records
            for spp in [r.get("samples_per_problem")]
            if isinstance(spp, (int, float))
        }
        if ns == {1}:
            return "—"
        return "p@4"
    if ks == {4}:
        return "p@4"
    if ks == {5}:
        return "p@5"
    return "p@4/5"


def _lcb_subheader(records: list[dict]) -> str:
    ns = sorted({
        int(spp) for r in records
        for spp in [r.get("samples_per_problem")]
        if isinstance(spp, (int, float))
    })
    if not ns:
        return "n=?"
    if len(ns) == 1:
        return f"n={ns[0]}"
    return "mixed n"


# ── display ───────────────────────────────────────────────────────────────────────

def display(best: dict[tuple, dict]) -> None:
    # Organize: experiment → model_label → benchmark → record
    by_exp: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (exp, model, bench), rec in best.items():
        by_exp[exp][model][bench] = rec

    exp_order = {e: i for i, e in enumerate(EXPERIMENT_ORDER)}
    experiments = sorted(by_exp.keys(), key=lambda e: (exp_order.get(e, 99), e))

    W = 104

    for exp in experiments:
        model_data = by_exp[exp]
        sorted_models = sorted(model_data.keys(), key=_step_sort)

        all_benches: set[str] = set()
        for md in model_data.values():
            all_benches.update(md.keys())
        has_math = "math500" in all_benches
        has_aime = "aime2025" in all_benches
        has_lcb  = "lcb"     in all_benches

        math_recs = [r for md in model_data.values() for b, r in md.items() if b == "math500"]
        aime_recs = [r for md in model_data.values() for b, r in md.items() if b == "aime2025"]
        lcb_recs  = [r for md in model_data.values() for b, r in md.items() if b == "lcb"]

        math_sub = _math_subheader(math_recs)
        aime_sub = _math_subheader(aime_recs)
        p4_label = _lcb_pass4_label(lcb_recs)
        lcb_sub = _lcb_subheader(lcb_recs)

        step_w = max((len(m) for m in sorted_models), default=4) + 2
        step_w = max(step_w, 10)

        # column group widths
        # math: "  acc    tok/gen" = 2+7+2+8 = 19 + group label overhead
        MATH_W  = 20   # acc(7) + gap(2) + tok/gen(8) + padding
        AIME_W  = 20
        LCB_W   = 28   # p@1(6) + gap(2) + p@4(6) + gap(2) + tok/gen(8) + padding

        print()
        print("═" * W)
        print(f"  {exp}")
        print("═" * W)

        # Row 1: group headers
        row1 = f"  {'Step':<{step_w}}"
        if has_math:
            row1 += f"  {'── math500 ──':^{MATH_W}}"
        if has_aime:
            row1 += f"  {'── aime2025 ──':^{AIME_W}}"
        if has_lcb:
            row1 += f"  {'── LiveCodeBench ──':^{LCB_W}}"
        print(row1)

        # Row 2: sub-config
        row2 = f"  {'':^{step_w}}"
        if has_math:
            row2 += f"  {math_sub:^{MATH_W}}"
        if has_aime:
            row2 += f"  {aime_sub:^{AIME_W}}"
        if has_lcb:
            row2 += f"  {lcb_sub:^{LCB_W}}"
        print(row2)

        # Row 3: column names
        row3 = f"  {'':^{step_w}}"
        if has_math:
            row3 += f"  {'acc':>7}  {'tok/gen':>8}"
        if has_aime:
            row3 += f"  {'acc':>7}  {'tok/gen':>8}"
        if has_lcb:
            row3 += f"  {'p@1':>6}  {p4_label:>6}  {'tok/gen':>8}"
        print(row3)

        print(f"  {'─' * (W - 2)}")

        for model_label in sorted_models:
            benches = model_data[model_label]
            math_r = benches.get("math500")
            aime_r = benches.get("aime2025")
            lcb_r  = benches.get("lcb")

            line = f"  {model_label:<{step_w}}"

            if has_math:
                if math_r:
                    warn = math_r.get("status") not in ("success", None)
                    line += f"  {_pct(math_r['accuracy'], warn):>7}  {_tok(math_r['tok_per_gen']):>8}"
                else:
                    line += f"  {'—':>7}  {'—':>8}"

            if has_aime:
                if aime_r:
                    warn = aime_r.get("status") not in ("success", None)
                    line += f"  {_pct(aime_r['accuracy'], warn):>7}  {_tok(aime_r['tok_per_gen']):>8}"
                else:
                    line += f"  {'—':>7}  {'—':>8}"

            if has_lcb:
                if lcb_r:
                    warn = lcb_r.get("status") not in ("success", None)
                    line += f"  {_pct(lcb_r['pass1']):>6}  {_pct(lcb_r['pass4']):>6}  {_tok(lcb_r['tok_per_gen']):>8}"
                else:
                    line += f"  {'—':>6}  {'—':>6}  {'—':>8}"

            print(line)

        print()

    print(f"  tok/gen = avg_tokens_per_sample (tokens for one generation, not summed over all samples per problem)")


# ── entry point ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Pretty-print all eval results.")
    ap.add_argument("--eval-root", default=str(EVAL_ROOT), help="Root eval results directory")
    ap.add_argument(
        "--rerun-root",
        action="append",
        default=[str(p) for p in DEFAULT_RERUN_ROOTS],
        help="Additional eval root to merge. Can be repeated; later roots win on duplicates.",
    )
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.exists():
        sys.exit(f"[ERROR] eval root not found: {eval_root}")

    records = collect_records(eval_root, root_priority=0)
    seen_roots = {eval_root.resolve()}
    for idx, rerun_root_str in enumerate(args.rerun_root, start=1):
        rerun_root = Path(rerun_root_str)
        if not rerun_root.exists():
            continue
        resolved = rerun_root.resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        records.extend(collect_records(rerun_root, root_priority=idx))
    if not records:
        print("[WARN] No eval records found under:", eval_root)
        return

    best = deduplicate(records)
    display(best)


if __name__ == "__main__":
    main()
