#!/usr/bin/env python3
"""
Aggregate lm-eval `results_*.json` files into eval_results.csv (oellm-style columns).

Supported directory layouts:

  Legacy (single run):
    <root>/<task>/n<k>/.../results_*.json

  Batch (multiple experiments; optional outer batch folder):
    <root>/<exp_dir>/<task>/<run_id>/results_*.json
    <root>/<batch>/<exp_dir>/<task>/<run_id>/results_*.json

If several `results_*.json` exist under the same immediate parent directory, the newest
(by mtime) is used. `n_shot` is taken from the path (`n<k>`) when present, otherwise from
the JSON (`n-shot` / task config).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

TASK_METRICS: dict[str, str] = {
    "mmlu": "acc",
    "copa": "acc",
    "lambada_openai": "acc",
    "openbookqa": "acc_norm",
    "winogrande": "acc",
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "boolq": "acc",
    "commonsense_qa": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "social_iqa": "acc",
    "agieval_lsat_ar": "acc",
    "wsc273": "acc",
    "bigbench_language_identification_multiple_choice": "acc",
    "squadv2": "f1",
    "coqa": "f1",
    "bigbench_qa_wikidata_generate_until": "exact_match",
    "bigbench_dyck_languages_generate_until": "exact_match",
    "bigbench_operators_generate_until": "exact_match",
    "bigbench_repeat_copy_logic_generate_until": "exact_match",
    "bigbench_cs_algorithms_generate_until": "exact_match",
}


def parse_n_shot_segment(name: str) -> int | None:
    m = re.fullmatch(r"n(\d+)", name)
    return int(m.group(1)) if m else None


def default_model_label(config: dict) -> str:
    ma = config.get("model_args") or {}
    load = ma.get("load", "")
    step = ma.get("ckpt_step", "")
    p = Path(str(load))
    if p.name == "checkpoints" and p.parent.name:
        return f"{p.parent.name}_step{step}"
    if load:
        return f"{load}_step{step}"
    return str(config.get("model_name") or "unknown")


def resolve_results_entry(results: dict, task: str) -> dict:
    if task in results:
        return results[task]
    keys = [k for k in results if isinstance(results[k], dict) and "alias" in results[k]]
    if len(keys) == 1:
        return results[keys[0]]
    raise KeyError(f"No results for task {task!r}; keys: {list(results)}")


def metric_value(task_block: dict, metric: str) -> float:
    key = f"{metric},none"
    if key not in task_block:
        raise KeyError(f"Missing {key!r} in results; have: {list(task_block)}")
    v = task_block[key]
    if not isinstance(v, (int, float)):
        raise TypeError(f"Expected number for {key}, got {type(v)}")
    return float(v)


def n_shot_from_json(data: dict, task: str) -> int:
    ns = data.get("n-shot")
    if isinstance(ns, dict):
        if task in ns:
            return int(ns[task])
        # MMLU aggregate: JSON uses per-subject keys (mmlu_abstract_algebra, …), not "mmlu".
        if task == "mmlu":
            for k, v in ns.items():
                if isinstance(k, str) and k.startswith("mmlu_"):
                    return int(v)
    cfgc = data.get("configs") or {}
    if task in cfgc and isinstance(cfgc[task], dict) and "num_fewshot" in cfgc[task]:
        return int(cfgc[task]["num_fewshot"])
    if task == "mmlu":
        for k, cfg in cfgc.items():
            if (
                isinstance(k, str)
                and k.startswith("mmlu_")
                and isinstance(cfg, dict)
                and "num_fewshot" in cfg
            ):
                return int(cfg["num_fewshot"])
    raise KeyError(f"Could not determine n_shot for task {task!r} from JSON")


def parse_task_and_path_n_shot(root: Path, jpath: Path) -> tuple[str, int | None]:
    """
    Infer lm-eval task name and optional n_shot encoded as `n<k>` in the path.
    Returns (task, n_shot_or_none).
    """
    root_r = root.resolve()
    jpath_r = jpath.resolve()
    try:
        rel = jpath_r.relative_to(root_r)
    except ValueError as e:
        raise ValueError(f"{jpath} is not under {root}") from e

    dirs = rel.parts[:-1]
    if len(dirs) < 2:
        raise ValueError(f"Unexpected depth for {rel}: need at least task/.../results.json")

    if parse_n_shot_segment(dirs[1]) is not None:
        return dirs[0], parse_n_shot_segment(dirs[1])

    return dirs[-2], None


def pick_latest_json_per_output_dir(root: Path) -> list[Path]:
    root_r = root.resolve()
    candidates = [p for p in root_r.rglob("results_*.json") if p.is_file()]
    by_parent: dict[Path, Path] = {}
    for p in candidates:
        par = p.parent
        cur = by_parent.get(par)
        if cur is None or p.stat().st_mtime > cur.stat().st_mtime:
            by_parent[par] = p
    return sorted(by_parent.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path(
            "/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-evals/outputs/lmeval_megatron_support"
        ),
        help="Root directory to scan (legacy tree, one batch folder, or lmeval_megatron_batch)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/eval_results.csv)",
    )
    ap.add_argument(
        "--model-name",
        default=None,
        help="Override model_name for every row; default is derived per JSON from checkpoint path + step",
    )
    args = ap.parse_args()
    root: Path = args.root
    out_path = args.output or (root / "eval_results.csv")

    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    rows: list[tuple[str, str, int, float, str]] = []

    for jpath in pick_latest_json_per_output_dir(root):
        try:
            task, nshot_path = parse_task_and_path_n_shot(root, jpath)
        except ValueError as err:
            print(f"warning: skip {jpath}: {err}", file=sys.stderr)
            continue

        if task not in TASK_METRICS:
            print(f"warning: task {task!r} not in TASK_METRICS, skipping {jpath}", file=sys.stderr)
            continue

        metric = TASK_METRICS[task]
        with jpath.open(encoding="utf-8") as f:
            data = json.load(f)

        nshot = nshot_path if nshot_path is not None else n_shot_from_json(data, task)

        cfg = data.get("config") or {}
        model_name = args.model_name if args.model_name is not None else default_model_label(cfg)

        task_block = resolve_results_entry(data["results"], task)
        perf = metric_value(task_block, metric)
        metric_col = f"{metric},none"
        rows.append((model_name, task, nshot, perf, metric_col))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_name", "task", "n_shot", "performance", "metric_name"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4]])

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
