#!/usr/bin/env python3
"""
Run ``oellm collect-results`` on many result directories.

Supports:
  - One or more explicit directories (--dir, repeatable).
  - A file with one directory path per line (--dirs-file).
  - Timestamp-named subfolders under --base in a range: ``--since`` (inclusive
    lower bound), ``--until`` (inclusive upper bound), or both. Stamps look like
    ``2026-04-17-16-49-20`` or pass a path whose basename is the stamp.

Timestamp folders are matched as ``YYYY-MM-DD-HH-MM-SS`` (only these are
selected unless --all-subdirs is set).

With ``--save-merged-csv``, all per-run ``eval_results.csv`` files are
concatenated and written to ``merged.csv`` in the shared parent directory
(typically ``.../outputs/merged.csv``), unless ``--merged-csv`` sets an
explicit path. A ``run_folder`` column holds each run directory basename.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_OUTPUTS_BASE = Path(
    "/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-evals/outputs"
)
STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")


def _parse_stamp_name(label: str, raw: str) -> str:
    p = Path(raw)
    name = p.name if p.name else raw.rstrip("/")
    if not name:
        raise ValueError(f"Invalid {label} value: {raw!r}")
    return name


def _discover_since_dirs(
    base: Path,
    since_name: str,
    *,
    until_name: str | None,
    all_subdirs: bool,
) -> list[Path]:
    if not base.is_dir():
        raise FileNotFoundError(f"--base is not a directory: {base}")
    out: list[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if until_name is not None and p.name > until_name:
            continue
        if all_subdirs:
            if p.name >= since_name:
                out.append(p)
        else:
            if STAMP_RE.fullmatch(p.name) and p.name >= since_name:
                out.append(p)
    return out


def _load_dirs_file(path: Path) -> list[Path]:
    lines: list[Path] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(Path(line).expanduser())
    return lines


def _build_dir_list(args: argparse.Namespace) -> list[Path]:
    found: list[Path] = []
    for d in args.dir:
        found.append(Path(d).expanduser().resolve())
    found.extend(_load_dirs_file(Path(args.dirs_file).expanduser()) if args.dirs_file else [])
    if args.since or args.until:
        since_name = (
            _parse_stamp_name("--since", args.since) if args.since else ""
        )
        until_name = (
            _parse_stamp_name("--until", args.until) if args.until else None
        )
        if until_name is not None and since_name > until_name:
            raise ValueError(
                "--since must be <= --until (lexicographic order matches time for YYYY-MM-DD-HH-MM-SS)"
            )
        found.extend(
            _discover_since_dirs(
                Path(args.base).expanduser().resolve(),
                since_name,
                until_name=until_name,
                all_subdirs=args.all_subdirs,
            )
        )
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    return unique


def _infer_merged_csv_path(
    run_dirs: list[Path],
    *,
    fallback_base: Path | None,
) -> Path:
    """Parent is shared by all runs (siblings) -> ``<parent>/merged.csv``."""
    resolved = [p.resolve() for p in run_dirs if p.is_dir()]
    if not resolved:
        raise ValueError("merge: no existing result directories")
    parents = {p.parent for p in resolved}
    if len(parents) == 1:
        return next(iter(parents)) / "merged.csv"
    if fallback_base is not None and fallback_base.is_dir():
        return fallback_base.resolve() / "merged.csv"
    raise ValueError(
        "Runs are not all under the same parent; set --merged-csv to an explicit path."
    )


def _merge_eval_csvs(
    run_dirs: list[Path],
    merged_csv: Path,
    *,
    dry_run: bool,
) -> None:
    """Concatenate per-run ``eval_results.csv`` using the stdlib (no pandas)."""
    pieces: list[tuple[str, list[dict[str, str]]]] = []
    union_fields: list[str] = []
    field_set: set[str] = set()
    n_files = 0
    for results_dir in run_dirs:
        if not results_dir.is_dir():
            continue
        csv_path = results_dir / "eval_results.csv"
        if not csv_path.is_file():
            print(f"merge skip (no file): {csv_path}", file=sys.stderr)
            continue
        n_files += 1
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for k in reader.fieldnames or []:
                if k not in field_set:
                    field_set.add(k)
                    union_fields.append(k)
            file_rows = [dict(row) for row in reader]
        pieces.append((results_dir.name, file_rows))

    if not pieces:
        print("merge: no eval_results.csv files found; not writing merged.csv", file=sys.stderr)
        return
    fieldnames = ["run_folder"] + union_fields
    rows_out: list[dict[str, str]] = []
    for run_name, file_rows in pieces:
        for row in file_rows:
            out: dict[str, str] = {"run_folder": run_name}
            for k in union_fields:
                out[k] = row.get(k, "")
            rows_out.append(out)

    n_rows = len(rows_out)
    print(
        f"merge: {n_files} csv(s) -> {merged_csv} ({n_rows} rows)",
        flush=True,
    )
    if dry_run:
        return
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch oellm collect-results over output directories.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_OUTPUTS_BASE,
        help=(
            f"Parent of timestamp run folders for --since / --until (default: {DEFAULT_OUTPUTS_BASE})"
        ),
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        metavar="OUTPUT_DIR",
        help="Results directory (repeat for multiple)",
    )
    parser.add_argument(
        "--dirs-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="File with one output directory path per line (# comments allowed)",
    )
    parser.add_argument(
        "--since",
        metavar="STAMP_OR_PATH",
        help=(
            "Under --base, include run dirs whose name is >= this stamp (inclusive). "
            "Optional if --until is set (then there is no lower bound). "
            "Example: 2026-04-17-16-49-20"
        ),
    )
    parser.add_argument(
        "--until",
        metavar="STAMP_OR_PATH",
        help=(
            "Under --base, include run dirs whose name is <= this stamp (inclusive). "
            "Optional if --since is set (then there is no upper bound). "
            "Example: 2026-04-17-18-00-00"
        ),
    )
    parser.add_argument(
        "--all-subdirs",
        action="store_true",
        help=(
            "With --since / --until, include every matching subdirectory name "
            "(not only YYYY-MM-DD-HH-MM-SS folders)"
        ),
    )
    chk = parser.add_mutually_exclusive_group()
    chk.add_argument(
        "--check",
        dest="check",
        action="store_true",
        default=True,
        help="Pass --check true to oellm (default)",
    )
    chk.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="Pass --check false to oellm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to oellm",
    )
    parser.add_argument(
        "--save-merged-csv",
        action="store_true",
        help="After collect-results, merge each run's eval_results.csv into one file",
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for merged CSV (default: <shared-parent>/merged.csv)",
    )
    args = parser.parse_args()

    try:
        dirs = _build_dir_list(args)
    except ValueError as e:
        parser.error(str(e))
    if not dirs:
        parser.error(
            "Provide at least one of --dir, --dirs-file, --since, or --until"
        )

    merged_path: Path | None = None
    if args.save_merged_csv:
        merged_path = (
            args.merged_csv.expanduser().resolve()
            if args.merged_csv
            else _infer_merged_csv_path(
                dirs,
                fallback_base=Path(args.base).expanduser().resolve()
                if (args.since or args.until)
                else None,
            )
        )

    for results_dir in dirs:
        if not results_dir.is_dir():
            print(f"skip (not a directory): {results_dir}", file=sys.stderr)
            continue
        out_csv = results_dir / "eval_results.csv"
        cmd = [
            "oellm",
            "collect-results",
            "--results_dir",
            str(results_dir),
            "--output_csv",
            str(out_csv),
            "--check",
            "true" if args.check else "false",
        ]
        if args.verbose:
            cmd.append("--verbose")
        print("+", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            print(
                f"command failed ({r.returncode}) for {results_dir}",
                file=sys.stderr,
            )
            return r.returncode

    if merged_path is not None:
        try:
            _merge_eval_csvs(dirs, merged_path, dry_run=args.dry_run)
        except ValueError as e:
            print(f"merge error: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
