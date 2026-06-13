"""
Merge ``eval_results.csv`` files from timestamped run folders under
``/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-evals/outputs`` within a
given time window.

Each run folder is expected to be named with a timestamp prefix of the form
``YYYY-MM-DD-HH-MM-SS`` (optionally followed by arbitrary suffixes, e.g.
``2026-04-05-09-34-48_tau_commonsense_qa_error``). Folders whose timestamp
falls in ``[--since, --until]`` (inclusive) and that contain an
``eval_results.csv`` file are merged.

The merged CSV is written to::

    <outputs_base>/merged_since_<since>_until_<until>/merged.csv

``--since`` is required. If ``--until`` is omitted, every folder from
``--since`` onwards is included.

Timestamps may be given either as ``YYYY-MM-DD-HH-MM-SS`` or as
``YYYY-MM-DD`` (interpreted as start-of-day for ``--since`` and end-of-day for
``--until``).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_BASE = Path(
    "/leonardo_work/OELLM_prod2026/users/shaldar0/oellm-evals/outputs"
)

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")
FULL_FMT = "%Y-%m-%d-%H-%M-%S"
DATE_FMT = "%Y-%m-%d"


def parse_timestamp(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse a CLI timestamp argument into a ``datetime``.

    Accepts ``YYYY-MM-DD-HH-MM-SS`` or ``YYYY-MM-DD``. When only a date is
    given, ``end_of_day`` controls whether to use 00:00:00 (default) or
    23:59:59.
    """
    try:
        return datetime.strptime(value, FULL_FMT)
    except ValueError:
        pass
    try:
        dt = datetime.strptime(value, DATE_FMT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid timestamp {value!r}; expected YYYY-MM-DD-HH-MM-SS or YYYY-MM-DD"
        ) from exc
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt


def folder_timestamp(folder_name: str) -> datetime | None:
    """Return the timestamp parsed from ``folder_name`` or ``None`` if absent."""
    match = TIMESTAMP_RE.match(folder_name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), FULL_FMT)
    except ValueError:
        return None


def collect_csvs(base: Path, since: datetime, until: datetime) -> list[Path]:
    csvs: list[tuple[datetime, Path]] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        ts = folder_timestamp(child.name)
        if ts is None:
            continue
        if ts < since or ts > until:
            continue
        csv_path = child / "eval_results.csv"
        if csv_path.is_file():
            csvs.append((ts, csv_path))
    csvs.sort(key=lambda x: x[0])
    return [p for _, p in csvs]


def merge_csvs(csv_paths: list[Path], output_path: Path) -> tuple[int, int]:
    """Merge CSVs into ``output_path``. Returns (rows_written, files_merged)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header: list[str] | None = None
    rows_written = 0
    files_merged = 0

    with output_path.open("w", newline="") as out_f:
        writer: csv.writer | None = None
        for csv_path in csv_paths:
            with csv_path.open("r", newline="") as in_f:
                reader = csv.reader(in_f)
                try:
                    file_header = next(reader)
                except StopIteration:
                    print(f"[warn] empty csv, skipping: {csv_path}", file=sys.stderr)
                    continue

                if header is None:
                    header = file_header
                    writer = csv.writer(out_f)
                    writer.writerow(header + ["source_run"])
                elif file_header != header:
                    print(
                        f"[warn] header mismatch in {csv_path}; "
                        f"expected {header} got {file_header}. Skipping.",
                        file=sys.stderr,
                    )
                    continue

                run_name = csv_path.parent.name
                assert writer is not None
                for row in reader:
                    writer.writerow(row + [run_name])
                    rows_written += 1
                files_merged += 1

    if header is None:
        output_path.unlink(missing_ok=True)
    return rows_written, files_merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--since",
        required=True,
        help="Inclusive lower bound timestamp (YYYY-MM-DD-HH-MM-SS or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Inclusive upper bound timestamp. If omitted, no upper bound is applied.",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Base outputs directory (default: {DEFAULT_BASE}).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override the merged folder name (default: merged_since_<since>_until_<until>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the CSVs that would be merged and the output path, without writing anything.",
    )
    args = parser.parse_args()

    since_dt = parse_timestamp(args.since, end_of_day=False)
    if args.until is None:
        until_dt = datetime.max
        until_label = "now"
    else:
        until_dt = parse_timestamp(args.until, end_of_day=True)
        until_label = args.until

    base: Path = args.base
    if not base.is_dir():
        print(f"[error] base directory does not exist: {base}", file=sys.stderr)
        return 1

    csv_paths = collect_csvs(base, since_dt, until_dt)
    if not csv_paths:
        print(
            f"[error] no eval_results.csv files found in {base} between "
            f"{args.since} and {until_label}",
            file=sys.stderr,
        )
        return 1

    folder_name = args.output_name or f"merged_since_{args.since}_until_{until_label}"
    output_dir = base / folder_name
    output_path = output_dir / "merged.csv"

    action = "Would merge" if args.dry_run else "Merging"
    print(f"{action} {len(csv_paths)} file(s) into {output_path}")
    for p in csv_paths:
        print(f"  - {p}")

    if args.dry_run:
        print("[dry-run] no files written.")
        return 0

    rows, files = merge_csvs(csv_paths, output_path)
    print(f"Done. Merged {files} file(s), {rows} data row(s) into {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
