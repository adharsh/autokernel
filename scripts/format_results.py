"""Print results/experiments.tsv as an aligned human-readable table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = ROOT / "results" / "experiments.tsv"

sys.path.insert(0, str(ROOT))

from profile_utils import TSV_COLUMNS, read_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", nargs="?", type=Path, default=DEFAULT_TSV)
    parser.add_argument(
        "--sort",
        choices=("file", "timestamp", "experiment_id", "agent"),
        default="file",
        help="Display order. The TSV file itself is not modified.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show only the last N rows after sorting; 0 means all rows.",
    )
    return parser.parse_args()


def sort_rows(rows: list[dict], mode: str) -> list[dict]:
    def agent_id(row: dict) -> int:
        try:
            return int(row.get("agent_id") or -1)
        except ValueError:
            return -1

    if mode == "timestamp":
        return sorted(rows, key=lambda r: r.get("timestamp", ""))
    if mode == "experiment_id":
        return sorted(rows, key=lambda r: r.get("experiment_id", ""))
    if mode == "agent":
        return sorted(
            rows,
            key=lambda r: (
                agent_id(r),
                r.get("timestamp", ""),
                r.get("experiment_id", ""),
            ),
        )
    return rows


def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def main() -> None:
    args = parse_args()
    if not args.tsv.exists():
        raise SystemExit(f"No results TSV found: {args.tsv}")

    rows = sort_rows(read_results(str(args.tsv)), args.sort)
    if args.limit > 0:
        rows = rows[-args.limit :]
    if not rows:
        print("No experiment rows found.")
        return
    print_table(rows, TSV_COLUMNS)


if __name__ == "__main__":
    main()
