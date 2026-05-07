"""Append one validate.py run to the shared experiments TSV with file locking."""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = ROOT / "results" / "experiments.tsv"
sys.path.insert(0, str(ROOT))

from profile_utils import append_result, init_results_tsv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--status", required=True, choices=("keep", "discard", "crash"))
    parser.add_argument("--description", required=True)
    parser.add_argument("--agent-id", default=os.environ.get("AGENT_ID"))
    parser.add_argument("--run-log", type=Path, default=Path("run.log"))
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path(os.environ.get("AUTOKERNEL_EXPERIMENTS_TSV", DEFAULT_TSV)),
    )
    return parser.parse_args()


def read_metric(
    text: str,
    key: str,
    *,
    required: bool = True,
    default: str = "",
) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        if required:
            raise RuntimeError(f"Missing `{key}:` in run log")
        return default
    return match.group(1).strip()


def read_run_log(path: Path, *, status: str) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        if status == "crash":
            return ""
        raise


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def speedup(reference_us: str, candidate_us: str) -> str:
    try:
        reference = float(reference_us)
        candidate = float(candidate_us)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(reference) or not math.isfinite(candidate) or candidate <= 0:
        return "nan"
    return f"{reference / candidate:.6f}"


def metrics_from_log(text: str, *, status: str) -> dict[str, str]:
    if status == "crash":
        return {
            "candidate_us": read_metric(text, "candidate_us", required=False, default="nan"),
            "reference_us": read_metric(text, "reference_us", required=False, default="nan"),
            "correctness": read_metric(text, "correctness", required=False, default="CRASH"),
            "peak_vram_mb": read_metric(text, "peak_vram_mb", required=False, default="nan"),
        }

    return {
        "candidate_us": read_metric(text, "candidate_us"),
        "reference_us": read_metric(text, "reference_us"),
        "correctness": read_metric(text, "correctness"),
        "peak_vram_mb": read_metric(text, "peak_vram_mb"),
    }


def main() -> None:
    args = parse_args()
    if not args.agent_id:
        raise RuntimeError("Missing --agent-id or AGENT_ID environment variable")

    text = read_run_log(args.run_log, status=args.status)
    metrics = metrics_from_log(text, status=args.status)

    if args.status == "keep" and metrics["correctness"] != "PASS":
        raise RuntimeError("Refusing to record a non-PASS result with status=keep")

    row = {
        "experiment_id": args.experiment_id,
        "parent_id": args.parent_id,
        "agent_id": args.agent_id,
        "commit": current_commit(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidate_us": metrics["candidate_us"],
        "reference_us": metrics["reference_us"],
        "speedup": speedup(metrics["reference_us"], metrics["candidate_us"]),
        "correctness": metrics["correctness"],
        "peak_vram_mb": metrics["peak_vram_mb"],
        "status": args.status,
        "description": args.description,
    }

    init_results_tsv(str(args.tsv))
    append_result(str(args.tsv), row)
    print(
        f"recorded {row['experiment_id']} status={row['status']} "
        f"speedup={row['speedup']} tsv={args.tsv}"
    )


if __name__ == "__main__":
    main()
