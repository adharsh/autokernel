"""Append one experiment result to the shared TSV with file locking.

The official candidate timing is the sum of Nsight Compute kernel Duration rows
from the experiment's ncu/details.txt. validate.py is still used for
correctness, reference_us, and peak VRAM.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSV = ROOT / "results" / "experiments.tsv"
DEFAULT_EXPERIMENTS_DIR = ROOT / "results" / "experiments"
sys.path.insert(0, str(ROOT))

from profile_utils import append_result, init_results_tsv, ncu_duration_rows_us  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--status", required=True, choices=("keep", "discard", "crash"))
    parser.add_argument(
        "--interface-variant",
        default=os.environ.get("AUTOKERNEL_INTERFACE_VARIANT", "default"),
        help="Input/API representation used by this experiment, e.g. default or seq_idx.",
    )
    parser.add_argument("--description", required=True)
    parser.add_argument("--agent-id", default=os.environ.get("AGENT_ID"))
    parser.add_argument("--run-log", type=Path, default=Path("run.log"))
    parser.add_argument(
        "--ncu-details",
        type=Path,
        default=None,
        help=(
            "Path to Nsight Compute details.txt. Defaults to "
            "$AUTOKERNEL_EXPERIMENTS_DIR/<experiment_id>/ncu/details.txt."
        ),
    )
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
    import re

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


def default_ncu_details_path(experiment_id: str) -> Path:
    experiments_dir = Path(
        os.environ.get("AUTOKERNEL_EXPERIMENTS_DIR", DEFAULT_EXPERIMENTS_DIR)
    )
    safe_id = experiment_id.replace("/", "_")
    return experiments_dir / safe_id / "ncu" / "details.txt"


def read_ncu_details(path: Path, *, status: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        if status == "crash":
            return ""
        raise RuntimeError(
            f"Missing NCU details file: {path}. Run scripts/profile_ncu.sh for "
            "this experiment before recording the result."
        ) from None


def ncu_duration_metrics(text: str, *, status: str) -> tuple[str, str]:
    try:
        durations = ncu_duration_rows_us(text)
    except ValueError:
        if status == "crash":
            return "nan", "0"
        raise

    if not durations:
        if status == "crash":
            return "nan", "0"
        raise RuntimeError("Missing `Duration` rows in NCU details output")

    total_us = sum(durations)
    if not math.isfinite(total_us) or total_us <= 0:
        if status == "crash":
            return "nan", str(len(durations))
        raise RuntimeError(f"Invalid total NCU duration: {total_us}")

    return f"{total_us:.3f}", str(len(durations))


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def speedup(reference_us: str, ncu_duration: str) -> str:
    try:
        reference = float(reference_us)
        candidate = float(ncu_duration)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(reference) or not math.isfinite(candidate) or candidate <= 0:
        return "nan"
    return f"{reference / candidate:.6f}"


def metrics_from_log(text: str, *, status: str) -> dict[str, str]:
    if status == "crash":
        return {
            "reference_us": read_metric(text, "reference_us", required=False, default="nan"),
            "correctness": read_metric(text, "correctness", required=False, default="CRASH"),
            "peak_vram_mb": read_metric(text, "peak_vram_mb", required=False, default="nan"),
        }

    return {
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
    ncu_details_path = args.ncu_details or default_ncu_details_path(args.experiment_id)
    ncu_text = read_ncu_details(ncu_details_path, status=args.status)
    ncu_us, ncu_kernel_count = ncu_duration_metrics(ncu_text, status=args.status)

    if args.status == "keep" and metrics["correctness"] != "PASS":
        raise RuntimeError("Refusing to record a non-PASS result with status=keep")
    if args.status == "keep" and ncu_us == "nan":
        raise RuntimeError("Refusing to record a keep result without NCU duration")

    row = {
        "experiment_id": args.experiment_id,
        "parent_id": args.parent_id,
        "agent_id": args.agent_id,
        "commit": current_commit(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ncu_duration_us": ncu_us,
        "ncu_kernel_count": ncu_kernel_count,
        "reference_us": metrics["reference_us"],
        "speedup": speedup(metrics["reference_us"], ncu_us),
        "correctness": metrics["correctness"],
        "peak_vram_mb": metrics["peak_vram_mb"],
        "status": args.status,
        "interface_variant": args.interface_variant,
        "description": args.description,
    }

    init_results_tsv(str(args.tsv))
    append_result(str(args.tsv), row)
    print(
        f"recorded {row['experiment_id']} status={row['status']} "
        f"speedup={row['speedup']} interface={row['interface_variant']} "
        f"tsv={args.tsv}"
    )


if __name__ == "__main__":
    main()
