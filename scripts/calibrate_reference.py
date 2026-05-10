"""Calibrate and store the reference NCU duration used by validate.py."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "reference_timing.json"
DEFAULT_NCU_DIR = ROOT / "results" / "reference_ncu"
DEFAULT_REFERENCE_WARMUP = 20
sys.path.insert(0, str(ROOT))

import validate  # noqa: E402
from profile_utils import ncu_duration_rows_us  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(
            os.environ.get("AUTOKERNEL_REFERENCE_NCU_WARMUP", DEFAULT_REFERENCE_WARMUP)
        ),
        help="Reference warmup calls before the single NCU-profiled invocation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("AUTOKERNEL_REFERENCE_TIMING_PATH", DEFAULT_OUTPUT)),
    )
    parser.add_argument(
        "--ncu-dir",
        type=Path,
        default=Path(os.environ.get("AUTOKERNEL_REFERENCE_NCU_DIR", DEFAULT_NCU_DIR)),
        help="Directory for reference NCU profile artifacts.",
    )
    parser.add_argument(
        "--ncu-set",
        default=os.environ.get("AUTOKERNEL_REFERENCE_NCU_SET", "full"),
        help="Nsight Compute section set for reference calibration.",
    )
    return parser.parse_args()


def ncu_reference_metrics(args: argparse.Namespace) -> tuple[float, int, Path, Path, Path]:
    args.ncu_dir.mkdir(parents=True, exist_ok=True)
    report_base = args.ncu_dir / "reference"
    report_path = report_base.with_suffix(".ncu-rep")
    log_path = args.ncu_dir / "reference.log"
    details_path = args.ncu_dir / "details.txt"

    env = os.environ.copy()
    env["AUTOKERNEL_REFERENCE_NCU_WARMUP"] = str(args.warmup)

    cmd = [
        "ncu",
        "--set",
        args.ncu_set,
        "--target-processes",
        "all",
        "--kernel-name-base",
        "demangled",
        "--force-overwrite",
        "--profile-from-start",
        "off",
        "-o",
        str(report_base),
        sys.executable,
        str(ROOT / "scripts" / "profile_reference_once.py"),
    ]

    with log_path.open("w") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    details = subprocess.check_output(
        ["ncu", "--import", str(report_path), "--page", "details"],
        cwd=ROOT,
        text=True,
    )
    details_path.write_text(details)
    durations = ncu_duration_rows_us(details)
    if not durations:
        raise RuntimeError("Missing `Duration` rows in reference NCU details output")
    total_us = sum(durations)
    if not math.isfinite(total_us) or total_us <= 0:
        raise RuntimeError(f"Invalid total reference NCU duration: {total_us}")
    return total_us, len(durations), report_path, log_path, details_path


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. NCU reference calibration must run on the GPU "
            "agent host."
        )

    timing_source = "ncu_duration_us"
    reference_us, ncu_kernel_count, report_path, log_path, details_path = (
        ncu_reference_metrics(args)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_us": round(reference_us, 3),
        "timing_source": timing_source,
        "warmup": args.warmup,
        "iters": 1,
        "ncu_kernel_count": ncu_kernel_count,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "cpu"
        ),
        "dtype": str(validate.benchmark_dtype()).replace("torch.", ""),
        "case": validate.STRESS_BENCHMARK_CASE.name,
    }
    payload.update(
        {
            "ncu_report": str(report_path),
            "ncu_log": str(log_path),
            "ncu_details": str(details_path),
            "ncu_set": args.ncu_set,
        }
    )

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"reference_us: {reference_us:.3f}")
    print(f"ncu_kernel_count: {ncu_kernel_count}")
    print(f"timing_source: {timing_source}")
    print(f"output: {args.output}")
    print(f"ncu_details: {details_path}")


if __name__ == "__main__":
    main()
