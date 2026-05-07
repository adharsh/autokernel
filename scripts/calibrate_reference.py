"""Calibrate and store the reference runtime used by validate.py."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "reference_timing.json"
sys.path.insert(0, str(ROOT))

import validate  # noqa: E402
from reference import kernel_fn as reference_kernel_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU calibration. By default calibration requires CUDA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("AUTOKERNEL_REFERENCE_TIMING_PATH", DEFAULT_OUTPUT)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Run calibration on the GPU agent host, or pass "
            "--allow-cpu only for local CPU smoke tests."
        )

    bench_args = validate.make_inputs()

    with torch.no_grad():
        reference_us = validate.time_us(
            lambda: reference_kernel_fn(*bench_args),
            warmup=args.warmup,
            iters=args.iters,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_us": round(reference_us, 3),
        "warmup": args.warmup,
        "iters": args.iters,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "cpu"
        ),
        "dtype": str(validate.benchmark_dtype()).replace("torch.", ""),
        "case": validate.STRESS_BENCHMARK_CASE.name,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"reference_us: {reference_us:.3f}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
