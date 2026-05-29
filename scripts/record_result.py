"""Append one experiment result to the shared TSV with file locking.

The official candidate timing is the sum of Nsight Compute kernel Duration rows
from the experiment's basic profile at ncu/details.txt. validate.py is still
used for correctness, reference_us, peak VRAM, and quality diagnostics. Every
recorded experiment must already have note.md in its experiment artifact
directory.
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

from profile_utils import (  # noqa: E402
    append_result,
    init_results_tsv,
    ncu_duration_rows_us,
    read_results,
)


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
            "Official basic Nsight Compute details.txt path. Defaults to "
            "$AUTOKERNEL_EXPERIMENTS_DIR/<experiment_id>/ncu/details.txt; "
            "custom paths are rejected for official TSV timing."
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


def experiment_artifact_dir(experiment_id: str) -> Path:
    experiments_dir = Path(
        os.environ.get("AUTOKERNEL_EXPERIMENTS_DIR", DEFAULT_EXPERIMENTS_DIR)
    )
    safe_id = experiment_id.replace("/", "_")
    return experiments_dir / safe_id


def default_ncu_details_path(experiment_id: str) -> Path:
    return experiment_artifact_dir(experiment_id) / "ncu" / "details.txt"


def default_supplemental_ncu_details_path(experiment_id: str, profile_set: str) -> Path:
    return experiment_artifact_dir(experiment_id) / "ncu" / profile_set / "details.txt"


def default_note_path(experiment_id: str) -> Path:
    return experiment_artifact_dir(experiment_id) / "note.md"


def read_ncu_details(path: Path, *, status: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        if status == "crash":
            return ""
        raise RuntimeError(
            f"Missing NCU details file: {path}. Run "
            "scripts/profile_ncu.sh <experiment_id> basic for this experiment "
            "before recording the result."
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


def is_baseline_experiment(experiment_id: str, parent_id: str) -> bool:
    return parent_id == "-" and experiment_id.rsplit("/", 1)[-1] == "0"


def require_detailed_for_keep(args: argparse.Namespace) -> None:
    if args.status != "keep" or is_baseline_experiment(
        args.experiment_id,
        args.parent_id,
    ):
        return
    if os.environ.get("AUTOKERNEL_REQUIRE_DETAILED_FOR_KEEP", "1") == "0":
        return

    detailed_path = default_supplemental_ncu_details_path(
        args.experiment_id,
        "detailed",
    )
    if not detailed_path.is_file() or detailed_path.stat().st_size == 0:
        raise RuntimeError(
            f"Refusing to record status=keep without detailed NCU evidence: "
            f"{detailed_path}. Run "
            f"`scripts/profile_ncu.sh {args.experiment_id} detailed` before "
            "recording a non-baseline keep result."
        )


def require_note(args: argparse.Namespace) -> None:
    note_path = default_note_path(args.experiment_id)
    if not note_path.is_file():
        raise RuntimeError(
            f"Refusing to record {args.experiment_id} without a note: "
            f"{note_path}. Write note.md before recording the TSV row."
        )
    if not note_path.read_text(encoding="utf-8", errors="replace").strip():
        raise RuntimeError(
            f"Refusing to record {args.experiment_id} with an empty note: "
            f"{note_path}. Fill in note.md before recording the TSV row."
        )


def require_official_basic_details(args: argparse.Namespace) -> None:
    if args.ncu_details is None:
        return

    expected = default_ncu_details_path(args.experiment_id).resolve()
    actual = args.ncu_details.resolve()
    if actual != expected:
        raise RuntimeError(
            f"Refusing custom NCU details path for official TSV timing: {actual}. "
            f"Use the basic profile at {expected}. Supplemental detailed/full "
            "profiles belong in note.md, not TSV timing fields."
        )


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def current_commit() -> str:
    return git_output("rev-parse", "--short", "HEAD")


def ensure_experiment_branch(experiment_id: str) -> str:
    head = git_output("rev-parse", "HEAD")
    active_branch = git_output("branch", "--show-current")
    ref = f"refs/heads/{experiment_id}"
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 1:
        subprocess.check_call(["git", "branch", experiment_id, "HEAD"])
        branch_status = "created"
    elif result.returncode == 0 and result.stdout.strip() == head:
        branch_status = "verified"
    elif result.returncode == 0:
        raise RuntimeError(
            f"Experiment branch {experiment_id} points to "
            f"{result.stdout.strip()[:7]}, but HEAD is {head[:7]}. "
            "Refusing to move an existing experiment branch."
        )
    else:
        raise RuntimeError(result.stderr.strip() or f"Could not inspect {ref}")

    if active_branch and active_branch != experiment_id:
        raise RuntimeError(
            f"Current branch is {active_branch}, but experiment_id is "
            f"{experiment_id}. Experiment branch was {branch_status} at HEAD; "
            "switch to it before recording."
        )

    return branch_status


def speedup(reference_us: str, ncu_duration: str) -> str:
    try:
        reference = float(reference_us)
        candidate = float(ncu_duration)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(reference) or not math.isfinite(candidate) or candidate <= 0:
        return "nan"
    return f"{reference / candidate:.6f}"


def finite_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def best_recorded_keep_speedup(tsv_path: Path, interface_variant: str) -> float | None:
    """Best finalized keep row for this compatible interface variant."""
    if not tsv_path.exists():
        return None

    best: float | None = None
    for row in read_results(str(tsv_path)):
        if row.get("status") != "keep" or row.get("correctness") != "PASS":
            continue
        if str(row.get("interface_variant", "")) != interface_variant:
            continue
        row_speedup = finite_float(row.get("speedup"))
        if row_speedup is None:
            continue
        if best is None or row_speedup > best:
            best = row_speedup
    return best


def require_fast_pass_not_discarded(args: argparse.Namespace, row: dict[str, str]) -> None:
    if args.status != "discard" or row["correctness"] != "PASS":
        return

    row_speedup = finite_float(row["speedup"])
    if row_speedup is None:
        return

    best_keep = best_recorded_keep_speedup(args.tsv, args.interface_variant)
    if best_keep is None or row_speedup <= best_keep:
        return

    if os.environ.get("AUTOKERNEL_ALLOW_FAST_DISCARD", "0") == "1":
        print(
            "warning: recording a PASS discard that beats the best recorded keep; "
            "this should only be used for a genuinely disqualified result explained "
            "in note.md",
            file=sys.stderr,
        )
        return

    raise RuntimeError(
        f"Refusing to record {args.experiment_id} as discard: speedup "
        f"{row_speedup:.6f} beats the best recorded keep for "
        f"interface_variant={args.interface_variant!r} ({best_keep:.6f}). "
        "Pending/unrecorded notes are not a valid reason to discard a completed "
        "PASS speedup. Record it as keep after the required detailed profile, or "
        "set AUTOKERNEL_ALLOW_FAST_DISCARD=1 only for a genuinely disqualified "
        "result explained in note.md."
    )


def require_training_forward_interface(args: argparse.Namespace) -> None:
    if args.status != "keep":
        return

    variant = args.interface_variant.lower()
    forbidden = (
        "fp8_weight",
        "fp8_state",
        "weight_state",
        "weight_cache",
        "weight_scale",
        "warmup_cache",
        "runtime_cache",
        "scale_metadata",
        "scale_state",
        "activation_scale",
        "fixed_scale",
        "static_scale",
        "delayed_scale",
        "xscale",
        "x_scale",
        "precompute",
        "precomputed",
        "prepack",
        "prepacked",
        "maintained",
        "cached",
    )
    matched = [term for term in forbidden if term in variant]
    if not matched:
        return

    raise RuntimeError(
        f"Refusing to record status=keep for interface_variant="
        f"{args.interface_variant!r}: it looks like prebuilt FP8/scale state "
        f"({', '.join(matched)}) rather than the current training-forward "
        "contract. Official keeps must compute FP8 weight/scale work inside "
        "the measured candidate invocation. If the human changes the benchmark "
        "contract later, update record_result.py deliberately instead of using "
        "an environment bypass."
    )


def require_training_quality_evidence(args: argparse.Namespace, text: str) -> None:
    if args.status != "keep":
        return

    if "stress_quality:" not in text:
        raise RuntimeError(
            "Refusing to record status=keep: run log is missing stress_quality. "
            "Do not skip stress quality for official keep rows."
        )

    diagnostic_status = read_metric(
        text,
        "diagnostic_quality_status",
        required=False,
        default="",
    )
    if diagnostic_status != "PASS":
        found = diagnostic_status or "missing"
        raise RuntimeError(
            "Refusing to record status=keep: diagnostic_quality_status must be "
            f"PASS, got {found!r}. Do not skip diagnostic quality for official "
            "keep rows."
        )


def parse_utc_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def session_start_path(tsv_path: Path) -> Path:
    return tsv_path.parent / "session_started_at.txt"


def read_session_start(tsv_path: Path) -> datetime | None:
    path = session_start_path(tsv_path)
    try:
        return parse_utc_timestamp(path.read_text().strip())
    except FileNotFoundError:
        return None


def experiment_elapsed_s(tsv_path: Path, agent_id: str, now: datetime) -> str:
    """Wall-clock seconds since this agent's previous row or session start."""
    session_start = read_session_start(tsv_path)
    previous: datetime | None = None
    if tsv_path.exists():
        for row in read_results(str(tsv_path)):
            if str(row.get("agent_id", "")) != str(agent_id):
                continue
            ts = parse_utc_timestamp(str(row.get("timestamp", "")))
            if ts is None:
                continue
            if session_start is not None and ts < session_start:
                continue
            if previous is None or ts > previous:
                previous = ts

    start = previous or session_start
    if start is None:
        return ""
    elapsed = max(0.0, (now - start).total_seconds())
    return f"{elapsed:.0f}"


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
    require_note(args)
    require_official_basic_details(args)
    ncu_details_path = args.ncu_details or default_ncu_details_path(args.experiment_id)
    ncu_text = read_ncu_details(ncu_details_path, status=args.status)
    ncu_us, ncu_kernel_count = ncu_duration_metrics(ncu_text, status=args.status)
    require_detailed_for_keep(args)
    require_training_forward_interface(args)
    require_training_quality_evidence(args, text)

    if args.status == "keep" and metrics["correctness"] != "PASS":
        raise RuntimeError("Refusing to record a non-PASS result with status=keep")
    if args.status == "keep" and ncu_us == "nan":
        raise RuntimeError("Refusing to record a keep result without NCU duration")

    branch_status = ensure_experiment_branch(args.experiment_id)
    init_results_tsv(str(args.tsv))
    now = datetime.now(timezone.utc)

    row = {
        "experiment_id": args.experiment_id,
        "parent_id": args.parent_id,
        "agent_id": args.agent_id,
        "commit": current_commit(),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ncu_duration_us": ncu_us,
        "ncu_kernel_count": ncu_kernel_count,
        "reference_us": metrics["reference_us"],
        "speedup": speedup(metrics["reference_us"], ncu_us),
        "correctness": metrics["correctness"],
        "peak_vram_mb": metrics["peak_vram_mb"],
        "status": args.status,
        "interface_variant": args.interface_variant,
        "description": args.description,
        "experiment_elapsed_s": experiment_elapsed_s(args.tsv, args.agent_id, now),
    }

    require_fast_pass_not_discarded(args, row)

    append_result(str(args.tsv), row)
    print(
        f"recorded {row['experiment_id']} status={row['status']} "
        f"speedup={row['speedup']} interface={row['interface_variant']} "
        f"elapsed_s={row['experiment_elapsed_s']} "
        f"branch={branch_status} tsv={args.tsv}"
    )


if __name__ == "__main__":
    main()
