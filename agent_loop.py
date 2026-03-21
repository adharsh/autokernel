"""Single-experiment executor for the AutoKernel agent system.

Usage: python agent_loop.py --agent-id 0 --description "fuse norm+proj"
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from experiment_state import ExperimentStateManager
from git_utils import (
    checkout_branch, commit_candidate, create_experiment_branch, get_short_commit,
)
from profile_utils import append_result, read_results


def _safe_float(val: str | float | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def run_validation() -> dict:
    """Run ``uv run python validate.py``, parse key: value output.

    On subprocess failure returns ``{"correctness": "CRASH"}``.
    """
    try:
        result = subprocess.run(
            ["uv", "run", "python", "validate.py"],
            capture_output=True, text=True, timeout=600,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {"correctness": "CRASH"}
    if result.returncode != 0:
        return {"correctness": "CRASH"}

    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        parsed[key.strip()] = value.strip()
    return parsed


def check_cross_pollination(results_path: str, agent_id: int) -> list[dict]:
    """Top-5 kept experiments from *other* agents, sorted by speedup desc."""
    try:
        rows = read_results(results_path)
    except FileNotFoundError:
        return []

    other_kept = [
        r for r in rows
        if r.get("status") == "keep"
        and str(r.get("agent_id", "")) != str(agent_id)
    ]
    for r in other_kept:
        try:
            r["_speedup"] = float(r.get("speedup", 0))
        except (ValueError, TypeError):
            r["_speedup"] = 0.0
    other_kept.sort(key=lambda r: r["_speedup"], reverse=True)
    top = other_kept[:5]
    for r in top:
        r.pop("_speedup", None)
    return top


def record_experiment(
    results_path: str, exp_id: str, parent_id: str,
    agent_id: int, description: str, validation: dict,
) -> dict:
    """Build a full results row from validation output and append to TSV."""
    correctness = validation.get("correctness", "CRASH")
    candidate_us = _safe_float(validation.get("candidate_us"))
    reference_us = _safe_float(validation.get("reference_us"))
    peak_vram_mb = _safe_float(validation.get("peak_vram_mb"))

    if candidate_us and candidate_us > 0 and reference_us:
        speedup = round(reference_us / candidate_us, 4)
    else:
        speedup = 0.0

    if correctness == "CRASH":
        status = "crash"
    elif correctness != "PASS" or speedup <= 0:
        status = "discard"
    else:
        status = "keep"

    row = {
        "experiment_id": exp_id, "parent_id": parent_id,
        "agent_id": agent_id, "commit": get_short_commit(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate_us": candidate_us if candidate_us else "",
        "reference_us": reference_us if reference_us else "",
        "speedup": speedup, "correctness": correctness,
        "peak_vram_mb": peak_vram_mb if peak_vram_mb else "",
        "status": status, "description": description,
    }
    append_result(results_path, row)
    return row


def run_experiment(
    agent_id: int, description: str, results_path: str = "results.tsv",
) -> dict:
    """Execute one full experiment cycle and return the result row."""
    sm = ExperimentStateManager()
    state = sm.get_or_create(agent_id)
    exp_id = sm.get_next_experiment_id(agent_id)
    parent_id = state.current_base

    create_experiment_branch(exp_id, parent_id)
    commit_candidate(exp_id, description)
    validation = run_validation()
    row = record_experiment(
        results_path, exp_id, parent_id, agent_id, description, validation,
    )

    if row["status"] == "keep":
        sm.advance(agent_id, exp_id, float(row["speedup"] or 1.0))
    else:
        sm.increment_only(agent_id)
        checkout_branch(parent_id)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AutoKernel experiment.")
    parser.add_argument("--agent-id", type=int, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--results-path", type=str, default="results.tsv")
    args = parser.parse_args()

    row = run_experiment(args.agent_id, args.description, args.results_path)
    for key in ("experiment_id", "status", "correctness", "speedup",
                "candidate_us", "reference_us"):
        if row.get(key, "") != "":
            print(f"{key}: {row[key]}")


if __name__ == "__main__":
    main()
