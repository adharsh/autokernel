"""Experiment recording utilities for the AutoKernel agent system.

The agent runs the experiment loop itself (git, editing, validation).
This module provides helpers for recording results and reading other agents' work.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from profile_utils import append_result, read_results


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


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

    # Get commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        commit = result.stdout.strip()
    except subprocess.CalledProcessError:
        commit = ""

    row = {
        "experiment_id": exp_id, "parent_id": parent_id,
        "agent_id": agent_id, "commit": commit,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate_us": candidate_us if candidate_us else "",
        "reference_us": reference_us if reference_us else "",
        "speedup": speedup, "correctness": correctness,
        "peak_vram_mb": peak_vram_mb if peak_vram_mb else "",
        "status": status, "description": description,
    }
    append_result(results_path, row)
    return row
