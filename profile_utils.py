"""
Profiling and results utilities for the AutoKernel agent system.

Provides:
- cuda_timer / cpu_timer: Microbenchmark timing functions
- TSV file locking utilities for shared results/experiments.tsv
- Lineage tree construction from experiment history
"""

from __future__ import annotations

import csv
import fcntl
import os
import statistics
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# TSV schema
# ---------------------------------------------------------------------------

TSV_COLUMNS: list[str] = [
    "experiment_id",
    "parent_id",
    "agent_id",
    "commit",
    "timestamp",
    "candidate_us",
    "reference_us",
    "speedup",
    "correctness",
    "peak_vram_mb",
    "status",
    "description",
]

# ---------------------------------------------------------------------------
# Timer utilities
# ---------------------------------------------------------------------------


def cuda_timer(
    fn: Callable,
    *args: Any,
    warmup: int = 10,
    iters: int = 100,
    **kwargs: Any,
) -> dict:
    """Time a GPU operation using CUDA events.

    Returns dict with keys: median_ms, mean_ms, min_ms, max_ms, std_ms.
    """
    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))  # ms

    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }


def cpu_timer(
    fn: Callable,
    *args: Any,
    warmup: int = 10,
    iters: int = 100,
    sync_cuda: bool = True,
    **kwargs: Any,
) -> dict:
    """Time an operation using CPU wall clock.

    sync_cuda=True (default): includes GPU kernel completion time.
    sync_cuda=False: pure CPU operations only.

    Returns dict with keys: median_ms, mean_ms, min_ms, max_ms, std_ms.
    """
    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# TSV file locking utilities
# ---------------------------------------------------------------------------


def init_results_tsv(tsv_path: str) -> None:
    """Create the results TSV with a header row if it doesn't exist or is empty.

    Uses exclusive locking to prevent races between concurrent agents.
    """
    header_line = "\t".join(TSV_COLUMNS) + "\n"
    parent = os.path.dirname(tsv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Open in r+ if file exists so we can check its size, otherwise create.
    fd = os.open(tsv_path, os.O_RDWR | os.O_CREAT, 0o644)

    with os.fdopen(fd, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Only write header if the file is empty (new or truncated).
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(header_line)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_result(
    tsv_path: str,
    row: dict,
    columns: list[str] | None = None,
) -> None:
    """Atomically append a row to the shared results TSV.

    Uses fcntl.flock with LOCK_EX for exclusive access so multiple
    agents can safely write concurrently.
    """
    if columns is None:
        columns = TSV_COLUMNS

    parent = os.path.dirname(tsv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    line = "\t".join(str(row.get(col, "")) for col in columns) + "\n"

    with open(tsv_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_results(tsv_path: str) -> list[dict]:
    """Read a results TSV with a shared lock (allows concurrent reads).

    Returns a list of dicts, one per row, keyed by column name.
    """
    with open(tsv_path, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Lineage tracking
# ---------------------------------------------------------------------------


def build_lineage_tree(results: list[dict]) -> dict:
    """Build a tree structure from parent_id relationships.

    Each node in the returned tree has the shape::

        {
            "experiment_id": str,
            "children": [<child nodes>],
            "status": str,
            "speedup": float | None,
            "description": str,
        }

    Baseline rows (parent_id == "-") become root nodes.  All roots are
    collected under a single virtual root whose experiment_id is "__root__".

    Returns the virtual root dict.
    """
    # Index all rows by experiment_id
    nodes: dict[str, dict] = {}
    for row in results:
        eid = row.get("experiment_id", "")
        if not eid:
            continue
        speedup_raw = row.get("speedup", "")
        try:
            speedup = float(speedup_raw)
        except (ValueError, TypeError):
            speedup = None

        nodes[eid] = {
            "experiment_id": eid,
            "children": [],
            "status": row.get("status", ""),
            "speedup": speedup,
            "description": row.get("description", ""),
        }

    # Build parent -> children links
    roots: list[dict] = []
    for row in results:
        eid = row.get("experiment_id", "")
        pid = row.get("parent_id", "")
        if not eid or eid not in nodes:
            continue
        node = nodes[eid]

        if pid == "-" or pid == "" or pid not in nodes:
            roots.append(node)
        else:
            nodes[pid]["children"].append(node)

    return {
        "experiment_id": "__root__",
        "children": roots,
        "status": "",
        "speedup": None,
        "description": "virtual root",
    }
