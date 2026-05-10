"""
Profiling and results utilities for the AutoKernel agent system.

Provides:
- ncu_duration_rows_us: Nsight Compute duration parser
- TSV file locking utilities for shared results/experiments.tsv
- Lineage tree construction from experiment history
"""

from __future__ import annotations

import csv
import fcntl
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# TSV schema
# ---------------------------------------------------------------------------

TSV_COLUMNS: list[str] = [
    "experiment_id",
    "parent_id",
    "agent_id",
    "commit",
    "timestamp",
    "ncu_duration_us",
    "ncu_kernel_count",
    "reference_us",
    "speedup",
    "correctness",
    "peak_vram_mb",
    "status",
    "description",
]

_NCU_DURATION_RE = re.compile(
    r"^\s*Duration\s+"
    r"(?P<unit>nsecond|usecond|msecond|second|ns|us|ms|s)\s+"
    r"(?P<value>[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    re.MULTILINE,
)


def ncu_duration_rows_us(details_text: str) -> list[float]:
    """Return all Nsight Compute kernel Duration rows converted to microseconds."""
    scale = {
        "nsecond": 1e-3,
        "ns": 1e-3,
        "usecond": 1.0,
        "us": 1.0,
        "msecond": 1e3,
        "ms": 1e3,
        "second": 1e6,
        "s": 1e6,
    }
    durations: list[float] = []
    for match in _NCU_DURATION_RE.finditer(details_text):
        value = float(match.group("value").replace(",", ""))
        durations.append(value * scale[match.group("unit")])
    return durations


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


def _tsv_cell(value: Any) -> str:
    """Format one TSV cell without allowing embedded row/column separators."""
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


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

    line = "\t".join(_tsv_cell(row.get(col, "")) for col in columns) + "\n"

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
