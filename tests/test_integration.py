"""Integration tests for the AutoKernel agent system.

Tests the full pipeline with mock data -- no GPU required.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest

# Ensure project root is on sys.path so imports resolve without installation.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from profile_utils import TSV_COLUMNS, append_result, init_results_tsv, read_results
from profile_utils import build_lineage_tree
from experiment_state import ExperimentStateManager
from analysis import classify_row, load_results, make_progress_plot, make_speedup_plot
from analysis import print_terminal_summary, print_delta_ranking
from agent_loop import check_cross_pollination


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(
    experiment_id: str = "a0/1",
    parent_id: str = "a0/0",
    agent_id: int = 0,
    commit: str = "abc1234",
    timestamp: str = "2026-03-20T14:00:00",
    candidate_us: float = 50.0,
    reference_us: float = 100.0,
    speedup: float = 2.0,
    correctness: str = "PASS",
    peak_vram_mb: float = 2048.0,
    status: str = "keep",
    description: str = "test change",
) -> dict:
    return {
        "experiment_id": experiment_id,
        "parent_id": parent_id,
        "agent_id": agent_id,
        "commit": commit,
        "timestamp": timestamp,
        "candidate_us": candidate_us,
        "reference_us": reference_us,
        "speedup": speedup,
        "correctness": correctness,
        "peak_vram_mb": peak_vram_mb,
        "status": status,
        "description": description,
    }


def _write_mock_tsv(path: str, rows: list[dict]) -> None:
    """Write a mock results.tsv with header + rows."""
    init_results_tsv(path)
    for row in rows:
        append_result(path, row)


# ---------------------------------------------------------------------------
# Test 1: TSV round-trip
# ---------------------------------------------------------------------------

class TestTsvRoundTrip:
    """init_results_tsv, append_result, read_results -- verify header, data, locking."""

    def test_init_creates_header(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)
        with open(tsv) as f:
            header = f.readline().strip()
        assert header == "\t".join(TSV_COLUMNS)

    def test_init_idempotent(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)
        init_results_tsv(tsv)  # second call should be a no-op
        with open(tsv) as f:
            lines = f.readlines()
        assert len(lines) == 1, "Header should only appear once"

    def test_append_and_read(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)

        row = _make_row(experiment_id="a0/1", candidate_us=42.3, speedup=2.01)
        append_result(tsv, row)

        results = read_results(tsv)
        assert len(results) == 1
        assert results[0]["experiment_id"] == "a0/1"
        assert results[0]["candidate_us"] == "42.3"
        assert results[0]["speedup"] == "2.01"

    def test_multiple_appends(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)

        for i in range(5):
            append_result(tsv, _make_row(experiment_id=f"a0/{i}"))

        results = read_results(tsv)
        assert len(results) == 5
        ids = [r["experiment_id"] for r in results]
        assert ids == [f"a0/{i}" for i in range(5)]

    def test_file_locking_no_corruption(self, tmp_path):
        """Verify that concurrent-style appends don't corrupt the file."""
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)

        # Simulate 20 rapid appends (sequential, but exercises the locking code)
        for i in range(20):
            append_result(tsv, _make_row(experiment_id=f"a{i % 3}/{i}"))

        results = read_results(tsv)
        assert len(results) == 20


# ---------------------------------------------------------------------------
# Test 2: Lineage tree
# ---------------------------------------------------------------------------

class TestLineageTree:
    """Build tree from parent_id chains; verify structure."""

    def test_single_chain(self):
        rows = [
            _make_row(experiment_id="a0/0", parent_id="-", status="keep",
                      description="baseline"),
            _make_row(experiment_id="a0/1", parent_id="a0/0", status="keep",
                      description="opt1"),
            _make_row(experiment_id="a0/2", parent_id="a0/1", status="discard",
                      description="opt2"),
        ]
        tree = build_lineage_tree(rows)
        assert tree["experiment_id"] == "__root__"
        assert len(tree["children"]) == 1  # a0/0 is the root

        a0_0 = tree["children"][0]
        assert a0_0["experiment_id"] == "a0/0"
        assert len(a0_0["children"]) == 1  # a0/1

        a0_1 = a0_0["children"][0]
        assert a0_1["experiment_id"] == "a0/1"
        assert len(a0_1["children"]) == 1  # a0/2

        a0_2 = a0_1["children"][0]
        assert a0_2["experiment_id"] == "a0/2"
        assert a0_2["status"] == "discard"
        assert len(a0_2["children"]) == 0

    def test_cross_agent_lineage(self):
        """a0/0 -> a0/1 -> a0/2, and a0/1 -> a1/3 (cross-agent)."""
        rows = [
            _make_row(experiment_id="a0/0", parent_id="-", agent_id=0),
            _make_row(experiment_id="a0/1", parent_id="a0/0", agent_id=0),
            _make_row(experiment_id="a0/2", parent_id="a0/1", agent_id=0),
            _make_row(experiment_id="a1/3", parent_id="a0/1", agent_id=1,
                      description="cross-agent adapt"),
        ]
        tree = build_lineage_tree(rows)

        a0_0 = tree["children"][0]
        assert a0_0["experiment_id"] == "a0/0"

        a0_1 = a0_0["children"][0]
        assert a0_1["experiment_id"] == "a0/1"
        # a0/1 should have two children: a0/2 and a1/3
        child_ids = sorted(c["experiment_id"] for c in a0_1["children"])
        assert child_ids == ["a0/2", "a1/3"]

    def test_multiple_roots(self):
        """Multiple agents with separate baselines produce multiple roots."""
        rows = [
            _make_row(experiment_id="a0/0", parent_id="-", agent_id=0),
            _make_row(experiment_id="a1/0", parent_id="-", agent_id=1),
            _make_row(experiment_id="a0/1", parent_id="a0/0", agent_id=0),
            _make_row(experiment_id="a1/1", parent_id="a1/0", agent_id=1),
        ]
        tree = build_lineage_tree(rows)
        root_ids = sorted(c["experiment_id"] for c in tree["children"])
        assert root_ids == ["a0/0", "a1/0"]


# ---------------------------------------------------------------------------
# Test 3: Experiment state
# ---------------------------------------------------------------------------

class TestExperimentState:
    """ExperimentStateManager: get_or_create, advance, increment_only."""

    def test_get_or_create_default(self, tmp_path):
        sm = ExperimentStateManager(state_dir=str(tmp_path / "state"))
        state = sm.get_or_create(0)
        assert state.agent_id == 0
        assert state.current_base == "a0/0"
        assert state.experiment_count == 1
        assert state.best_speedup == 1.0

    def test_get_or_create_idempotent(self, tmp_path):
        sm = ExperimentStateManager(state_dir=str(tmp_path / "state"))
        s1 = sm.get_or_create(0)
        s2 = sm.get_or_create(0)
        assert s1.agent_id == s2.agent_id
        assert s1.current_base == s2.current_base
        assert s1.experiment_count == s2.experiment_count

    def test_advance_updates_base_and_speedup(self, tmp_path):
        sm = ExperimentStateManager(state_dir=str(tmp_path / "state"))
        sm.get_or_create(0)
        sm.advance(0, new_base="a0/1", speedup=1.5)

        state = sm.get_or_create(0)
        assert state.current_base == "a0/1"
        assert state.best_speedup == 1.5
        assert state.experiment_count == 2

    def test_advance_only_improves_speedup(self, tmp_path):
        sm = ExperimentStateManager(state_dir=str(tmp_path / "state"))
        sm.get_or_create(0)
        sm.advance(0, new_base="a0/1", speedup=2.0)
        sm.advance(0, new_base="a0/2", speedup=1.5)  # worse speedup

        state = sm.get_or_create(0)
        assert state.current_base == "a0/2"
        assert state.best_speedup == 2.0  # should NOT decrease

    def test_increment_only_changes_count(self, tmp_path):
        sm = ExperimentStateManager(state_dir=str(tmp_path / "state"))
        sm.get_or_create(0)
        sm.increment_only(0)

        state = sm.get_or_create(0)
        assert state.current_base == "a0/0"  # unchanged
        assert state.best_speedup == 1.0  # unchanged
        assert state.experiment_count == 2  # incremented

    def test_next_experiment_id(self, tmp_path):
        sm = ExperimentStateManager(state_dir=str(tmp_path / "state"))
        assert sm.get_next_experiment_id(0) == "a0/1"
        sm.advance(0, "a0/1", 1.2)
        assert sm.get_next_experiment_id(0) == "a0/2"

    def test_multiple_agents_independent(self, tmp_path):
        sm = ExperimentStateManager(state_dir=str(tmp_path / "state"))
        sm.get_or_create(0)
        sm.get_or_create(1)
        sm.advance(0, "a0/1", 1.5)

        s0 = sm.get_or_create(0)
        s1 = sm.get_or_create(1)
        assert s0.current_base == "a0/1"
        assert s1.current_base == "a1/0"  # unaffected


# ---------------------------------------------------------------------------
# Test 4: classify_row
# ---------------------------------------------------------------------------

class TestClassifyRow:
    """classify_row with keep/discard/crash rows and edge cases."""

    def test_keep(self):
        row = {"status": "keep", "correctness": "PASS"}
        assert classify_row(row) == "keep"

    def test_kept_variant(self):
        row = {"status": "kept", "correctness": "PASS"}
        assert classify_row(row) == "keep"

    def test_discard(self):
        row = {"status": "discard", "correctness": "FAIL"}
        assert classify_row(row) == "discard"

    def test_crash(self):
        row = {"status": "crash", "correctness": "CRASH"}
        assert classify_row(row) == "crash"

    def test_infer_from_correctness_pass(self):
        row = {"status": "", "correctness": "PASS"}
        assert classify_row(row) == "keep"

    def test_infer_from_correctness_fail(self):
        row = {"status": "", "correctness": "FAIL"}
        assert classify_row(row) == "discard"

    def test_infer_from_correctness_crash(self):
        row = {"status": "", "correctness": "CRASH"}
        assert classify_row(row) == "crash"

    def test_infer_from_correctness_error(self):
        row = {"status": "", "correctness": "ERROR"}
        assert classify_row(row) == "crash"

    def test_nan_status(self):
        """NaN status should fall through to correctness-based classification."""
        import pandas as pd
        row = {"status": float("nan"), "correctness": "PASS"}
        assert classify_row(row) == "keep"

    def test_missing_fields(self):
        """Missing both status and correctness should default to discard."""
        row = {}
        assert classify_row(row) == "discard"

    def test_nan_correctness(self):
        import pandas as pd
        row = {"status": float("nan"), "correctness": float("nan")}
        assert classify_row(row) == "discard"


# ---------------------------------------------------------------------------
# Test 5: Analysis with mock data
# ---------------------------------------------------------------------------

class TestAnalysisWithMockData:
    """Create mock results.tsv, run analysis functions, verify outputs."""

    def _make_mock_rows(self) -> list[dict]:
        """Generate 12 mock experiment rows with varied statuses."""
        base_time = "2026-03-20T14:0"
        rows = []
        configs = [
            ("a0/0", "-", 0, 100.0, 100.0, 1.0, "PASS", "keep", "baseline"),
            ("a0/1", "a0/0", 0, 80.0, 100.0, 1.25, "PASS", "keep", "fuse norm"),
            ("a0/2", "a0/1", 0, 90.0, 100.0, 1.11, "PASS", "discard", "bad idea"),
            ("a0/3", "a0/1", 0, 60.0, 100.0, 1.67, "PASS", "keep", "shared mem"),
            ("a0/4", "a0/3", 0, 0.0, 100.0, 0.0, "CRASH", "crash", "segfault"),
            ("a0/5", "a0/3", 0, 50.0, 100.0, 2.0, "PASS", "keep", "vectorize"),
            ("a1/0", "-", 1, 100.0, 100.0, 1.0, "PASS", "keep", "baseline"),
            ("a1/1", "a1/0", 1, 72.0, 100.0, 1.39, "PASS", "keep", "loop unroll"),
            ("a1/2", "a1/1", 1, 85.0, 100.0, 1.18, "FAIL", "discard", "bad corr"),
            ("a1/3", "a1/1", 1, 65.0, 100.0, 1.54, "PASS", "keep", "warp shfl"),
            ("a1/4", "a1/3", 1, 0.0, 100.0, 0.0, "CRASH", "crash", "oom"),
            ("a1/5", "a1/3", 1, 55.0, 100.0, 1.82, "PASS", "keep", "tile 128"),
        ]
        for i, (eid, pid, aid, cus, rus, spd, corr, st, desc) in enumerate(configs):
            rows.append(_make_row(
                experiment_id=eid, parent_id=pid, agent_id=aid,
                candidate_us=cus, reference_us=rus, speedup=spd,
                correctness=corr, status=st, description=desc,
                timestamp=f"{base_time}{i:02d}:00",
            ))
        return rows

    def test_load_results(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        _write_mock_tsv(tsv, self._make_mock_rows())

        df = load_results(tsv)
        assert df is not None
        assert len(df) == 12

    def test_progress_plot_created(self, tmp_path, monkeypatch):
        import analysis
        tsv = str(tmp_path / "results.tsv")
        progress_png = str(tmp_path / "progress.png")
        _write_mock_tsv(tsv, self._make_mock_rows())

        monkeypatch.setattr(analysis, "PROGRESS_PNG", progress_png)
        df = load_results(tsv)
        make_progress_plot(df)
        assert os.path.exists(progress_png), "progress.png was not created"
        assert os.path.getsize(progress_png) > 0

    def test_speedup_plot_created(self, tmp_path, monkeypatch):
        import analysis
        tsv = str(tmp_path / "results.tsv")
        speedup_png = str(tmp_path / "speedup.png")
        _write_mock_tsv(tsv, self._make_mock_rows())

        monkeypatch.setattr(analysis, "SPEEDUP_PNG", speedup_png)
        df = load_results(tsv)
        make_speedup_plot(df)
        assert os.path.exists(speedup_png), "speedup.png was not created"
        assert os.path.getsize(speedup_png) > 0

    def test_terminal_summary_runs(self, tmp_path, capsys):
        tsv = str(tmp_path / "results.tsv")
        _write_mock_tsv(tsv, self._make_mock_rows())

        df = load_results(tsv)
        print_terminal_summary(df)
        captured = capsys.readouterr()
        assert "AutoKernel" in captured.out
        assert "Session Summary" in captured.out
        assert "Agent 0" in captured.out
        assert "Agent 1" in captured.out

    def test_delta_ranking_runs(self, tmp_path, capsys):
        tsv = str(tmp_path / "results.tsv")
        _write_mock_tsv(tsv, self._make_mock_rows())

        df = load_results(tsv)
        print_delta_ranking(df)
        captured = capsys.readouterr()
        assert "Delta Ranking" in captured.out


# ---------------------------------------------------------------------------
# Test 6: Cross-pollination
# ---------------------------------------------------------------------------

class TestCrossPollination:
    """check_cross_pollination filters other agents' kept experiments."""

    def test_filters_other_agents(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        rows = [
            _make_row(experiment_id="a0/0", agent_id=0, status="keep",
                      speedup=1.0, parent_id="-"),
            _make_row(experiment_id="a0/1", agent_id=0, status="keep",
                      speedup=1.5),
            _make_row(experiment_id="a1/0", agent_id=1, status="keep",
                      speedup=1.0, parent_id="-"),
            _make_row(experiment_id="a1/1", agent_id=1, status="keep",
                      speedup=2.0),
            _make_row(experiment_id="a1/2", agent_id=1, status="discard",
                      speedup=0.8),
        ]
        _write_mock_tsv(tsv, rows)

        # Agent 0 should see agent 1's kept experiments
        result = check_cross_pollination(tsv, agent_id=0)
        assert len(result) == 2
        eids = [r["experiment_id"] for r in result]
        assert "a1/1" in eids
        assert "a1/0" in eids
        # Should be sorted by speedup desc
        assert result[0]["experiment_id"] == "a1/1"

    def test_excludes_own_agent(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        rows = [
            _make_row(experiment_id="a0/0", agent_id=0, status="keep", speedup=1.0),
            _make_row(experiment_id="a0/1", agent_id=0, status="keep", speedup=2.0),
        ]
        _write_mock_tsv(tsv, rows)

        result = check_cross_pollination(tsv, agent_id=0)
        assert len(result) == 0, "Should not include own agent's experiments"

    def test_excludes_discard_crash(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        rows = [
            _make_row(experiment_id="a1/0", agent_id=1, status="discard", speedup=0.5),
            _make_row(experiment_id="a1/1", agent_id=1, status="crash", speedup=0.0),
        ]
        _write_mock_tsv(tsv, rows)

        result = check_cross_pollination(tsv, agent_id=0)
        assert len(result) == 0, "Should not include discard/crash experiments"

    def test_top5_limit(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        rows = [
            _make_row(experiment_id=f"a1/{i}", agent_id=1, status="keep",
                      speedup=1.0 + i * 0.1)
            for i in range(10)
        ]
        _write_mock_tsv(tsv, rows)

        result = check_cross_pollination(tsv, agent_id=0)
        assert len(result) == 5, "Should return at most 5 results"
        # Should be sorted by speedup descending
        speedups = [float(r["speedup"]) for r in result]
        assert speedups == sorted(speedups, reverse=True)

    def test_missing_file(self):
        result = check_cross_pollination("/nonexistent/results.tsv", agent_id=0)
        assert result == []
