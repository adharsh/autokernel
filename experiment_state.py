"""Experiment state tracking for per-agent state persistence.

Tracks current_base, experiment_count, and best_speedup for each agent.
State is stored as JSON files under a configurable directory, one file per agent.
File locking via fcntl.flock ensures safe concurrent access.
"""

import fcntl
import json
import os
from dataclasses import asdict, dataclass


@dataclass
class AgentState:
    """Per-agent experiment state."""

    agent_id: int
    current_base: str       # e.g., "a0/0"
    experiment_count: int   # next experiment number
    best_speedup: float     # best speedup achieved so far


class ExperimentStateManager:
    """Manages per-agent experiment state as JSON files with file locking."""

    def __init__(self, state_dir: str = ".agent_state") -> None:
        self._state_dir = state_dir
        os.makedirs(self._state_dir, exist_ok=True)

    def _state_path(self, agent_id: int) -> str:
        """Return the JSON state file path for a given agent."""
        return os.path.join(self._state_dir, f"a{agent_id}.json")

    def _read_locked(self, agent_id: int) -> AgentState | None:
        """Read agent state from disk. Returns None if file does not exist.

        Caller must hold the file lock.
        """
        path = self._state_path(agent_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return AgentState(**data)

    def _write_locked(self, state: AgentState) -> None:
        """Write agent state to disk atomically.

        Caller must hold the file lock.
        """
        path = self._state_path(state.agent_id)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(asdict(state), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def get_or_create(self, agent_id: int) -> AgentState:
        """Get existing agent state or create a new default one.

        Default state: current_base="a{id}/0", experiment_count=1, best_speedup=1.0
        """
        path = self._state_path(agent_id)
        lock_path = path + ".lock"
        os.makedirs(os.path.dirname(lock_path) if os.path.dirname(lock_path) else self._state_dir, exist_ok=True)

        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_locked(agent_id)
                if state is None:
                    state = AgentState(
                        agent_id=agent_id,
                        current_base=f"a{agent_id}/0",
                        experiment_count=1,
                        best_speedup=1.0,
                    )
                    self._write_locked(state)
                return state
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def advance(self, agent_id: int, new_base: str, speedup: float) -> None:
        """Advance agent state: update current_base, best_speedup, and increment experiment_count.

        Called when an experiment is kept (correctness PASS and speedup improved).
        """
        path = self._state_path(agent_id)
        lock_path = path + ".lock"

        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_locked(agent_id)
                if state is None:
                    state = AgentState(
                        agent_id=agent_id,
                        current_base=f"a{agent_id}/0",
                        experiment_count=1,
                        best_speedup=1.0,
                    )
                state.current_base = new_base
                state.experiment_count += 1
                if speedup > state.best_speedup:
                    state.best_speedup = speedup
                self._write_locked(state)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def increment_only(self, agent_id: int) -> None:
        """Increment experiment_count without advancing base.

        Called when an experiment is discarded or crashes.
        """
        path = self._state_path(agent_id)
        lock_path = path + ".lock"

        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_locked(agent_id)
                if state is None:
                    state = AgentState(
                        agent_id=agent_id,
                        current_base=f"a{agent_id}/0",
                        experiment_count=1,
                        best_speedup=1.0,
                    )
                state.experiment_count += 1
                self._write_locked(state)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def get_next_experiment_id(self, agent_id: int) -> str:
        """Return the next experiment ID string: "a{id}/{count}".

        This reads the current experiment_count without incrementing it.
        """
        state = self.get_or_create(agent_id)
        return f"a{agent_id}/{state.experiment_count}"
