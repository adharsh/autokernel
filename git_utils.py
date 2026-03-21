"""Git worktree management utilities for the AutoKernel agent system.

Thin wrappers around subprocess calls for experiment branch management.
All functions raise on errors via subprocess check=True.
"""

import subprocess


def create_experiment_branch(exp_id: str, base_branch: str) -> None:
    """Create a new experiment branch from a base branch.

    Runs: git checkout -b {exp_id} {base_branch}
    """
    subprocess.run(
        ["git", "checkout", "-b", exp_id, base_branch],
        check=True,
        capture_output=True,
        text=True,
    )


def commit_candidate(exp_id: str, description: str) -> str:
    """Stage candidate/ and commit with a message, return 7-char commit hash.

    Runs: git add candidate/ && git commit -m "{exp_id}: {description}"
    Returns the 7-character short commit hash.
    """
    subprocess.run(
        ["git", "add", "candidate/"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"{exp_id}: {description}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return get_short_commit()


def checkout_branch(branch: str) -> None:
    """Check out an existing branch.

    Runs: git checkout {branch}
    """
    subprocess.run(
        ["git", "checkout", branch],
        check=True,
        capture_output=True,
        text=True,
    )


def get_short_commit() -> str:
    """Return the 7-character short hash of HEAD.

    Runs: git rev-parse --short HEAD
    """
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def show_file_at_branch(branch: str, filepath: str) -> str:
    """Return the contents of a file at a given branch without checking it out.

    Runs: git show {branch}:{filepath}
    """
    result = subprocess.run(
        ["git", "show", f"{branch}:{filepath}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def diff_branches(base: str, target: str, path: str = "candidate/") -> str:
    """Return the diff between two branches for a given path.

    Runs: git diff {base}..{target} -- {path}
    """
    result = subprocess.run(
        ["git", "diff", f"{base}..{target}", "--", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
