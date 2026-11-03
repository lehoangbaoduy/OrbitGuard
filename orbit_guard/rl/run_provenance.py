"""Git provenance of a training run, recorded in its run_config.json sidecar (Ray-free).

`git_state` is read when training STARTS, because the trainer and its Ray env-runner workers import the code
then: HEAD (`git_commit`) and the `git status --porcelain` lines under `TRAINING_CODE_PATHS` (`git_dirty`).
`git_drift` is read when it ends and adds a line for any change under those paths while the run was in
progress. The freeze (`evaluation.freeze`) refuses a checkpoint whose `git_dirty` is not an empty list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orbit_guard.evaluation.freeze import (
    TRAINING_CODE_PATHS,
    FrozenConfigError,
    GitTree,
)


def git_state(git: GitTree) -> dict[str, Any]:
    """{"git_commit": HEAD, "git_dirty": porcelain lines under TRAINING_CODE_PATHS}; both None without git."""
    try:
        return {"git_commit": git.head(), "git_dirty": git.dirty_paths(TRAINING_CODE_PATHS)}
    except FrozenConfigError:
        return {"git_commit": None, "git_dirty": None}


def git_drift(git: GitTree, start: Mapping[str, Any]) -> list[str]:
    """Lines for `git_dirty` describing changes under TRAINING_CODE_PATHS since `start` (empty when none).

    A run that started dirty is already refused, so only a clean start is compared commit-wise; new porcelain
    lines (e.g. an untracked module) are reported either way.
    """
    commit, start_dirty = start.get("git_commit"), start.get("git_dirty")
    if commit is None or start_dirty is None:
        return []
    try:
        end_dirty = git.dirty_paths(TRAINING_CODE_PATHS)
        moved = not start_dirty and git.changed_since(commit, TRAINING_CODE_PATHS)
    except FrozenConfigError as error:
        return [f"git unavailable when training ended, so changes while it ran are unknown: {error}"]
    paths = " ".join(TRAINING_CODE_PATHS)
    lines = [f"changed while training ran: git diff {commit} -- {paths} is not empty at the end"] if moved else []
    return lines + [f"appeared while training ran: {line}" for line in end_dirty if line not in start_dirty]
