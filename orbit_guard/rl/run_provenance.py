"""Git provenance of a training run, recorded in its run_config.json sidecar (Ray-free).

`git_state` is read when training STARTS, because the trainer and its Ray env-runner workers import the code
then: HEAD (`git_commit`) and the `git status --porcelain` lines under `TRAINING_CODE_PATHS` (`git_dirty`).
`git_drift` is read when it ends and adds a line for any change under those paths while the run was in
progress. The freeze (`evaluation.freeze`) refuses a checkpoint whose `git_dirty` is not an empty list.

`run_dir_status` (CLI: `python -m orbit_guard.rl.run_provenance <run dir>... [--overlay X]`, used by
scripts/queue_n15.sh) says whether a run directory is "done" under the config + overlay a queue trains with (its
run_config.json records that config_digest), "todo" (no actor weights yet) or "foreign" (weights trained under
another configuration, or with no readable sidecar), so a queue never reuses nor overwrites a foreign checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.evaluation.freeze import (
    RUN_CONFIG_FILENAME,
    TRAINING_CODE_PATHS,
    FrozenConfigError,
    GitTree,
)

ACTOR_WEIGHTS_FILENAME = "actor_weights.npz"  # rl/training.py's name (importing it would pull in Ray)
FOREIGN_EXIT_CODE = 3


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


def run_dir_status(
    run_dir: str | Path, *, config_path: str | Path = DEFAULT_CONFIG_PATH, overlay_path: str | Path | None = None
) -> tuple[str, str]:
    """("done" | "todo" | "foreign", reason): see the module docstring. The reason is empty unless foreign."""
    run = Path(run_dir)
    if not (run / ACTOR_WEIGHTS_FILENAME).exists():
        return "todo", ""
    sidecar = run / RUN_CONFIG_FILENAME
    try:
        trained = json.loads(sidecar.read_text())["config_digest"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        return "foreign", f"{run} holds actor weights but no readable {RUN_CONFIG_FILENAME} with a config_digest ({error!r})"
    expected = config_digest(config_path, overlay_path)
    if trained != expected:
        return "foreign", f"{run} was trained under config_digest {trained}, this queue trains under {expected}"
    return "done", ""


def main(argv: list[str] | None = None) -> None:
    """Print "<status> <run dir>" per run dir; exit FOREIGN_EXIT_CODE (reasons on stderr) if any is foreign."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--overlay", default=None, help="overlay the queue trains with ('' = none)")
    args = parser.parse_args(argv)
    overlay = args.overlay or None
    foreign = []
    for run_dir in args.run_dirs:
        status, reason = run_dir_status(run_dir, config_path=args.config, overlay_path=overlay)
        print(f"{status} {run_dir}")
        if status == "foreign":
            foreign.append(reason)
    if foreign:
        print("\n".join(foreign), file=sys.stderr)
        raise SystemExit(FOREIGN_EXIT_CODE)


if __name__ == "__main__":
    main()
