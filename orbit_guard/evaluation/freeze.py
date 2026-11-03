"""One-way freeze snapshot: the gate in front of the final-test pool (PRD §33, P4-M4 exit / P5-M2 entry).

PRD §33 holds the final test "strictly held out until every model/config choice is frozen". That is
enforced, not promised:

  * `write_freeze_snapshot` writes ONCE: it refuses if the snapshot file exists, and refuses unless
    `git status --porcelain -- orbit_guard configs scripts` is empty (edits under docs/ do not block).
    It records config_digest(config, overlay) (CORE-1), the sha256 of the test and validation pool files,
    every checkpoint's actor_weights sha256 and its run_config.json sha256, and HEAD. A checkpoint with no
    run_config.json sidecar is refused: its training configuration could not be shown.
  * `verify_freeze` re-checks every recorded hash, that the checkpoints passed are exactly the frozen
    (model, training seed) set (adding or dropping a training seed after seeing results fails), that
    those paths are still clean, and that
    `git diff --quiet <frozen commit> -- orbit_guard configs scripts` holds: no code, config or script
    change since the freeze, committed or not. Any failure raises `FrozenConfigError` naming what moved.
  * the final-test loader (`test_seed_pool.load_test_pool`) calls `verify_freeze`, so retraining,
    retuning or editing the code after the freeze closes the gate.

Git runs in the repository root (`GIT_ROOT`, resolved at call time), never the working directory, and
every entry point takes an injectable `GitTree` so tests use a throwaway repository. A tuned overlay that
lives under configs/ must therefore be committed before freezing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from orbit_guard.config import DEFAULT_CONFIG_PATH, REPO_ROOT, config_digest

GIT_ROOT = REPO_ROOT
FROZEN_PATHS = ("orbit_guard", "configs", "scripts")
SNAPSHOT_FORMAT = 2
DEFAULT_TEST_POOL = REPO_ROOT / "configs" / "test_seeds.json"
DEFAULT_VALIDATION_POOL = REPO_ROOT / "configs" / "validation_seeds.json"
RUN_CONFIG_FILENAME = "run_config.json"  # written by rl/training.py next to actor_weights.npz
_GIT_TIMEOUT_S = 30
_POOLS = ("test", "validation")


class FrozenConfigError(RuntimeError):
    """The configuration, a seed pool, a checkpoint or the code changed after the freeze, or was never frozen."""


class _RecordedOverlay:
    """Sentinel: verify against the overlay path the snapshot recorded (its bytes must still match)."""

    def __repr__(self) -> str:
        return "RECORDED_OVERLAY"


RECORDED_OVERLAY = _RecordedOverlay()


@dataclass(frozen=True)
class GitTree:
    """Git queries against one repository root (`git -C root ...`), independent of the working directory."""

    root: Path

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", "-C", str(self.root), *args],
                capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, check=False,  # callers read returncode
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise FrozenConfigError(f"git is unavailable for {self.root}: {error}") from error

    def head(self) -> str:
        completed = self._git("rev-parse", "--verify", "HEAD")
        if completed.returncode != 0:
            raise FrozenConfigError(f"cannot read the git commit of {self.root}: {completed.stderr.strip()}")
        return completed.stdout.strip()

    def dirty_paths(self, paths: Sequence[str]) -> list[str]:
        """`git status --porcelain` lines for `paths` (modified, staged, deleted or untracked files)."""
        completed = self._git("status", "--porcelain", "--untracked-files=all", "--", *paths)
        if completed.returncode != 0:
            raise FrozenConfigError(f"git status failed in {self.root}: {completed.stderr.strip()}")
        return [line for line in completed.stdout.splitlines() if line.strip()]

    def changed_since(self, commit: str, paths: Sequence[str]) -> bool:
        """True if the working tree under `paths` differs from `commit` (`git diff --quiet`)."""
        if self._git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            raise FrozenConfigError(f"the frozen commit {commit} is unknown to the repository at {self.root}")
        completed = self._git("diff", "--quiet", commit, "--", *paths)
        if completed.returncode not in (0, 1):
            raise FrozenConfigError(f"git diff against {commit} failed: {completed.stderr.strip()}")
        return completed.returncode == 1


def _default_git() -> GitTree:
    return GitTree(Path(GIT_ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    """HEAD of the repository (not the cwd), or "unknown" when git is unavailable; for episode metadata."""
    try:
        return _default_git().head()
    except FrozenConfigError:
        return "unknown"


def _require_clean(git: GitTree, action: str) -> None:
    dirty = git.dirty_paths(FROZEN_PATHS)
    if dirty:
        raise FrozenConfigError(
            f"cannot {action}: uncommitted changes under {'/, '.join(FROZEN_PATHS)}/ "
            f"(commit them first; docs/ is exempt): {'; '.join(dirty)}"
        )


def _checkpoint_record(weights: Path) -> dict:
    sidecar = weights.parent / RUN_CONFIG_FILENAME
    return {
        "actor_weights_path": str(weights),
        "actor_weights_sha256": _sha256(weights),
        "run_config_sha256": _sha256(sidecar) if sidecar.exists() else None,
    }


def _require_sidecars(checkpoints: Mapping[str, str | Path]) -> None:
    """Every frozen checkpoint must carry the run_config.json that records the config it was trained under.

    The provenance check in experiment.py still accepts a sidecar-less checkpoint for validation runs, but the
    freeze is the gate to the final-test pool: a checkpoint whose training configuration cannot be shown never
    passes it. (Every checkpoint passed here is a learned model's; scripted models have none.)
    """
    missing = sorted(
        name for name, file in checkpoints.items() if not (Path(file).parent / RUN_CONFIG_FILENAME).exists()
    )
    if missing:
        raise FrozenConfigError(
            f"cannot freeze checkpoints without a {RUN_CONFIG_FILENAME} sidecar (their training config is "
            f"unrecorded; rl/training.py writes it next to actor_weights.npz): {missing}"
        )


def _pool_record(path: str | Path, split: str) -> dict:
    pool = Path(path)
    if not pool.exists():
        raise FrozenConfigError(f"the {split} pool file {pool} does not exist; write it before freezing")
    return {"path": str(pool), "sha256": _sha256(pool)}


def write_freeze_snapshot(
    path: str | Path,
    checkpoints: Mapping[str, str | Path],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    overlay_path: str | Path | None = None,
    *,
    test_pool_path: str | Path = DEFAULT_TEST_POOL,
    validation_pool_path: str | Path = DEFAULT_VALIDATION_POOL,
    git: GitTree | None = None,
) -> dict:
    """Freeze config + overlay, both seed pools, every checkpoint and the commit, once; return the snapshot."""
    snapshot_path = Path(path)
    if snapshot_path.exists():
        raise FrozenConfigError(f"freeze snapshot {snapshot_path} already exists: the freeze is one-way")
    _require_sidecars(checkpoints)
    git = git or _default_git()
    _require_clean(git, "freeze")
    snapshot = {
        "format": SNAPSHOT_FORMAT,
        "config_path": str(config_path),
        "overlay_path": None if overlay_path is None else str(overlay_path),
        "config_digest": config_digest(config_path, overlay_path),
        "pools": {
            "test": _pool_record(test_pool_path, "test"),
            "validation": _pool_record(validation_pool_path, "validation"),
        },
        "checkpoints": {name: _checkpoint_record(Path(file)) for name, file in sorted(checkpoints.items())},
        "git_commit": git.head(),
        "frozen_paths": list(FROZEN_PATHS),
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    with snapshot_path.open("x") as handle:  # exclusive create: never overwrite, even in a race
        handle.write(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def _load_snapshot(path: Path) -> dict:
    if not path.exists():
        raise FrozenConfigError(f"no freeze snapshot at {path}: freeze configs and checkpoints before the final test")
    try:
        snapshot = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise FrozenConfigError(f"freeze snapshot {path} is not valid JSON: {error}") from error
    if not isinstance(snapshot, dict) or snapshot.get("format") != SNAPSHOT_FORMAT:
        raise FrozenConfigError(
            f"freeze snapshot {path} has an old or unknown format (expected format {SNAPSHOT_FORMAT}); "
            "it predates the one-way freeze and cannot be verified"
        )
    return snapshot


def _verify_config(snapshot: dict, config_path: str | Path | None, overlay_path) -> None:
    config = snapshot["config_path"] if config_path is None else config_path
    overlay = snapshot["overlay_path"] if overlay_path is RECORDED_OVERLAY else overlay_path
    current = config_digest(config, overlay)
    if current != snapshot["config_digest"]:
        raise FrozenConfigError(
            f"config changed since it was frozen: config {config!s} + overlay {overlay!s} has digest {current}, "
            f"the freeze recorded {snapshot['config_digest']} (config {snapshot['config_path']}, "
            f"overlay {snapshot['overlay_path']})"
        )


def _verify_pools(snapshot: dict, overrides: Mapping[str, str | Path | None]) -> None:
    for split in _POOLS:
        recorded = snapshot["pools"][split]
        pool = Path(overrides[split] or recorded["path"])
        if not pool.exists() or _sha256(pool) != recorded["sha256"]:
            raise FrozenConfigError(f"the {split} pool file {pool} changed (or is missing) since the freeze")


def _verify_checkpoint_set(snapshot: dict, checkpoints: Mapping[str, str | Path]) -> None:
    """The (model, training seed) checkpoints passed must be exactly the frozen set: none added, none dropped."""
    frozen, passed = set(snapshot["checkpoints"]), set(checkpoints)
    if frozen == passed:
        return
    problems = []
    if passed - frozen:
        problems.append(f"never frozen, so they cannot enter the final test: {sorted(passed - frozen)}")
    if frozen - passed:
        problems.append(f"frozen but not passed (missing; a training seed cannot be dropped): {sorted(frozen - passed)}")
    raise FrozenConfigError(f"the checkpoints differ from the frozen set: {'; '.join(problems)}")


def _verify_checkpoints(snapshot: dict, checkpoints: Mapping[str, str | Path]) -> None:
    _verify_checkpoint_set(snapshot, checkpoints)
    for name, file in checkpoints.items():
        recorded = snapshot["checkpoints"][name]
        current = _checkpoint_record(Path(file))
        if current["actor_weights_sha256"] != recorded["actor_weights_sha256"]:
            raise FrozenConfigError(f"checkpoint {name} changed since the freeze snapshot (actor weights {file})")
        if current["run_config_sha256"] != recorded["run_config_sha256"]:
            raise FrozenConfigError(
                f"checkpoint {name}: its {RUN_CONFIG_FILENAME} changed, appeared or disappeared since the freeze"
            )


def _verify_code(snapshot: dict, git: GitTree) -> None:
    _require_clean(git, "verify the freeze")
    commit = snapshot["git_commit"]
    if git.changed_since(commit, FROZEN_PATHS):
        raise FrozenConfigError(
            f"code, configs or scripts changed since the freeze at commit {commit} "
            f"(git diff {commit} -- {' '.join(FROZEN_PATHS)} is not empty)"
        )


def verify_freeze(
    path: str | Path,
    checkpoints: Mapping[str, str | Path],
    config_path: str | Path | None = None,
    overlay_path: str | Path | None | _RecordedOverlay = RECORDED_OVERLAY,
    *,
    test_pool_path: str | Path | None = None,
    validation_pool_path: str | Path | None = None,
    git: GitTree | None = None,
) -> dict:
    """Raise `FrozenConfigError` unless the snapshot exists and config, pools, checkpoints and code still match.

    `checkpoints` must name exactly the frozen set ("model/training_seed" -> actor weights), none added or dropped.

    `config_path`/`overlay_path` name the configuration about to be evaluated; left out, the files the snapshot
    recorded are re-hashed (the overlay may be passed as None explicitly to mean "no overlay"). Pool paths
    likewise default to the recorded ones.
    """
    snapshot = _load_snapshot(Path(path))
    _verify_config(snapshot, config_path, overlay_path)
    _verify_pools(snapshot, {"test": test_pool_path, "validation": validation_pool_path})
    _verify_checkpoints(snapshot, checkpoints)
    _verify_code(snapshot, git or _default_git())
    return snapshot
