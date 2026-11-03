"""The code version a grid point is trained under: git HEAD plus a hash of any uncommitted change.

The grid logs this with every point, so `--resume` (and the --only union ranking) never ranks together points
trained on different code. A bare dirty flag is not enough for that: two different uncommitted edits on the same
commit would match. So a dirty tree is also stamped with the sha256 of its exact diff against HEAD (tracked
changes, binary-safe) plus the path and bytes of every untracked file, all restricted to CODE_PATHS -- the paths
the one-way freeze guards. Edits elsewhere (docs/, the grid's own untracked out-dir) do not change the stamp.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from orbit_guard.config import REPO_ROOT

CODE_PATHS = ("orbit_guard", "configs", "scripts", "data")
_GIT_TIMEOUT_S = 30
_UNKNOWN = {"git_commit": None, "git_dirty": None, "git_diff_sha256": None}


def _git(root: Path, *args: str) -> bytes:
    """stdout of `git -C root ...`; raises OSError/SubprocessError on any failure (caller maps it to unknown)."""
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True, timeout=_GIT_TIMEOUT_S
    ).stdout


def _diff_digest(root: Path) -> str:
    digest = hashlib.sha256(_git(root, "diff", "HEAD", "--binary", "--", *CODE_PATHS))
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *CODE_PATHS)
    for name in sorted(filter(None, untracked.split(b"\0"))):
        digest.update(b"\0untracked\0" + name + b"\0")
        digest.update((root / name.decode()).read_bytes())
    return digest.hexdigest()


def code_state(root: str | Path = REPO_ROOT) -> dict:
    """{"git_commit", "git_dirty", "git_diff_sha256"} of `root`; all None when git or the repository is unavailable.

    `git_diff_sha256` is None for a clean tree (under CODE_PATHS), else the digest described in the module docstring.
    """
    root = Path(root)
    try:
        commit = _git(root, "rev-parse", "--verify", "HEAD").decode().strip()
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
        dirty = bool(status.strip())
        return {"git_commit": commit, "git_dirty": dirty, "git_diff_sha256": _diff_digest(root) if dirty else None}
    except (OSError, subprocess.SubprocessError):
        return dict(_UNKNOWN)
