"""Freeze snapshot: the gate in front of the final-test pool (PRD §33, P4-M4 exit / P5-M2 entry).

"Freeze tuned params before final test" is enforced, not promised: the final-test
loader (`test_seed_pool.load_test_pool`) refuses to run unless a snapshot exists
whose config hash and per-checkpoint weight hashes still match the files on
disk. Retrain or retune after peeking and the gate closes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class FrozenConfigError(RuntimeError):
    """The configuration or a checkpoint changed after it was frozen, or was never frozen."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_freeze_snapshot(
    path: str | Path, checkpoints: Mapping[str, str | Path], config_path: str | Path = _DEFAULT_CONFIG
) -> dict:
    """Record the config hash and every checkpoint's weight hash; return the snapshot."""
    snapshot = {
        "config_hash": _sha256(Path(config_path)),
        "checkpoints": {name: _sha256(Path(file)) for name, file in sorted(checkpoints.items())},
        "git_commit": _git_commit(),
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    Path(path).write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def verify_freeze(
    path: str | Path, checkpoints: Mapping[str, str | Path], config_path: str | Path = _DEFAULT_CONFIG
) -> dict:
    """Raise `FrozenConfigError` unless the snapshot exists and still matches config + checkpoints."""
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FrozenConfigError(f"no freeze snapshot at {path}: freeze configs and checkpoints before the final test")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot["config_hash"] != _sha256(Path(config_path)):
        raise FrozenConfigError("config changed since it was frozen")
    for name, file in checkpoints.items():
        if snapshot["checkpoints"].get(name) != _sha256(Path(file)):
            raise FrozenConfigError(f"checkpoint {name} changed (or was never frozen) since the freeze snapshot")
    return snapshot
