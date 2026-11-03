"""Files a freeze test needs, outside the throwaway repository: config, overlay, both pools and one checkpoint.

The checkpoint's run_config.json carries what the freeze reads from a real training sidecar: the config_digest,
and the git commit and (empty) dirty list the run recorded when it started (`training_sidecar`).
"""

from __future__ import annotations

import json
from pathlib import Path

from orbit_guard.config import config_digest
from orbit_guard.evaluation.seed_pools import generate_validation_pool, write_seed_pool
from orbit_guard.evaluation.test_seed_pool import write_test_pool
from tests.evaluation.git_repo_fixture import git


def training_sidecar(repo: Path, config: Path | None = None, overlay: Path | None = None, **fields) -> dict:
    """A sidecar trained at `repo`'s HEAD from a clean tree; `fields` override or add keys."""
    record = {"git_commit": git(repo, "rev-parse", "HEAD"), "git_dirty": []}
    if config is not None:
        record["config_digest"] = config_digest(config, overlay)
    return {**record, **fields}


def write_checkpoint(run: Path, sidecar: dict | None, weights: bytes = b"weights-v1") -> Path:
    """actor_weights.npz (+ run_config.json unless `sidecar` is None) in `run`; returns the weights path."""
    run.mkdir(parents=True, exist_ok=True)
    (run / "actor_weights.npz").write_bytes(weights)
    if sidecar is not None:
        (run / "run_config.json").write_text(json.dumps(sidecar))
    return run / "actor_weights.npz"


def build_world(files: Path, repo: Path) -> dict:
    files.mkdir()
    config = files / "default.yaml"
    config.write_text("a: 1\n")
    overlay = files / "tuned.yaml"
    overlay.write_text("reward:\n  w3: 1.5\n")
    test_pool = files / "test_seeds.json"
    write_test_pool(test_pool)
    validation_pool = files / "validation_seeds.json"
    write_seed_pool(validation_pool, "validation", generate_validation_pool())
    run = files / "n15_adaptive_s0"
    weights = write_checkpoint(run, training_sidecar(repo, config, overlay))
    return {
        "snapshot": files / "frozen.json", "config": config, "overlay": overlay, "test_pool": test_pool,
        "validation_pool": validation_pool, "run": run, "repo": repo,
        "checkpoints": {"adaptive_mappo/0": weights},
    }
