"""The held-out FINAL-TEST seed pool (PRD §33): 10 seeds x 5 conditions = 50 unique scenarios.

Lives in its own module so tuning code can be proven never to import it
(tests/tuning/test_grid_search.py). Loading it requires a valid freeze snapshot
(`evaluation.freeze`), so no model or config choice can change after the test
seeds are first read. The same seeds are reused for every method, which is what
makes paired comparisons possible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from orbit_guard.evaluation.freeze import verify_freeze
from orbit_guard.evaluation.seed_pools import TEST_SEED_RANGE, generate_seed_pool
from orbit_guard.evaluation.shifts import CONDITION_NAMES

TEST_MASTER_SEED = 20_261_003
SEEDS_PER_CONDITION = 10


def generate_test_pool(per_condition: int = SEEDS_PER_CONDITION) -> dict[str, tuple[int, ...]]:
    flat = generate_seed_pool(per_condition * len(CONDITION_NAMES), TEST_SEED_RANGE, TEST_MASTER_SEED)
    rng_order = list(flat)
    return {
        name: tuple(sorted(rng_order[index::len(CONDITION_NAMES)]))
        for index, name in enumerate(CONDITION_NAMES)
    }


def _checksum(pool: Mapping[str, tuple[int, ...]]) -> str:
    flattened = ";".join(f"{name}={','.join(str(s) for s in seeds)}" for name, seeds in pool.items())
    return hashlib.sha256(f"test:{flattened}".encode()).hexdigest()


def write_test_pool(path: str | Path) -> str:
    pool = generate_test_pool()
    checksum = _checksum(pool)
    record = {"split": "test", "conditions": {name: list(seeds) for name, seeds in pool.items()}, "sha256": checksum}
    Path(path).write_text(json.dumps(record, indent=2) + "\n")
    return checksum


def load_test_pool(
    path: str | Path,
    freeze_snapshot: str | Path,
    checkpoints: Mapping[str, str | Path],
    config_path: str | Path | None = None,
) -> dict[str, tuple[int, ...]]:
    """Return the test pool -- only if configs and checkpoints are frozen and unchanged."""
    if config_path is None:
        verify_freeze(freeze_snapshot, checkpoints)
    else:
        verify_freeze(freeze_snapshot, checkpoints, config_path)
    record = json.loads(Path(path).read_text())
    if record["split"] != "test":
        raise ValueError(f"{path} is a {record['split']!r} pool, expected 'test'")
    pool = {name: tuple(int(s) for s in seeds) for name, seeds in record["conditions"].items()}
    if _checksum(pool) != record["sha256"]:
        raise ValueError(f"{path} fails its own checksum -- the seed list was edited after generation")
    return pool
