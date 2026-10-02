"""Fixed, checksummed scenario-seed pools for the validation split (PRD §33, P4-M4).

Seed-space layout (so splits cannot collide by construction):
  * [0, 100_000)            ad-hoc development seeds (smoke tests, the Phase 3 gate's 2000-2039 / 5000-5039)
  * [100_000, 200_000)      the VALIDATION pool -- 50 seeds drawn once, then frozen in a file
  * [200_000, 300_000)      reserved for the FINAL-TEST pool (Phase 5; lives in its own module that
                            tuning code never imports -- see tests/tuning/test_grid_search_guard.py)
  * [2**30, 2**31)          training episodes: `OrbitGuardEnv.reset(seed=None)` draws from here

A pool file records its split and a SHA-256 over the split name and the seed
list; loading verifies it, so a hand-edited or swapped file fails loudly
instead of silently changing what "validation" means mid-project.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

VALIDATION_SEED_RANGE = (100_000, 200_000)
TEST_SEED_RANGE = (200_000, 300_000)
TRAINING_SEED_FLOOR = 2**30
VALIDATION_MASTER_SEED = 20_261_002  # fixed once; the pool is then read from the committed file


def pool_checksum(split: str, seeds: tuple[int, ...]) -> str:
    payload = f"{split}:{','.join(str(seed) for seed in seeds)}".encode()
    return hashlib.sha256(payload).hexdigest()


def generate_seed_pool(
    count: int, seed_range: tuple[int, int], master_seed: int
) -> tuple[int, ...]:
    """`count` distinct seeds from `seed_range`, drawn reproducibly and returned sorted."""
    low, high = seed_range
    if count < 1 or count > high - low:
        raise ValueError(f"cannot draw {count} distinct seeds from [{low}, {high})")
    drawn = np.random.default_rng(master_seed).choice(np.arange(low, high), size=count, replace=False)
    return tuple(sorted(int(seed) for seed in drawn))


def write_seed_pool(path: str | Path, split: str, seeds: tuple[int, ...]) -> str:
    """Write the pool file and return its checksum."""
    checksum = pool_checksum(split, seeds)
    Path(path).write_text(json.dumps({"split": split, "seeds": list(seeds), "sha256": checksum}, indent=2) + "\n")
    return checksum


def load_seed_pool(path: str | Path, expected_split: str, expected_sha256: str | None = None) -> tuple[int, ...]:
    """Load a pool, refusing a wrong split or any checksum mismatch."""
    record = json.loads(Path(path).read_text())
    if record["split"] != expected_split:
        raise ValueError(f"{path} is a {record['split']!r} pool, expected {expected_split!r}")
    seeds = tuple(int(seed) for seed in record["seeds"])
    checksum = pool_checksum(record["split"], seeds)
    if checksum != record["sha256"]:
        raise ValueError(f"{path} fails its own checksum -- the seed list was edited after generation")
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError(f"{path} checksum {checksum} != the configured {expected_sha256}")
    return seeds


def generate_validation_pool(count: int = 50) -> tuple[int, ...]:
    return generate_seed_pool(count, VALIDATION_SEED_RANGE, VALIDATION_MASTER_SEED)


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/validation_seeds.json")
    print(write_seed_pool(target, "validation", generate_validation_pool()))
