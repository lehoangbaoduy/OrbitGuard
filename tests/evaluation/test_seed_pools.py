"""Tests for orbit_guard.evaluation.seed_pools (PRD §33: fixed, checksummed, disjoint pools)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_guard.evaluation.seed_pools import (
    TEST_SEED_RANGE,
    TRAINING_SEED_FLOOR,
    VALIDATION_SEED_RANGE,
    generate_seed_pool,
    generate_validation_pool,
    load_seed_pool,
    write_seed_pool,
)

_COMMITTED_POOL = Path(__file__).resolve().parents[2] / "configs" / "validation_seeds.json"


def test_validation_pool_has_fifty_distinct_seeds_inside_its_reserved_range():
    pool = generate_validation_pool()

    assert len(pool) == len(set(pool)) == 50
    assert all(VALIDATION_SEED_RANGE[0] <= seed < VALIDATION_SEED_RANGE[1] for seed in pool)


def test_validation_pool_is_reproducible():
    assert generate_validation_pool() == generate_validation_pool()


def test_reserved_ranges_are_disjoint_from_each_other_and_from_training_seeds():
    assert VALIDATION_SEED_RANGE[1] <= TEST_SEED_RANGE[0]
    assert TEST_SEED_RANGE[1] <= TRAINING_SEED_FLOOR


def test_pool_cannot_exceed_its_range():
    with pytest.raises(ValueError):
        generate_seed_pool(11, (0, 10), master_seed=0)


def test_round_trip_and_checksum_verification(tmp_path):
    path = tmp_path / "pool.json"
    checksum = write_seed_pool(path, "validation", (5, 6, 7))

    assert load_seed_pool(path, "validation", expected_sha256=checksum) == (5, 6, 7)


def test_loading_the_wrong_split_is_refused(tmp_path):
    path = tmp_path / "pool.json"
    write_seed_pool(path, "test", (1, 2))

    with pytest.raises(ValueError, match="expected 'validation'"):
        load_seed_pool(path, "validation")


def test_a_tampered_pool_fails_its_checksum(tmp_path):
    path = tmp_path / "pool.json"
    write_seed_pool(path, "validation", (1, 2, 3))
    record = json.loads(path.read_text())
    record["seeds"] = [1, 2, 4]
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="checksum"):
        load_seed_pool(path, "validation")


def test_a_mismatch_with_the_configured_checksum_is_refused(tmp_path):
    path = tmp_path / "pool.json"
    write_seed_pool(path, "validation", (1, 2, 3))

    with pytest.raises(ValueError, match="configured"):
        load_seed_pool(path, "validation", expected_sha256="0" * 64)


def test_the_committed_validation_pool_matches_the_frozen_generator():
    assert load_seed_pool(_COMMITTED_POOL, "validation") == generate_validation_pool()
