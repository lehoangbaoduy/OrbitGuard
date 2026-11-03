"""Final-test seed pool and the freeze-before-test rule (PRD §33, §6 structural protection)."""

from __future__ import annotations

import json

import pytest

from orbit_guard.evaluation import freeze
from orbit_guard.evaluation.freeze import (
    FrozenConfigError,
    verify_freeze,
    write_freeze_snapshot,
)
from orbit_guard.evaluation.seed_pools import (
    TEST_SEED_RANGE,
    VALIDATION_SEED_RANGE,
    generate_validation_pool,
)
from orbit_guard.evaluation.shifts import CONDITION_NAMES
from orbit_guard.evaluation.test_seed_pool import (
    generate_test_pool,
    load_test_pool,
    write_test_pool,
)
from tests.evaluation.git_repo_fixture import make_repo


@pytest.fixture(autouse=True)
def _clean_git_repo(tmp_path, monkeypatch):
    """Freeze git checks run against a committed throwaway repo, never the (possibly dirty) real checkout."""
    monkeypatch.setattr(freeze, "GIT_ROOT", make_repo(tmp_path / "repo"))


def test_test_pool_has_ten_unique_seeds_for_each_of_the_five_conditions():
    pool = generate_test_pool()

    assert tuple(pool) == CONDITION_NAMES
    assert all(len(seeds) == 10 for seeds in pool.values())
    flat = [seed for seeds in pool.values() for seed in seeds]
    assert len(flat) == len(set(flat)) == 50
    assert all(TEST_SEED_RANGE[0] <= seed < TEST_SEED_RANGE[1] for seed in flat)


def test_test_pool_is_disjoint_from_the_validation_pool_and_its_range():
    flat = {seed for seeds in generate_test_pool().values() for seed in seeds}

    assert flat.isdisjoint(generate_validation_pool())
    assert all(not VALIDATION_SEED_RANGE[0] <= seed < VALIDATION_SEED_RANGE[1] for seed in flat)


def _weights(tmp_path, content=b"weights-v1"):
    path = tmp_path / "actor_weights.npz"
    path.write_bytes(content)
    (tmp_path / "run_config.json").write_text("{}")  # the freeze refuses a checkpoint without its sidecar
    return {"adaptive_mappo/0": path}


def test_loading_the_test_pool_requires_a_matching_freeze_snapshot(tmp_path):
    pool_file = tmp_path / "test_seeds.json"
    write_test_pool(pool_file)
    freeze_file = tmp_path / "frozen.json"
    weights = _weights(tmp_path)

    with pytest.raises(FrozenConfigError, match="no freeze snapshot"):
        load_test_pool(pool_file, freeze_file, weights)

    write_freeze_snapshot(freeze_file, weights)
    assert load_test_pool(pool_file, freeze_file, weights) == generate_test_pool()


def test_changing_a_checkpoint_after_freezing_blocks_the_test_pool(tmp_path):
    pool_file = tmp_path / "test_seeds.json"
    write_test_pool(pool_file)
    freeze_file = tmp_path / "frozen.json"
    weights = _weights(tmp_path)
    write_freeze_snapshot(freeze_file, weights)

    weights["adaptive_mappo/0"].write_bytes(b"retrained-after-peeking")

    with pytest.raises(FrozenConfigError, match="adaptive_mappo/0"):
        load_test_pool(pool_file, freeze_file, weights)


def test_changing_the_config_after_freezing_blocks_the_test_pool(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("a: 1\n")
    freeze_file = tmp_path / "frozen.json"
    write_freeze_snapshot(freeze_file, {}, config_path=config)

    config.write_text("a: 2\n")

    with pytest.raises(FrozenConfigError, match="config"):
        verify_freeze(freeze_file, {}, config_path=config)


def test_the_snapshot_records_the_config_digest_and_checkpoint_hashes(tmp_path):
    freeze_file = tmp_path / "frozen.json"
    weights = _weights(tmp_path)

    snapshot = write_freeze_snapshot(freeze_file, weights)

    on_disk = json.loads(freeze_file.read_text())
    assert on_disk["config_digest"] == snapshot["config_digest"]
    assert set(on_disk["checkpoints"]) == {"adaptive_mappo/0"}
    assert "frozen_at" in on_disk


def test_a_tampered_test_pool_file_fails_its_checksum(tmp_path):
    pool_file = tmp_path / "test_seeds.json"
    write_test_pool(pool_file)
    record = json.loads(pool_file.read_text())
    record["conditions"]["normal"][0] += 1
    pool_file.write_text(json.dumps(record))
    freeze_file = tmp_path / "frozen.json"
    write_freeze_snapshot(freeze_file, {})

    with pytest.raises(ValueError, match="checksum"):
        load_test_pool(pool_file, freeze_file, {})


def test_the_committed_test_pool_matches_the_frozen_generator(tmp_path):
    from pathlib import Path

    record = json.loads((Path(__file__).resolve().parents[2] / "configs" / "test_seeds.json").read_text())

    assert {name: tuple(seeds) for name, seeds in record["conditions"].items()} == generate_test_pool()
