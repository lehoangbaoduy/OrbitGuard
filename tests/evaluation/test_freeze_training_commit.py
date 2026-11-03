"""The freeze checks the code the checkpoints were TRAINED at, not only the commit it runs at (review2, high).

rl/training.py records in run_config.json the commit it started from (`git_commit`) and the porcelain status of
orbit_guard/, configs/ and data/ (`git_dirty`). write_freeze_snapshot refuses a checkpoint trained from a dirty
tree, checkpoints trained at different commits, and any change to those paths between the training commit and the
freeze; it records the training commit, and verify_freeze re-checks it. Git runs in a throwaway repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_guard.evaluation import freeze
from orbit_guard.evaluation.freeze import (
    FrozenConfigError,
    verify_freeze,
    write_freeze_snapshot,
)
from tests.evaluation.freeze_world import (
    build_world,
    training_sidecar,
    write_checkpoint,
)
from tests.evaluation.git_repo_fixture import commit_all, git, make_repo


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = make_repo(tmp_path / "repo")
    monkeypatch.setattr(freeze, "GIT_ROOT", root)
    return root


@pytest.fixture
def world(tmp_path, repo) -> dict:
    return build_world(tmp_path / "files", repo)


def _paths(world: dict) -> dict:
    return {
        "config_path": world["config"], "overlay_path": world["overlay"],
        "test_pool_path": world["test_pool"], "validation_pool_path": world["validation_pool"],
    }


def _freeze(world: dict, checkpoints: dict | None = None) -> dict:
    return write_freeze_snapshot(world["snapshot"], checkpoints or world["checkpoints"], **_paths(world))


def _verify(world: dict, checkpoints: dict | None = None) -> dict:
    return verify_freeze(world["snapshot"], checkpoints or world["checkpoints"], **_paths(world))


def _rewrite_sidecar(world: dict, **fields) -> None:
    record = json.loads((world["run"] / "run_config.json").read_text())
    (world["run"] / "run_config.json").write_text(json.dumps({**record, **fields}))


def _second_seed(world: dict, **fields) -> dict:
    sidecar = training_sidecar(world["repo"], world["config"], world["overlay"], **fields)
    weights = write_checkpoint(world["run"].parent / "n15_adaptive_s1", sidecar, b"weights-seed-1")
    return {**world["checkpoints"], "adaptive_mappo/1": weights}


def test_the_snapshot_records_the_commit_the_checkpoints_were_trained_at(world, repo):
    trained_at = git(repo, "rev-parse", "HEAD")
    (repo / "docs" / "notes.md").write_text("written while the queue ran\n")
    frozen_at = commit_all(repo, "docs after training")

    snapshot = _freeze(world)

    assert snapshot["training_commit"] == trained_at
    assert snapshot["git_commit"] == frozen_at


def test_a_checkpoint_trained_from_a_dirty_tree_cannot_be_frozen(world):
    _rewrite_sidecar(world, git_dirty=[" M orbit_guard/module.py"])

    with pytest.raises(FrozenConfigError, match=r"adaptive_mappo/0.*uncommitted.*orbit_guard/module\.py"):
        _freeze(world)

    assert not world["snapshot"].exists()


@pytest.mark.parametrize(
    "fields",
    [{"git_dirty": None}, {"git_commit": None}, {"git_dirty": "clean"}],
    ids=["dirty-unknown", "commit-unknown", "dirty-not-a-list"],
)
def test_a_sidecar_whose_git_state_is_unknown_cannot_be_frozen(world, fields):
    _rewrite_sidecar(world, **fields)

    with pytest.raises(FrozenConfigError, match="adaptive_mappo/0"):
        _freeze(world)


@pytest.mark.parametrize("key", ["git_commit", "git_dirty"])
def test_a_sidecar_that_predates_the_git_record_cannot_be_frozen(world, key):
    record = json.loads((world["run"] / "run_config.json").read_text())
    del record[key]
    (world["run"] / "run_config.json").write_text(json.dumps(record))

    with pytest.raises(FrozenConfigError, match=rf"adaptive_mappo/0.*{key}"):
        _freeze(world)


def test_checkpoints_trained_at_different_commits_cannot_be_frozen(world, repo):
    # Even a docs-only commit between two runs of the queue: the rule is literal (every run at ONE commit).
    (repo / "docs" / "notes.md").write_text("docs commit between two runs\n")
    commit_all(repo, "docs")
    both = _second_seed(world)

    with pytest.raises(FrozenConfigError, match=r"different commits.*adaptive_mappo/0.*adaptive_mappo/1"):
        _freeze(world, both)

    assert not world["snapshot"].exists()


@pytest.mark.parametrize("relative", ["orbit_guard/module.py", "configs/default.yaml", "data/tle.csv"])
def test_code_committed_between_training_and_the_freeze_blocks_the_freeze(world, repo, relative):
    trained_at = git(repo, "rev-parse", "HEAD")
    (repo / relative).write_text("changed after training\n")
    commit_all(repo, "fix landed after the queue")

    with pytest.raises(FrozenConfigError, match=rf"since training.*{trained_at}"):
        _freeze(world)

    assert not world["snapshot"].exists()


def test_a_script_committed_after_training_does_not_block_the_freeze(world, repo):
    (repo / "scripts" / "run.sh").write_text("echo fixed\n")  # scripts never reach the trained weights
    commit_all(repo, "script fix after training")

    _freeze(world)

    assert world["snapshot"].exists()


def test_a_training_commit_unknown_to_the_repository_blocks_the_freeze(world):
    _rewrite_sidecar(world, git_commit="0" * 40)

    with pytest.raises(FrozenConfigError, match=r"training commit .*0{40}"):
        _freeze(world)


def test_a_freeze_with_no_checkpoints_records_no_training_commit(world):
    snapshot = write_freeze_snapshot(world["snapshot"], {}, **_paths(world))

    assert snapshot["training_commit"] is None


def test_verify_accepts_an_unchanged_freeze(world):
    frozen = _freeze(world)

    assert _verify(world) == frozen


def test_verify_refuses_a_snapshot_whose_training_commit_was_edited(world, repo):
    _freeze(world)
    snapshot = json.loads(world["snapshot"].read_text())
    snapshot["training_commit"] = "f" * 40
    world["snapshot"].write_text(json.dumps(snapshot))

    with pytest.raises(FrozenConfigError, match="training commit"):
        _verify(world)


def test_verify_refuses_a_format_2_snapshot_that_never_checked_the_training_commit(world):
    _freeze(world)
    snapshot = json.loads(world["snapshot"].read_text())
    snapshot["format"] = 2
    del snapshot["training_commit"]
    world["snapshot"].write_text(json.dumps(snapshot))

    with pytest.raises(FrozenConfigError, match="format"):
        _verify(world)
