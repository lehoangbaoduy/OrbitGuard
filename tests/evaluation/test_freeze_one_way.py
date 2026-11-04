"""The one-way freeze (PRD §33: the final test is "strictly held out until every model/config choice is frozen").

A snapshot can be written once, only from a committed tree (orbit_guard/, configs/, scripts/; docs/ edits do
not block), and records the config digest with its overlay, both seed-pool files, every checkpoint's weights
and run_config.json, and the git commit. verify_freeze re-checks every one of these and that no code or
config changed since the frozen commit. All git calls go to a temporary repository, never the real checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.evaluation import experiment, freeze
from orbit_guard.evaluation.freeze import (
    FrozenConfigError,
    GitTree,
    verify_freeze,
    write_freeze_snapshot,
)
from orbit_guard.evaluation.registry_checks import IncompleteRegistryError
from orbit_guard.evaluation.test_seed_pool import load_test_pool
from tests.evaluation.freeze_world import (
    build_world,
    training_sidecar,
    write_checkpoint,
)
from tests.evaluation.git_repo_fixture import commit_all, git, make_repo
from tests.evaluation.sidecar_fixtures import sidecar_record


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = make_repo(tmp_path / "repo")
    monkeypatch.setattr(freeze, "GIT_ROOT", root)
    return root


@pytest.fixture
def world(tmp_path, repo) -> dict:
    """Config, overlay, both pools and one checkpoint with a run_config.json sidecar (trained at the repo's HEAD
    from a clean tree), outside the repo."""
    return build_world(tmp_path / "files", repo)


def _freeze(world: dict, **overrides) -> dict:
    kwargs = {
        "config_path": world["config"], "overlay_path": world["overlay"], "test_pool_path": world["test_pool"],
        "validation_pool_path": world["validation_pool"],
    }
    kwargs.update(overrides)
    return write_freeze_snapshot(world["snapshot"], world["checkpoints"], **kwargs)


def _verify(world: dict, **overrides) -> dict:
    kwargs = {
        "config_path": world["config"], "overlay_path": world["overlay"], "test_pool_path": world["test_pool"],
        "validation_pool_path": world["validation_pool"],
    }
    kwargs.update(overrides)
    return verify_freeze(world["snapshot"], world["checkpoints"], **kwargs)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- writing the snapshot -----------------------------------------------------------------------------------


def test_the_snapshot_records_digest_pools_checkpoints_and_commit(world, repo):
    snapshot = _freeze(world)

    on_disk = json.loads(world["snapshot"].read_text())
    assert on_disk == snapshot
    assert snapshot["config_digest"] == config_digest(world["config"], world["overlay"])
    assert snapshot["overlay_path"] == str(world["overlay"])
    assert snapshot["pools"]["test"]["sha256"] == _sha256(world["test_pool"])
    assert snapshot["pools"]["validation"]["sha256"] == _sha256(world["validation_pool"])
    entry = snapshot["checkpoints"]["adaptive_mappo/0"]
    assert entry["actor_weights_sha256"] == _sha256(world["run"] / "actor_weights.npz")
    assert entry["run_config_sha256"] == _sha256(world["run"] / "run_config.json")
    assert snapshot["git_commit"] == git(repo, "rev-parse", "HEAD")


def test_a_checkpoint_without_a_sidecar_cannot_be_frozen(world):
    (world["run"] / "run_config.json").unlink()

    with pytest.raises(FrozenConfigError, match=r"run_config\.json.*adaptive_mappo/0"):
        _freeze(world)

    assert not world["snapshot"].exists()


def test_one_sidecar_less_checkpoint_among_several_blocks_the_whole_freeze(world):
    both = _second_seed(world)
    (both["adaptive_mappo/1"].parent / "run_config.json").unlink()

    with pytest.raises(FrozenConfigError, match="adaptive_mappo/1"):
        write_freeze_snapshot(
            world["snapshot"], both, world["config"], world["overlay"],
            test_pool_path=world["test_pool"], validation_pool_path=world["validation_pool"],
        )

    assert not world["snapshot"].exists()


def test_an_existing_snapshot_is_never_overwritten(world):
    _freeze(world)
    before = world["snapshot"].read_text()

    with pytest.raises(FrozenConfigError, match="already exists"):
        _freeze(world)

    assert world["snapshot"].read_text() == before


def _names(world: dict) -> set[str]:
    return {path.name for path in world["snapshot"].parent.iterdir()}


def test_a_snapshot_that_cannot_be_serialized_leaves_no_file_behind(world, monkeypatch):
    before = _names(world)
    monkeypatch.setattr(freeze, "_training_commit", lambda checkpoints, git: object())  # json cannot encode it

    with pytest.raises(TypeError):
        _freeze(world)

    assert _names(world) == before


def test_a_disk_failure_while_writing_leaves_no_torn_snapshot(world, monkeypatch):
    before = _names(world)

    def disk_full(fd):
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "fsync", disk_full)

    with pytest.raises(OSError, match="No space"):
        _freeze(world)

    assert _names(world) == before


def test_a_snapshot_created_during_the_freeze_is_never_replaced(world, monkeypatch):
    real_require_sidecars = freeze._require_sidecars
    before_race = _names(world)

    def concurrent_freeze_wins(checkpoints):
        world["snapshot"].write_text("written by a concurrent freeze\n")
        real_require_sidecars(checkpoints)

    monkeypatch.setattr(freeze, "_require_sidecars", concurrent_freeze_wins)

    with pytest.raises(FrozenConfigError, match="already exists"):
        _freeze(world)

    assert world["snapshot"].read_text() == "written by a concurrent freeze\n"
    assert _names(world) == {world["snapshot"].name, *before_race}


@pytest.mark.parametrize(
    "relative", ["orbit_guard/module.py", "configs/default.yaml", "scripts/run.sh", "data/tle.csv"]
)
def test_an_uncommitted_change_to_code_configs_scripts_or_data_blocks_the_freeze(world, repo, relative):
    (repo / relative).write_text("changed\n")

    with pytest.raises(FrozenConfigError, match=relative):
        _freeze(world)

    assert not world["snapshot"].exists()


def test_an_untracked_config_file_blocks_the_freeze(world, repo):
    (repo / "configs" / "tuned.yaml").write_text("reward: {}\n")

    with pytest.raises(FrozenConfigError, match="configs/tuned.yaml"):
        _freeze(world)


def test_a_docs_edit_does_not_block_the_freeze(world, repo):
    (repo / "docs" / "notes.md").write_text("edited while the sweep runs\n")

    _freeze(world)

    assert world["snapshot"].exists()


def test_freezing_outside_a_git_repository_is_refused(world, tmp_path, monkeypatch):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    monkeypatch.setattr(freeze, "GIT_ROOT", not_a_repo)

    with pytest.raises(FrozenConfigError, match="git"):
        _freeze(world)


def test_git_runs_in_the_injected_root_not_the_working_directory(world, repo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a repository

    snapshot = _freeze(world, git=GitTree(repo))

    assert snapshot["git_commit"] == git(repo, "rev-parse", "HEAD")


# --- verifying it -------------------------------------------------------------------------------------------


def test_an_unchanged_world_verifies(world):
    frozen = _freeze(world)

    assert _verify(world) == frozen


def test_verify_defaults_to_the_files_the_snapshot_recorded(world):
    _freeze(world)

    verify_freeze(world["snapshot"], world["checkpoints"])


def test_a_different_overlay_fails_verification(world, tmp_path):
    _freeze(world)
    other = tmp_path / "other.yaml"
    other.write_text("reward:\n  w3: 2.0\n")

    with pytest.raises(FrozenConfigError, match="config"):
        _verify(world, overlay_path=other)


def test_evaluating_without_the_frozen_overlay_fails_verification(world):
    _freeze(world)

    with pytest.raises(FrozenConfigError, match="config"):
        _verify(world, overlay_path=None)


def test_an_edited_overlay_fails_verification(world):
    _freeze(world)
    world["overlay"].write_text("reward:\n  w3: 9.0\n")

    with pytest.raises(FrozenConfigError, match="config"):
        verify_freeze(world["snapshot"], world["checkpoints"])


@pytest.mark.parametrize("pool", ["test_pool", "validation_pool"])
def test_an_edited_seed_pool_fails_verification(world, pool):
    _freeze(world)
    world[pool].write_text(world[pool].read_text() + " ")

    with pytest.raises(FrozenConfigError, match=pool.replace("_pool", "")):
        _verify(world)


def test_changed_weights_fail_verification(world):
    _freeze(world)
    (world["run"] / "actor_weights.npz").write_bytes(b"retrained-after-peeking")

    with pytest.raises(FrozenConfigError, match="adaptive_mappo/0"):
        _verify(world)


def test_a_changed_run_config_fails_verification(world):
    _freeze(world)
    (world["run"] / "run_config.json").write_text("{}")

    with pytest.raises(FrozenConfigError, match="run_config.json"):
        _verify(world)


def test_a_run_config_that_disappears_after_the_freeze_fails_verification(world):
    _freeze(world)
    (world["run"] / "run_config.json").unlink()

    with pytest.raises(FrozenConfigError, match="run_config.json"):
        _verify(world)


def test_a_checkpoint_that_was_never_frozen_fails_verification(world, tmp_path):
    _freeze(world)
    extra = tmp_path / "extra.npz"
    extra.write_bytes(b"w")

    with pytest.raises(FrozenConfigError, match="never frozen"):
        verify_freeze(world["snapshot"], {**world["checkpoints"], "fixed_margin_mappo/0": extra})


def _second_seed(world: dict) -> dict:
    """A second training seed of the same model, with its own sidecar; returns the two-checkpoint mapping."""
    run = world["run"].parent / "n15_adaptive_s1"
    run.mkdir()
    (run / "actor_weights.npz").write_bytes(b"weights-seed-1")
    (run / "run_config.json").write_text((world["run"] / "run_config.json").read_text())
    return {**world["checkpoints"], "adaptive_mappo/1": run / "actor_weights.npz"}


def test_dropping_a_frozen_training_seed_fails_verification(world):
    both = _second_seed(world)
    write_freeze_snapshot(
        world["snapshot"], both, world["config"], world["overlay"],
        test_pool_path=world["test_pool"], validation_pool_path=world["validation_pool"],
    )

    with pytest.raises(FrozenConfigError, match=r"adaptive_mappo/1") as caught:
        _verify(world)  # only adaptive_mappo/0: seed 1 dropped after seeing results

    assert "not passed" in str(caught.value) or "missing" in str(caught.value)


def test_verifying_no_checkpoints_against_a_freeze_that_has_some_fails(world):
    _freeze(world)

    with pytest.raises(FrozenConfigError, match=r"adaptive_mappo/0"):
        verify_freeze(
            world["snapshot"], {}, world["config"], world["overlay"],
            test_pool_path=world["test_pool"], validation_pool_path=world["validation_pool"],
        )


def test_an_added_and_a_dropped_checkpoint_are_both_named(world, tmp_path):
    both = _second_seed(world)
    write_freeze_snapshot(
        world["snapshot"], both, world["config"], world["overlay"],
        test_pool_path=world["test_pool"], validation_pool_path=world["validation_pool"],
    )
    extra = tmp_path / "extra.npz"
    extra.write_bytes(b"w")

    with pytest.raises(FrozenConfigError) as caught:
        verify_freeze(
            world["snapshot"], {"adaptive_mappo/0": both["adaptive_mappo/0"], "adaptive_mappo/2": extra},
            world["config"], world["overlay"],
            test_pool_path=world["test_pool"], validation_pool_path=world["validation_pool"],
        )

    assert "adaptive_mappo/1" in str(caught.value) and "adaptive_mappo/2" in str(caught.value)


def test_the_full_frozen_set_still_verifies(world):
    both = _second_seed(world)
    write_freeze_snapshot(
        world["snapshot"], both, world["config"], world["overlay"],
        test_pool_path=world["test_pool"], validation_pool_path=world["validation_pool"],
    )

    verify_freeze(
        world["snapshot"], both, world["config"], world["overlay"],
        test_pool_path=world["test_pool"], validation_pool_path=world["validation_pool"],
    )


def test_code_committed_after_the_freeze_fails_verification(world, repo):
    _freeze(world)
    (repo / "orbit_guard" / "module.py").write_text("VALUE = 2\n")
    commit_all(repo, "tweak after peeking")

    with pytest.raises(FrozenConfigError, match="since the freeze"):
        _verify(world)


def test_an_uncommitted_change_after_the_freeze_fails_verification(world, repo):
    _freeze(world)
    (repo / "scripts" / "run.sh").write_text("echo changed\n")

    with pytest.raises(FrozenConfigError, match="scripts/run.sh"):
        _verify(world)


def test_tle_data_committed_after_the_freeze_fails_verification(world, repo):
    _freeze(world)
    (repo / "data" / "tle.csv").write_text("norad,a,e\n1,6900,0.001\n")
    commit_all(repo, "new TLE snapshot after peeking")

    with pytest.raises(FrozenConfigError, match="since the freeze"):
        _verify(world)


def test_docs_committed_after_the_freeze_do_not_fail_verification(world, repo):
    _freeze(world)
    (repo / "docs" / "notes.md").write_text("results written up\n")
    commit_all(repo, "docs")

    _verify(world)


def test_an_unknown_frozen_commit_is_reported_as_such(world):
    _freeze(world)
    snapshot = json.loads(world["snapshot"].read_text())
    snapshot["git_commit"] = "0" * 40
    world["snapshot"].write_text(json.dumps(snapshot))

    with pytest.raises(FrozenConfigError, match="unknown"):
        _verify(world)


def test_an_old_format_snapshot_is_refused_with_a_clear_error(world):
    world["snapshot"].write_text(json.dumps({"config_hash": "x", "checkpoints": {}, "git_commit": "y"}))

    with pytest.raises(FrozenConfigError, match="format"):
        _verify(world)


def test_the_test_pool_loader_still_requires_a_valid_freeze(world):
    _freeze(world)
    world["overlay"].write_text("reward:\n  w3: 9.0\n")

    with pytest.raises(FrozenConfigError):
        load_test_pool(world["test_pool"], world["snapshot"], world["checkpoints"])


# --- the experiment CLI ---------------------------------------------------------------------------------------


def _registry(world: dict) -> Path:
    path = world["snapshot"].parent / "registry.json"
    path.write_text(json.dumps({"adaptive_mappo": {"0": str(world["run"] / "actor_weights.npz")}}))
    return path


def _learned_registry(world: dict) -> Path:
    """Seed 0 of each of the three learned models (seeds 1 and 2 missing), each with its training sidecar.

    The command's provenance check rebuilds the evaluation margin and reward weights, so it needs a real config
    and the sidecar a training run under it would write (tests/evaluation/sidecar_fixtures.py).
    """
    world["config"].write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    registry = {}
    for model, margin, shield in (
        ("adaptive_mappo", "adaptive", True), ("fixed_margin_mappo", "fixed", True),
        ("adaptive_mappo_no_shield", "adaptive", False),
    ):
        record = sidecar_record(world["overlay"], config_path=world["config"], margin_kind=margin, shield=shield)
        sidecar = {**record, **training_sidecar(world["repo"])}
        registry[model] = {"0": str(write_checkpoint(world["run"].parent / f"{model}_s0", sidecar, model.encode()))}
    path = world["snapshot"].parent / "registry.json"
    path.write_text(json.dumps(registry))
    return path


def _freeze_command(world: dict, *extra: str) -> None:
    experiment.main(
        [
            "freeze", "--registry", str(_learned_registry(world)), "--out", str(world["snapshot"]),
            "--config", str(world["config"]), "--overlay", str(world["overlay"]),
            "--test-pool", str(world["test_pool"]), "--validation-pool", str(world["validation_pool"]), *extra,
        ]
    )


def test_the_freeze_command_records_the_overlay_and_an_explicit_partial_freeze(world):
    _freeze_command(world, "--allow-partial")  # the registry holds seed 0 of each learned model only

    snapshot = json.loads(world["snapshot"].read_text())
    assert snapshot["config_digest"] == config_digest(world["config"], world["overlay"])
    assert snapshot["allow_partial"] is True


def test_the_freeze_command_refuses_a_registry_missing_training_seeds(world):
    with pytest.raises(IncompleteRegistryError, match="adaptive_mappo"):
        _freeze_command(world)

    assert not world["snapshot"].exists()


def test_a_freeze_records_that_it_is_not_partial_by_default(world):
    assert _freeze(world)["allow_partial"] is False


def test_the_test_split_refuses_an_overlay_other_than_the_frozen_one(world, monkeypatch):
    _freeze(world)
    # After the freeze (which requires the sidecar): isolates the freeze check from the provenance check.
    (world["run"] / "run_config.json").unlink()
    monkeypatch.setattr(experiment, "run_tasks", lambda *a, **k: pytest.fail("must refuse before running"))

    with pytest.raises(FrozenConfigError, match="config"):
        experiment.main(
            [
                "run", "--split", "test", "--freeze-snapshot", str(world["snapshot"]), "--models", "adaptive_mappo",
                "--registry", str(_registry(world)), "--out", str(world["snapshot"].parent / "e.jsonl"),
                "--config", str(world["config"]), "--test-pool", str(world["test_pool"]),
            ]
        )


@pytest.mark.parametrize("errno_name", ["EPERM", "ENOTSUP"])
def test_a_filesystem_without_hard_links_fails_with_a_clear_message_and_no_leftover(world, monkeypatch, errno_name):
    import errno

    before = _names(world)

    def no_links(source, target):
        raise OSError(getattr(errno, errno_name), "Operation not permitted")

    monkeypatch.setattr(freeze.os, "link", no_links)

    with pytest.raises(FrozenConfigError, match="hard link"):
        _freeze(world)

    assert _names(world) == before


def test_the_snapshot_directory_is_synced_after_the_link(world, monkeypatch):
    synced = []
    monkeypatch.setattr(freeze, "fsync_directory", lambda directory: synced.append(Path(directory)))

    _freeze(world)

    assert synced == [world["snapshot"].parent]
