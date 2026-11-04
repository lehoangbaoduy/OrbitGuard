"""Checkpoint provenance compares the sidecar's `shield` and `num_agents` too (review2, low).

A no-shield run registered as adaptive_mappo (or the reverse), or a run trained at N=2 (the training CLI's default)
evaluated at N=15, used to pass. The recorded shield must equal the model spec's `shield`, the recorded num_agents
the N the run evaluates at; a sidecar that has config_digest but lacks either key is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH
from orbit_guard.evaluation import experiment
from orbit_guard.evaluation.experiment import (
    CheckpointProvenanceError,
    build_tasks,
    check_checkpoint_provenance,
    check_registry_provenance,
)
from orbit_guard.evaluation.replay_recording import EpisodePlan, check_plan_provenance
from tests.evaluation.sidecar_fixtures import sidecar_record


def _checkpoint(tmp_path: Path, name: str = "run", *, drop: str | None = None, **record) -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "actor_weights.npz").write_bytes(b"weights")
    sidecar = {**sidecar_record(None), **record}
    if drop is not None:
        del sidecar[drop]
    (run / "run_config.json").write_text(json.dumps(sidecar))
    return run / "actor_weights.npz"


def _check(weights: Path, model: str | None, n_agents: int | None = None) -> None:
    check_checkpoint_provenance(weights, config_path=DEFAULT_CONFIG_PATH, model=model, n_agents=n_agents)


@pytest.mark.parametrize(("model", "shield"), [("adaptive_mappo", True), ("adaptive_mappo_no_shield", False)])
def test_a_matching_shield_and_constellation_size_are_accepted(tmp_path, model, shield):
    _check(_checkpoint(tmp_path, shield=shield, num_agents=15), model, n_agents=15)


@pytest.mark.parametrize(("model", "trained_shield"), [("adaptive_mappo", False), ("adaptive_mappo_no_shield", True)])
def test_a_checkpoint_registered_under_the_other_shield_setting_is_refused(tmp_path, model, trained_shield):
    with pytest.raises(CheckpointProvenanceError, match=rf"shield.*{model}"):
        _check(_checkpoint(tmp_path, shield=trained_shield), model)


def test_a_non_boolean_shield_record_is_refused(tmp_path):
    with pytest.raises(CheckpointProvenanceError, match="shield"):
        _check(_checkpoint(tmp_path, shield=1), "adaptive_mappo")  # 1 == True in Python; the record must be a bool


def test_a_checkpoint_trained_at_another_constellation_size_is_refused(tmp_path):
    weights = _checkpoint(tmp_path, num_agents=2)  # the training CLI's default --agents

    with pytest.raises(CheckpointProvenanceError, match=r"2 agents.*15"):
        _check(weights, "adaptive_mappo", n_agents=15)


@pytest.mark.parametrize("bad", [True, 15.0, "15"])
def test_a_malformed_num_agents_record_is_refused(tmp_path, bad):
    with pytest.raises(CheckpointProvenanceError, match="num_agents"):
        _check(_checkpoint(tmp_path, num_agents=bad), "adaptive_mappo", n_agents=15)


@pytest.mark.parametrize("key", ["shield", "num_agents"])
def test_a_sidecar_without_the_key_is_refused_even_without_a_model_or_n(tmp_path, key):
    with pytest.raises(CheckpointProvenanceError, match=key):
        _check(_checkpoint(tmp_path, drop=key), None)


def test_without_a_model_or_n_only_presence_is_checked(tmp_path):
    _check(_checkpoint(tmp_path, shield=False, num_agents=2), None)


def test_the_registry_check_threads_the_constellation_size(tmp_path):
    registry = {"adaptive_mappo": {0: str(_checkpoint(tmp_path, num_agents=2))}}

    check_registry_provenance(registry, config_path=DEFAULT_CONFIG_PATH, n_agents=2)
    with pytest.raises(CheckpointProvenanceError, match="agents"):
        check_registry_provenance(registry, config_path=DEFAULT_CONFIG_PATH, n_agents=15)


def test_build_tasks_compares_the_n_it_will_run_at(tmp_path):
    registry = {"adaptive_mappo": {0: str(_checkpoint(tmp_path, num_agents=15))}}

    with pytest.raises(CheckpointProvenanceError, match="agents"):
        build_tasks(("adaptive_mappo",), {"normal": (1,)}, registry, n_agents=3, experiment_id="e")


def test_the_run_command_refuses_a_checkpoint_trained_at_another_n(tmp_path, monkeypatch):
    registry = tmp_path / "reg.json"
    registry.write_text(json.dumps({"adaptive_mappo": {"0": str(_checkpoint(tmp_path, num_agents=15))}}))
    monkeypatch.setattr(experiment, "run_tasks", lambda *a, **k: pytest.fail("must refuse before running"))

    with pytest.raises(CheckpointProvenanceError, match="agents"):
        experiment.main([
            "run", "--split", "validation", "--max-seeds", "1", "--models", "adaptive_mappo", "--n-agents", "3",
            "--registry", str(registry), "--out", str(tmp_path / "e.jsonl"),
        ])


def test_the_registry_command_checks_n_agents(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for variant, margin, shield in (("adaptive", "adaptive", True), ("fixed", "fixed", True),
                                    ("noshield", "adaptive", False)):
        for seed in (0, 1, 2):
            sidecar = {**sidecar_record(None, margin_kind=margin, shield=shield, num_agents=2, seed=seed)}
            _checkpoint(runs, f"n15_{variant}_s{seed}", **sidecar)
    out = tmp_path / "reg.json"

    with pytest.raises(CheckpointProvenanceError, match="agents"):  # --n-agents defaults to 15
        experiment.main(["registry", "--runs-dir", str(runs), "--out", str(out)])
    experiment.main(["registry", "--runs-dir", str(runs), "--out", str(out), "--n-agents", "2"])

    assert out.exists()


def test_the_replay_recorder_checks_shield_and_n(tmp_path):
    no_shield = _checkpoint(tmp_path, shield=False, num_agents=15)

    check_plan_provenance([EpisodePlan("normal", 0, "trained", False, "adaptive_mappo_no_shield", no_shield)],
                          num_agents=15)
    with pytest.raises(CheckpointProvenanceError, match="shield"):
        check_plan_provenance([EpisodePlan("normal", 0, "trained", True, "adaptive_mappo", no_shield)])
    with pytest.raises(CheckpointProvenanceError, match="agents"):
        check_plan_provenance([EpisodePlan("normal", 0, "trained", False, "adaptive_mappo_no_shield", no_shield)],
                              num_agents=2)
