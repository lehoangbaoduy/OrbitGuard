"""Tests for orbit_guard.evaluation.runner: episodes -> §40 records, parallel and resumable."""

from __future__ import annotations

import json

import numpy as np
import pytest

from orbit_guard.evaluation.runner import EpisodeTask, run_episode_task, run_tasks
from orbit_guard.rl.actor_policy import ActorWeights, save_actor_weights

_FAST = {"episode_horizon_s": 240.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 300.0}
_TIMING_KEYS = {"start_time", "wall_time_s"}


def _task(**overrides) -> EpisodeTask:
    fields = {
        "model": "rule_based", "condition": "normal", "scenario_seed": 3, "n_agents": 4,
        "env_kwargs": _FAST, "experiment_id": "unit", "git_commit": "abc123",
    }
    fields.update(overrides)
    return EpisodeTask(**fields)


def _random_weights(tmp_path) -> str:
    rng = np.random.default_rng(0)
    sizes = [(128, 54), (128, 128), (6, 128)]
    layers = tuple((rng.normal(0, 0.1, size=s).astype(np.float32), np.zeros(s[0], np.float32)) for s in sizes)
    path = tmp_path / "actor.npz"
    save_actor_weights(ActorWeights(layers), path)
    return str(path)


def _stable(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in _TIMING_KEYS}


def test_record_carries_the_per_episode_metadata_the_logging_spec_requires():
    record = run_episode_task(_task())

    for key in (
        "experiment_id", "scenario_id", "scenario_seed", "training_seed", "model_name", "git_commit",
        "config_hash", "start_time", "n_satellites", "shift_type",
    ):
        assert key in record, key
    assert record["model_name"] == "rule_based"
    assert record["shift_type"] == "normal"
    assert record["n_satellites"] == 4
    assert record["scenario_seed"] == 3
    assert len(record["config_hash"]) == 64


def test_record_carries_the_episode_summary_fields():
    record = run_episode_task(_task())

    for key in (
        "collision_count", "unsafe_conjunction_count", "total_delta_v_mps", "maneuver_count",
        "mean_mission_deviation", "shield_intervention_count", "mean_residual_risk", "return_per_agent",
        "failure_labels", "steps", "status",
    ):
        assert key in record, key
    assert record["status"] == "ok"
    assert record["steps"] == 4


def test_the_same_task_reproduces_the_same_record_acceptance_f(tmp_path):
    shielded = _task(model="adaptive_mappo", actor_weights_path=_random_weights(tmp_path), n_agents=3)

    first = _stable(run_episode_task(shielded))
    second = _stable(run_episode_task(shielded))

    assert first == second


def test_different_scenario_seeds_give_different_episodes():
    a = run_episode_task(_task(scenario_seed=3))
    b = run_episode_task(_task(scenario_seed=4))

    assert a["scenario_id"] != b["scenario_id"]


def test_shift_condition_is_applied_and_recorded():
    record = run_episode_task(_task(condition="capability_reduction", n_agents=10))

    assert record["shift_type"] == "capability_reduction"
    assert record["status"] == "ok"


def test_a_failing_policy_is_recorded_as_a_failed_episode_not_a_crash(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, w0=np.zeros((2, 2), np.float32), b0=np.zeros(2, np.float32))
    task = _task(model="adaptive_mappo_no_shield", actor_weights_path=str(bad))

    record = run_episode_task(task)

    assert record["status"] == "failed"
    assert "numerical_simulation_failure" in record["failure_labels"]
    assert record["error"]


def test_learned_models_cannot_be_scheduled_without_weights():
    with pytest.raises(ValueError, match="actor weights"):
        run_episode_task(_task(model="adaptive_mappo"))


def test_run_tasks_writes_one_jsonl_line_per_task_in_parallel(tmp_path):
    out = tmp_path / "episodes.jsonl"
    tasks = [_task(scenario_seed=seed) for seed in (1, 2, 3)]

    records = run_tasks(tasks, out, workers=2)

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == len(records) == 3
    assert {line["scenario_seed"] for line in lines} == {1, 2, 3}


def test_run_tasks_resumes_by_skipping_completed_episodes(tmp_path):
    out = tmp_path / "episodes.jsonl"
    run_tasks([_task(scenario_seed=1)], out, workers=1)

    records = run_tasks([_task(scenario_seed=1), _task(scenario_seed=2)], out, workers=1)

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert [record["scenario_seed"] for record in records] == [2]  # only the new one ran
