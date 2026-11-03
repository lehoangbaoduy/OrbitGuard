"""Tests for the runner's PRD §40 per-step logging (FR13) and the §40 per-episode summary names."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orbit_guard.evaluation import runner
from orbit_guard.evaluation.runner import EpisodeTask, run_episode_task, run_tasks
from orbit_guard.evaluation.step_log import (
    StepLogIntegrityError,
    read_episode_step_log,
    read_step_log,
)
from orbit_guard.rl.actor_policy import ActorWeights, save_actor_weights
from orbit_guard.rl.step_records import STEP_RECORD_KEYS

_FAST = {"episode_horizon_s": 240.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 300.0}
_N = 3
_STEPS = 4  # 240 s horizon / 60 s control step


def _task(**overrides) -> EpisodeTask:
    fields = {
        "model": "rule_based", "condition": "normal", "scenario_seed": 3, "n_agents": _N,
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


def _steps_dir(tmp_path) -> str:
    return str(tmp_path / "steps")


def test_step_logging_is_off_by_default():
    task = _task()

    record = run_episode_task(task)

    assert task.step_log_dir is None
    assert record["step_log"] is None
    assert record["step_log_sha256"] is None


def test_the_step_log_directory_must_be_a_plain_string_for_spawn_workers(tmp_path):
    with pytest.raises(TypeError, match="step_log_dir"):
        _task(step_log_dir=tmp_path / "steps")


def test_an_empty_step_log_directory_is_refused_rather_than_meaning_the_cwd():
    with pytest.raises(ValueError, match="step_log_dir"):
        _task(step_log_dir="")


def test_the_record_carries_a_relative_path_and_the_sha256_of_the_step_log(tmp_path):
    record = run_episode_task(_task(step_log_dir=_steps_dir(tmp_path)))

    assert record["status"] == "ok"
    assert record["step_log"].startswith("steps/")
    assert not Path(record["step_log"]).is_absolute()
    lines = read_step_log(tmp_path / record["step_log"], record["step_log_sha256"])
    assert lines == read_episode_step_log(tmp_path, record)


@pytest.mark.parametrize("model", ["rule_based", "adaptive_mappo"])
def test_every_logged_line_has_exactly_the_section_40_step_keys_one_per_active_satellite_and_step(tmp_path, model):
    weights = _random_weights(tmp_path) if model == "adaptive_mappo" else None
    record = run_episode_task(_task(model=model, actor_weights_path=weights, step_log_dir=_steps_dir(tmp_path)))

    lines = read_episode_step_log(tmp_path, record)

    assert record["status"] == "ok"
    assert record["collision_count"] == 0  # every satellite stays active, so N_active = N on every step
    assert record["steps"] == _STEPS
    assert len(lines) == _N * _STEPS
    assert all(set(line) == set(STEP_RECORD_KEYS) for line in lines)
    assert [line["satellite_id"] for line in lines[:_N]] == [f"sat_{i}" for i in range(_N)]
    assert sorted({line["timestamp"] for line in lines}) == [60.0 * k for k in range(_STEPS)]


def test_a_shielded_episode_reproduces_the_same_step_log_hash(tmp_path):
    weights = _random_weights(tmp_path)
    task = _task(model="adaptive_mappo", actor_weights_path=weights, step_log_dir=_steps_dir(tmp_path))

    first = run_episode_task(task)
    second = run_episode_task(task)

    assert first["step_log"] == second["step_log"]
    assert first["step_log_sha256"] == second["step_log_sha256"]


def test_a_tampered_step_log_is_caught_through_the_record(tmp_path):
    record = run_episode_task(_task(step_log_dir=_steps_dir(tmp_path)))

    with pytest.raises(StepLogIntegrityError):
        read_episode_step_log(tmp_path, {**record, "step_log_sha256": "0" * 64})


def test_a_failed_episode_logs_the_steps_that_completed_and_stays_failed(tmp_path, monkeypatch):
    real_build_policy = runner.build_policy
    completed_steps = 2

    def failing_build_policy(*args, **kwargs):
        policy = real_build_policy(*args, **kwargs)
        calls = {"n": 0}

        def wrapped(agent, observation):
            calls["n"] += 1
            if calls["n"] > completed_steps * _N:
                raise FloatingPointError("diverged")
            return policy(agent, observation)

        return wrapped

    monkeypatch.setattr(runner, "build_policy", failing_build_policy)

    record = run_episode_task(_task(step_log_dir=_steps_dir(tmp_path)))

    assert record["status"] == "failed"
    assert "numerical_simulation_failure" in record["failure_labels"]
    assert len(read_episode_step_log(tmp_path, record)) == completed_steps * _N


def test_parallel_spawn_workers_write_one_step_log_per_episode(tmp_path):
    out = tmp_path / "episodes.jsonl"
    tasks = [_task(scenario_seed=seed, step_log_dir=_steps_dir(tmp_path)) for seed in (1, 2)]

    run_tasks(tasks, out, workers=2)

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len({record["step_log"] for record in records}) == 2
    for record in records:
        assert len(read_episode_step_log(tmp_path, record)) == _N * record["steps"]


def test_the_record_carries_the_section_40_per_episode_summary_names(tmp_path):
    record = run_episode_task(_task(model="adaptive_mappo", actor_weights_path=_random_weights(tmp_path)))

    for key in (
        "collision_count", "unsafe_conjunction_count", "total_delta_v", "maneuver_count",
        "mission_deviation_summary", "shield_intervention_count", "return",
    ):
        assert key in record, key
    assert record["total_delta_v"] == record["total_delta_v_mps"]
    assert record["return"] == record["return_per_agent"]
    assert record["mission_deviation_summary"] == {
        "mean": record["mean_mission_deviation"], "max": record["max_mission_deviation"],
    }
    assert record["mission_deviation_summary"]["max"] >= record["mission_deviation_summary"]["mean"]
    json.dumps(record, allow_nan=False)  # an ok record is strict JSON
