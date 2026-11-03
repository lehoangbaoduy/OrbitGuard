"""Acceptance Test F (PRD §38): same config + seed reproduces the same simulation and shield outputs.

In-process repeats are checked first. The cross-process tests then run the same (model, shift condition,
seed) episodes with a fixed scripted policy in two child Python processes started with DIFFERENT
PYTHONHASHSEED values, and compare sha256 digests of the whole trajectory (true states, observations, applied
delta-v, rewards, shield decisions, §40 step records, episode summary; see tests/acceptance/_repro_digest.py)
against the in-process run. They also check that the evaluation runner's record of one task is byte-identical
across the two processes once the timing fields are removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orbit_guard.evaluation.runner import EpisodeTask, run_episode_task
from orbit_guard.rl.actor_policy import ActorWeights, save_actor_weights
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel
from tests.acceptance import _repro_digest

_FAST = {"episode_horizon_s": 600.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0}


def _trace(seed: int):
    env = ShieldedOrbitGuardEnv(num_agents=2, noise_model=NoiseModel(0.0, 0.0), hard_body_radius_m=100.0, **_FAST)
    env.reset(seed=seed)
    trace = []
    while env.agents:
        _, rewards, _, _, _ = env.step({agent: 0 for agent in env.agents})
        result = env.last_shield_result
        trace.append((tuple(rewards.values()), result.approved, result.accepted, result.feasible))
    return trace, env.episode_summary()


def test_shield_decisions_rewards_and_summary_reproduce_exactly_from_the_same_seed():
    first_trace, first_summary = _trace(seed=3)
    second_trace, second_summary = _trace(seed=3)

    assert first_trace == second_trace
    assert first_summary == second_summary
    assert any(not accepted for _, _, accepted, _ in first_trace)  # the shield actually acted


def _random_actor_weights(path: Path) -> Path:
    rng = np.random.default_rng(1)
    sizes = [(128, 54), (128, 128), (6, 128)]
    layers = tuple((rng.normal(0, 0.1, size=s).astype(np.float32), np.zeros(s[0], np.float32)) for s in sizes)
    save_actor_weights(ActorWeights(layers), path)
    return path


def test_a_full_experiment_record_reproduces_from_the_same_task(tmp_path):
    weights = _random_actor_weights(tmp_path / "actor.npz")
    task = EpisodeTask(
        model="adaptive_mappo", condition="noise_increase", scenario_seed=5, n_agents=10, training_seed=0,
        actor_weights_path=str(weights), env_kwargs={**_FAST, "episode_horizon_s": 240.0, "lookahead_s": 300.0},
        experiment_id="f", git_commit="abc",
    )

    first = run_episode_task(task)
    second = run_episode_task(task)

    for key in ("start_time", "wall_time_s"):
        first.pop(key), second.pop(key)
    assert first == second


def test_different_seeds_do_not_collapse_to_the_same_trace():
    assert _trace(seed=3)[0] != _trace(seed=9)[0]


@pytest.mark.parametrize("seed", [3, 9])
def test_summary_counts_are_stable_across_repeats(seed):
    assert _trace(seed)[1].collision_count == _trace(seed)[1].collision_count


# --- cross-process ---------------------------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_CODE = "from tests.acceptance._repro_digest import main; main()"
_HASH_SEEDS = ("0", "4242")
_CHILD_TIMEOUT_S = 300
_SCRIPTED_SEED = 5
_SCRIPTED_N = 10
_SCRIPTED_KWARGS = {"episode_horizon_s": 600.0}  # 10 control steps; the shield rejects on seed 5 in both shifts
_SHIFTS = ("noise_increase", "capability_reduction")
_MODELS = ("adaptive_mappo_no_shield", "adaptive_mappo")  # base env and shielded env, same adaptive margin
_TRAJECTORIES = tuple(
    {"model": model, "condition": condition, "seed": _SCRIPTED_SEED, "n_agents": _SCRIPTED_N, "env_kwargs": _SCRIPTED_KWARGS}
    for condition in _SHIFTS
    for model in _MODELS
)


def _run_child(spec: dict, hash_seed: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_CODE, json.dumps(spec)],
        cwd=_REPO_ROOT, env={**os.environ, "PYTHONHASHSEED": hash_seed},
        capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def cross_process(tmp_path_factory) -> dict:
    weights = _random_actor_weights(tmp_path_factory.mktemp("cross_process") / "actor.npz")
    runner_task = {
        "model": "adaptive_mappo", "condition": "capability_reduction", "scenario_seed": _SCRIPTED_SEED,
        "n_agents": _SCRIPTED_N, "training_seed": 0, "actor_weights_path": str(weights),
        "env_kwargs": _SCRIPTED_KWARGS, "experiment_id": "f-cross", "git_commit": "abc",
    }
    spec = {"trajectories": list(_TRAJECTORIES), "runner_task": runner_task}
    children = [_run_child(spec, hash_seed) for hash_seed in _HASH_SEEDS]  # sequential: RAM is shared
    return {"children": children, "in_process": _repro_digest.compute(spec)}


def test_children_really_ran_under_different_hash_seeds(cross_process):
    first, second = cross_process["children"]
    assert first["str_hash"] != second["str_hash"]


@pytest.mark.parametrize("index", range(len(_TRAJECTORIES)), ids=lambda i: "{model}-{condition}".format(**_TRAJECTORIES[i]))
def test_trajectory_digest_is_identical_in_process_and_across_processes(cross_process, index):
    in_process = cross_process["in_process"]["trajectories"][index]
    for child in cross_process["children"]:
        assert child["trajectories"][index] == in_process


def test_the_compared_trajectories_actually_burn_and_trigger_the_shield(cross_process):
    trajectories = cross_process["in_process"]["trajectories"]
    by_model = {model: [t for spec, t in zip(_TRAJECTORIES, trajectories, strict=True) if spec["model"] == model] for model in _MODELS}

    assert all(t["steps"] == 10 and t["total_delta_v_mps"] > 0.0 for t in trajectories)
    assert all(t["shield_rejections"] > 0 for t in by_model["adaptive_mappo"])
    assert all(t["shield_rejections"] == 0 for t in by_model["adaptive_mappo_no_shield"])
    assert len({t["digest"] for t in trajectories}) == len(trajectories)  # each env/shift really differs


def test_the_digest_tells_seeds_apart():
    spec = {**_TRAJECTORIES[0], "seed": _SCRIPTED_SEED + 1}
    assert _repro_digest.trajectory_digest(**spec)["digest"] != _repro_digest.trajectory_digest(**_TRAJECTORIES[0])["digest"]


def test_runner_record_is_byte_identical_across_processes_apart_from_timing(cross_process):
    records = [child["runner_record"] for child in cross_process["children"]]
    in_process = cross_process["in_process"]["runner_record"]

    assert records[0] == records[1] == in_process
    record = json.loads(in_process)
    assert record["status"] == "ok"
    assert record["shift_type"] == "capability_reduction"
    assert not set(_repro_digest.TIMING_FIELDS) & set(record)
