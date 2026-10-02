"""Acceptance Test F (PRD §38): same config + seed reproduces the same simulation and shield outputs."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.evaluation.runner import EpisodeTask, run_episode_task
from orbit_guard.rl.actor_policy import ActorWeights, save_actor_weights
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel

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


def test_a_full_experiment_record_reproduces_from_the_same_task(tmp_path):
    rng = np.random.default_rng(1)
    sizes = [(128, 54), (128, 128), (6, 128)]
    layers = tuple((rng.normal(0, 0.1, size=s).astype(np.float32), np.zeros(s[0], np.float32)) for s in sizes)
    weights = tmp_path / "actor.npz"
    save_actor_weights(ActorWeights(layers), weights)
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
