"""Conformance tests for tests/rl/stub_env.StubOrbitGuardEnv.

This is test INFRASTRUCTURE (not a Part B/C frozen handoff spec): it lets
Part C build and unit-test `orbit_guard.rl.training` against a real,
API-conformant PettingZoo ParallelEnv before Part A's actual OrbitGuardEnv
exists. Includes PettingZoo's own official conformance checker
(`pettingzoo.test.parallel_api_test`) so Part C inherits that guarantee for
free, plus a few shape/behavior checks specific to the frozen dimensions
this project has agreed on.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces.utils import flatten
from pettingzoo.test import parallel_api_test

from tests.rl.stub_env import (
    AGENT_OBS_FLAT_DIM,
    EPISODE_HORIZON_STEPS,
    GLOBAL_SUMMARY_DIM,
    K_NEIGHBORS,
    NEIGHBOR_DIM,
    NUM_ACTIONS,
    OWN_DIM,
    StubOrbitGuardEnv,
)


def test_passes_pettingzoo_official_parallel_api_conformance_check():
    env = StubOrbitGuardEnv(num_agents=2)
    parallel_api_test(env, num_cycles=50)


def test_reset_returns_observations_for_all_agents_with_frozen_shapes():
    env = StubOrbitGuardEnv(num_agents=2)
    observations, _infos = env.reset(seed=0)
    assert set(observations.keys()) == {"sat_0", "sat_1"}
    for obs in observations.values():
        assert obs["own"].shape == (OWN_DIM,)
        assert obs["neighbors"].shape == (K_NEIGHBORS, NEIGHBOR_DIM)
        assert obs["global_summary"].shape == (GLOBAL_SUMMARY_DIM,)


def test_action_space_is_discrete_with_six_maneuver_intents():
    env = StubOrbitGuardEnv(num_agents=2)
    env.reset(seed=0)
    assert env.action_space("sat_0").n == NUM_ACTIONS


def test_episode_truncates_at_the_configured_horizon():
    env = StubOrbitGuardEnv(num_agents=2, episode_horizon_steps=3)
    env.reset(seed=0)
    for step_index in range(3):
        _, _, _, truncations, _ = env.step({"sat_0": 0, "sat_1": 0})
    assert all(truncations.values())
    assert env.agents == []


def test_step_rejects_an_out_of_range_action():
    env = StubOrbitGuardEnv(num_agents=2)
    env.reset(seed=0)
    try:
        env.step({"sat_0": NUM_ACTIONS, "sat_1": 0})
        assert False, "expected ValueError for out-of-range action"
    except ValueError:
        pass


def test_reset_with_seed_is_reproducible():
    env1 = StubOrbitGuardEnv(num_agents=2)
    env2 = StubOrbitGuardEnv(num_agents=2)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert (obs1["sat_0"]["own"] == obs2["sat_0"]["own"]).all()


def test_num_agents_can_scale_beyond_two():
    env = StubOrbitGuardEnv(num_agents=5)
    observations, _ = env.reset(seed=0)
    assert len(observations) == 5


def test_rejects_fewer_than_two_agents():
    try:
        StubOrbitGuardEnv(num_agents=1)
        assert False, "expected ValueError for num_agents < 2"
    except ValueError:
        pass


def test_default_episode_horizon_matches_frozen_constant():
    env = StubOrbitGuardEnv(num_agents=2)
    assert env._episode_horizon_steps == EPISODE_HORIZON_STEPS


def test_centralized_critic_state_has_the_frozen_concatenated_shape():
    env = StubOrbitGuardEnv(num_agents=2)
    env.reset(seed=0)
    assert env.state_space.shape == (2 * AGENT_OBS_FLAT_DIM,)
    assert env.state().shape == (2 * AGENT_OBS_FLAT_DIM,)


def test_centralized_critic_state_is_the_concatenation_of_each_agents_own_observation():
    env = StubOrbitGuardEnv(num_agents=2)
    observations, _infos = env.reset(seed=0)
    state = env.state()
    agent_0_flat = flatten(env.observation_space("sat_0"), observations["sat_0"])
    agent_1_flat = flatten(env.observation_space("sat_1"), observations["sat_1"])
    assert np.allclose(state[:AGENT_OBS_FLAT_DIM], agent_0_flat)
    assert np.allclose(state[AGENT_OBS_FLAT_DIM:], agent_1_flat)


def test_centralized_critic_state_updates_after_step():
    env = StubOrbitGuardEnv(num_agents=2)
    env.reset(seed=0)
    state_before = env.state()
    env.step({"sat_0": 0, "sat_1": 0})
    state_after = env.state()
    assert not np.array_equal(state_before, state_after)
