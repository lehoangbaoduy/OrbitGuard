"""Tests for orbit_guard.rl.rllib_env: the `[local | global_state]` wrapper.

Run against the frozen `StubOrbitGuardEnv`, whose `state()` interface the real
`OrbitGuardEnv` mirrors exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces.utils import flatten

from orbit_guard.rl.rllib_env import CTDEParallelPettingZooEnv, make_env_creator
from tests.rl.stub_env import AGENT_OBS_FLAT_DIM, StubOrbitGuardEnv

_NUM_AGENTS = 2
_GLOBAL_DIM = _NUM_AGENTS * AGENT_OBS_FLAT_DIM


def _wrapped_env(include_global_state: bool = True) -> CTDEParallelPettingZooEnv:
    return CTDEParallelPettingZooEnv(StubOrbitGuardEnv(num_agents=_NUM_AGENTS), include_global_state)


def test_observation_is_local_slice_followed_by_global_state():
    env = _wrapped_env()

    observations, _ = env.reset(seed=0)

    for obs in observations.values():
        assert obs.shape == (AGENT_OBS_FLAT_DIM + _GLOBAL_DIM,)
        assert obs.dtype == np.float32


def test_global_slice_equals_the_envs_state_after_reset():
    env = _wrapped_env()

    observations, _ = env.reset(seed=1)

    for obs in observations.values():
        np.testing.assert_array_equal(obs[AGENT_OBS_FLAT_DIM:], env.par_env.state())


def test_global_slice_tracks_the_envs_state_after_a_step():
    env = _wrapped_env()
    env.reset(seed=2)

    observations, _, _, _, _ = env.step({"sat_0": 0, "sat_1": 0})

    for obs in observations.values():
        np.testing.assert_array_equal(obs[AGENT_OBS_FLAT_DIM:], env.par_env.state())


def test_local_slice_is_that_agents_own_flattened_observation():
    inner = StubOrbitGuardEnv(num_agents=_NUM_AGENTS)
    env = CTDEParallelPettingZooEnv(inner)

    observations, _ = env.reset(seed=3)

    for agent, obs in observations.items():
        expected_local = flatten(inner.observation_space(agent), inner._last_observations[agent])
        np.testing.assert_array_equal(obs[:AGENT_OBS_FLAT_DIM], expected_local)


def test_local_only_mode_omits_the_global_state():
    env = _wrapped_env(include_global_state=False)

    observations, _ = env.reset(seed=4)

    assert env.global_state_dim == 0
    assert all(obs.shape == (AGENT_OBS_FLAT_DIM,) for obs in observations.values())


def test_exposes_local_and_global_dimensions_for_the_module_config():
    env = _wrapped_env()

    assert env.local_obs_dim == AGENT_OBS_FLAT_DIM
    assert env.global_state_dim == _GLOBAL_DIM


def test_observations_lie_inside_the_declared_observation_space():
    env = _wrapped_env()

    observations, _ = env.reset(seed=5)

    for agent, obs in observations.items():
        assert env.observation_space[agent].contains(obs)


def test_rejects_global_state_mode_for_an_env_without_state():
    class NoStateEnv(StubOrbitGuardEnv):
        state = None

    with pytest.raises(ValueError):
        CTDEParallelPettingZooEnv(NoStateEnv(num_agents=_NUM_AGENTS))


def test_env_creator_builds_a_wrapped_env_from_any_factory():
    creator = make_env_creator(lambda config: StubOrbitGuardEnv(num_agents=_NUM_AGENTS))

    env = creator({})

    assert isinstance(env, CTDEParallelPettingZooEnv)


def test_reward_offset_and_scale_reshape_rewards_for_the_learner_only():
    inner = StubOrbitGuardEnv(num_agents=_NUM_AGENTS)
    plain = CTDEParallelPettingZooEnv(StubOrbitGuardEnv(num_agents=_NUM_AGENTS))
    shaped = CTDEParallelPettingZooEnv(inner, reward_scale=0.5, reward_offset=-1.0)
    plain.reset(seed=6)
    shaped.reset(seed=6)

    _, plain_rewards, _, _, _ = plain.step({"sat_0": 0, "sat_1": 0})
    _, shaped_rewards, _, _, _ = shaped.step({"sat_0": 0, "sat_1": 0})

    for agent in plain_rewards:
        assert shaped_rewards[agent] == pytest.approx((plain_rewards[agent] - 1.0) * 0.5)


def test_rejects_a_non_positive_reward_scale():
    with pytest.raises(ValueError):
        CTDEParallelPettingZooEnv(StubOrbitGuardEnv(num_agents=_NUM_AGENTS), reward_scale=0.0)
