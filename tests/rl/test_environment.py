"""Tests for orbit_guard.rl.environment.OrbitGuardEnv (Phase 3 Part A, PRD §19-28, §37).

Verified against the throwaway Part B/C stand-ins described in
orbit_guard/safety/maneuver_mapper.py, orbit_guard/rl/observation.py, and
orbit_guard/rl/reward.py (each explicitly marked "Not for commit to main").
These tests exercise OrbitGuardEnv's OWN orchestration logic -- scenario
construction, capability enforcement, collision/termination handling,
PettingZoo API compliance -- not Part B/C's internal numerics, which are
covered by their own frozen suites (tests/safety/test_maneuver_mapper.py,
tests/rl/test_observation.py, tests/rl/test_reward.py).

Uses small physics_dt_s/lookahead_s/episode_horizon_s throughout so the
belief-prediction pipeline (RK4 propagation per satellite per step) stays
fast; production defaults in OrbitGuardEnv.__init__ are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces.utils import flatten
from pettingzoo.test import parallel_api_test, parallel_seed_test

from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.safety.maneuver_mapper import ManeuverIntent
from tests.rl.stub_env import StubOrbitGuardEnv

_FAST_KWARGS = dict(episode_horizon_s=120.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=60.0)


def _fast_env(num_agents: int = 2, **overrides) -> OrbitGuardEnv:
    kwargs = {**_FAST_KWARGS, **overrides}
    return OrbitGuardEnv(num_agents=num_agents, **kwargs)


def test_rejects_fewer_than_two_agents():
    with pytest.raises(ValueError):
        _fast_env(num_agents=1)


def test_reset_returns_observations_and_infos_for_every_agent():
    env = _fast_env()
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)


def test_observation_shapes_match_frozen_contract():
    env = _fast_env(k_neighbors=4)
    obs, _ = env.reset(seed=1)
    for agent in env.agents:
        assert obs[agent]["own"].shape == (10,)
        assert obs[agent]["neighbors"].shape == (4, 10)
        assert obs[agent]["global_summary"].shape == (4,)
        assert env.observation_space(agent).contains(obs[agent])


def test_reset_is_deterministic_under_seed():
    env_a = _fast_env()
    env_b = _fast_env()
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    for agent in env_a.possible_agents:
        np.testing.assert_array_equal(obs_a[agent]["own"], obs_b[agent]["own"])
        np.testing.assert_array_equal(obs_a[agent]["neighbors"], obs_b[agent]["neighbors"])


def test_different_seeds_produce_different_initial_observations():
    env_a = _fast_env()
    env_b = _fast_env()
    obs_a, _ = env_a.reset(seed=1)
    obs_b, _ = env_b.reset(seed=2)
    agent = env_a.possible_agents[0]
    assert not np.allclose(obs_a[agent]["own"], obs_b[agent]["own"])


def test_step_rejects_out_of_range_action():
    env = _fast_env()
    env.reset(seed=0)
    actions = dict.fromkeys(env.agents, 0)
    actions[env.agents[0]] = 99
    with pytest.raises(ValueError):
        env.step(actions)


def test_episode_truncates_at_the_configured_horizon():
    env = _fast_env()
    env.reset(seed=3)
    expected_steps = round(_FAST_KWARGS["episode_horizon_s"] / _FAST_KWARGS["control_dt_s"])
    step_count = 0
    while env.agents:
        actions = {agent: 0 for agent in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
        step_count += 1
        assert step_count <= expected_steps
    assert step_count == expected_steps
    assert all(truncations.values())
    assert not any(terminations.values())


def test_no_maneuver_action_never_increases_remaining_delta_v_budget():
    env = _fast_env()
    env.reset(seed=4)
    budgets_before = [cap.remaining_delta_v_mps for cap in env._capabilities]
    env.step({agent: int(ManeuverIntent.NO_MANEUVER) for agent in env.agents})
    budgets_after = [cap.remaining_delta_v_mps for cap in env._capabilities]
    assert budgets_after == budgets_before


def test_maneuver_action_decreases_remaining_delta_v_budget():
    env = _fast_env()
    env.reset(seed=5)
    budget_before = env._capabilities[0].remaining_delta_v_mps
    env.step({agent: int(ManeuverIntent.RAISE_ORBIT) for agent in env.agents})
    assert env._capabilities[0].remaining_delta_v_mps < budget_before


def test_maneuver_disabled_capability_never_applies_delta_v():
    disabled_capability = SatelliteCapability(
        max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=2.0,
        maneuver_enabled=False, mission_deviation_limit=1.0,
    )
    env = _fast_env(capability_template=disabled_capability)
    env.reset(seed=6)
    ground_truth_before = [gt.copy() for gt in env._ground_truth]
    env.step({agent: int(ManeuverIntent.RAISE_ORBIT) for agent in env.agents})
    budgets_after = [cap.remaining_delta_v_mps for cap in env._capabilities]
    assert budgets_after == [2.0] * len(budgets_after)

    # No applied delta-v means the post-step ground truth is exactly the
    # natural (unmaneuvered) propagation of the pre-step state.
    expected = [
        propagate(gt, _FAST_KWARGS["physics_dt_s"], env._substeps_per_control, state_derivative)[-1]
        for gt in ground_truth_before
    ]
    for actual, exp in zip(env._ground_truth, expected, strict=True):
        np.testing.assert_allclose(actual, exp)


def test_insufficient_budget_caps_applied_magnitude_not_substitutes_no_maneuver():
    tight_capability = SatelliteCapability(
        max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=0.02,
        maneuver_enabled=True, mission_deviation_limit=1.0,
    )
    env = _fast_env(capability_template=tight_capability)
    env.reset(seed=7)
    applied = env._apply_actions({agent: int(ManeuverIntent.RAISE_ORBIT) for agent in env.agents})
    for idx in range(env._num_agents):
        magnitude = float(np.linalg.norm(applied[idx]))
        assert magnitude == pytest.approx(0.02, rel=1e-9)
        assert magnitude > 0.0


def test_collision_terminates_only_the_involved_agents_at_n_equals_two():
    env = _fast_env(num_agents=2, hard_body_radius_m=1.0e7)
    env.reset(seed=8)
    _, _, terminations, _, _ = env.step({agent: 0 for agent in env.agents})
    assert all(terminations.values())
    assert env.agents == []


def test_no_collision_with_zero_hard_body_radius_never_terminates_early():
    env = _fast_env(num_agents=2, hard_body_radius_m=0.0)
    env.reset(seed=9)
    step_count = 0
    while env.agents:
        _, _, terminations, truncations, _ = env.step({agent: 0 for agent in env.agents})
        if any(terminations.values()):
            pytest.fail("spurious collision with zero hard-body radius")
        step_count += 1
    assert step_count > 0


def test_state_is_concatenation_of_each_agents_own_observation():
    env = _fast_env()
    obs, _ = env.reset(seed=10)
    state = env.state()
    assert state.shape == env.state_space.shape
    flat_dim = env.state_space.shape[0] // len(env.possible_agents)
    for slot, agent in enumerate(env.possible_agents):
        expected = flatten(env.observation_space(agent), obs[agent])
        np.testing.assert_allclose(state[slot * flat_dim : (slot + 1) * flat_dim], expected, atol=1e-5)


def test_n_equals_ten_routes_through_the_phase_two_generator():
    env = _fast_env(num_agents=10)
    obs, infos = env.reset(seed=11)
    assert len(env.agents) == 10
    obs, rewards, terminations, truncations, infos = env.step({agent: 0 for agent in env.agents})
    assert len(rewards) == 10


def test_swap_in_matches_stub_env_interface():
    stub = StubOrbitGuardEnv(num_agents=2)
    real = _fast_env(num_agents=2)
    assert real.possible_agents == stub.possible_agents
    real.reset(seed=0)
    stub.reset(seed=0)
    for agent in real.possible_agents:
        assert real.observation_space(agent).spaces.keys() == stub.observation_space(agent).spaces.keys()
        for key in real.observation_space(agent).spaces:
            assert real.observation_space(agent)[key].shape == stub.observation_space(agent)[key].shape
        assert real.action_space(agent).n == stub.action_space(agent).n
    assert real.state_space.shape == stub.state_space.shape
    assert real.state().shape == stub.state().shape


def test_parallel_api_compliance():
    env = _fast_env()
    parallel_api_test(env, num_cycles=20)


def test_parallel_seed_reproducibility():
    parallel_seed_test(lambda: _fast_env(), num_cycles=4)
