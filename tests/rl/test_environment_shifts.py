"""Environment hooks the distribution-shift suite needs (PRD §32): density and capability."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.rl.environment import CapabilityDegradation, OrbitGuardEnv
from orbit_guard.scenarios.conjunction_generator import DEFAULT_DISTRIBUTION_MIX

# Fixed budgets (no per-episode draw): these tests pin the degradation logic against the 2.0 m/s template;
# tests/rl/test_delta_v_budget_randomization.py covers degradation on top of drawn budgets.
_KW = {
    "episode_horizon_s": 120.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 60.0,
    "remaining_delta_v_range_mps": None,
}
_SINGLE_ONLY = {"benign": 0.0, "single_conjunction": 1.0, "multi_conjunction": 0.0}


def _mean_pairwise_distance_m(env: OrbitGuardEnv) -> float:
    positions = np.array([gt[:3] for gt in env._ground_truth])
    diffs = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return float(distances[np.triu_indices(len(positions), k=1)].mean())


def test_phase_spread_packs_a_fixed_number_of_satellites_into_a_denser_neighbourhood():
    default = OrbitGuardEnv(num_agents=10, **_KW)
    dense = OrbitGuardEnv(num_agents=10, phase_spread_rad=0.02, **_KW)
    default.reset(seed=4)
    dense.reset(seed=4)

    assert len(dense.agents) == len(default.agents) == 10
    assert _mean_pairwise_distance_m(dense) < 0.1 * _mean_pairwise_distance_m(default)


def test_phase_spread_is_reproducible_under_a_seed():
    first = OrbitGuardEnv(num_agents=10, phase_spread_rad=0.02, **_KW)
    second = OrbitGuardEnv(num_agents=10, phase_spread_rad=0.02, **_KW)
    first.reset(seed=9)
    second.reset(seed=9)

    np.testing.assert_array_equal(np.array(first._ground_truth), np.array(second._ground_truth))


def test_phase_spread_is_only_supported_for_the_n_ge_10_generator():
    env = OrbitGuardEnv(num_agents=4, phase_spread_rad=0.02, **_KW)

    with pytest.raises(ValueError, match="phase_spread_rad"):
        env.reset(seed=0)


def test_no_degradation_leaves_every_capability_untouched():
    env = OrbitGuardEnv(num_agents=10, **_KW)
    env.reset(seed=1)

    assert all(cap.maneuver_enabled for cap in env._capabilities)
    assert all(cap.remaining_delta_v_mps == 2.0 for cap in env._capabilities)


def test_disabled_satellites_are_the_primaries_of_injected_conjunctions_first():
    env = OrbitGuardEnv(
        num_agents=10, distribution_mix=_SINGLE_ONLY, capability_degradation=CapabilityDegradation(disabled_count=1), **_KW
    )
    env.reset(seed=2)

    disabled = [i for i, cap in enumerate(env._capabilities) if not cap.maneuver_enabled]
    assert len(disabled) == 1
    assert disabled == [env._last_scenario.injected_conjunctions[0].primary_index]


def test_reduced_budget_satellites_keep_maneuver_enabled_but_have_less_delta_v():
    env = OrbitGuardEnv(
        num_agents=10,
        distribution_mix=_SINGLE_ONLY,
        capability_degradation=CapabilityDegradation(disabled_count=1, reduced_count=2, reduced_budget_scale=0.25),
        **_KW,
    )
    env.reset(seed=2)

    reduced = [i for i, cap in enumerate(env._capabilities) if cap.remaining_delta_v_mps == pytest.approx(0.5)]
    disabled = [i for i, cap in enumerate(env._capabilities) if not cap.maneuver_enabled]
    assert len(reduced) == 2
    assert set(reduced).isdisjoint(disabled)
    assert all(env._capabilities[i].maneuver_enabled for i in reduced)


def test_degradation_is_deterministic_and_identical_across_resets_of_the_same_seed():
    env = OrbitGuardEnv(
        num_agents=10, capability_degradation=CapabilityDegradation(disabled_count=2), distribution_mix=DEFAULT_DISTRIBUTION_MIX, **_KW
    )
    env.reset(seed=7)
    first = [cap.maneuver_enabled for cap in env._capabilities]
    env.reset(seed=7)

    assert [cap.maneuver_enabled for cap in env._capabilities] == first
    assert first.count(False) == 2


def test_degradation_validates_its_arguments():
    with pytest.raises(ValueError):
        CapabilityDegradation(disabled_count=-1)
    with pytest.raises(ValueError):
        CapabilityDegradation(disabled_count=0, reduced_budget_scale=0.0)


def test_max_delta_v_per_action_is_public_and_matches_what_a_burn_applies():
    env = OrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=0)
    assert env.max_delta_v_per_action_mps == pytest.approx(0.1)
    env.step({agent: 1 for agent in env.agents})  # RAISE_ORBIT: one full-size burn
    assert max(env.delta_v_used_mps().values()) == pytest.approx(env.max_delta_v_per_action_mps)


def _reduced_budget_env(**overrides) -> OrbitGuardEnv:
    return OrbitGuardEnv(
        num_agents=10,
        distribution_mix=_SINGLE_ONLY,
        capability_degradation=CapabilityDegradation(reduced_count=2, reduced_budget_scale=0.25),
        **{**_KW, **overrides},
    )


def test_a_reduced_budget_satellite_that_never_burns_has_used_no_delta_v():
    env = _reduced_budget_env()
    env.reset(seed=2)
    while env.agents:
        env.step({agent: 0 for agent in env.agents})

    assert env.delta_v_used_mps() == {agent: 0.0 for agent in env.possible_agents}


def test_initial_budgets_are_each_satellites_own_starting_budget_after_degradation():
    env = _reduced_budget_env()
    env.reset(seed=2)
    starting = [cap.remaining_delta_v_mps for cap in env._capabilities]

    env.step({agent: 1 for agent in env.agents})  # RAISE_ORBIT: one full-size burn each

    initial = env.initial_delta_v_budgets_mps()
    assert [initial[agent] for agent in env.possible_agents] == starting
    assert sorted(starting)[:2] == pytest.approx([0.25 * env.delta_v_budget_mps] * 2)
    used = env.delta_v_used_mps()
    for agent in env.possible_agents:
        remaining = env._capabilities[env._agent_index[agent]].remaining_delta_v_mps
        assert used[agent] == pytest.approx(initial[agent] - remaining)
        assert used[agent] == pytest.approx(env.max_delta_v_per_action_mps)
