"""Tests for orbit_guard.scenarios.small_n_scenario (Phase 3 Part A).

Mirrors conjunction_generator's own test coverage pattern, scoped to the
N<10 behavior that generate_conjunction_scenario itself refuses to run.
"""

from __future__ import annotations

import pytest

from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.scenarios.constellation import ShellDistribution
from orbit_guard.scenarios.small_n_scenario import (
    generate_small_n_conjunction_scenario,
    sample_small_n_satellites,
)

_CAPABILITY = SatelliteCapability(
    max_delta_v_per_action_mps=0.1,
    remaining_delta_v_mps=2.0,
    maneuver_enabled=True,
    mission_deviation_limit=1.0,
)

_SHELL = ShellDistribution(
    altitude_km_mean=463.0, altitude_km_std=1.5, altitude_km_min=460.0, altitude_km_max=465.0,
    inclination_deg_mean=53.06, inclination_deg_std=0.03, inclination_deg_min=53.04, inclination_deg_max=53.17,
    eccentricity_mean=0.0002, eccentricity_std=0.00005, eccentricity_min=0.0001, eccentricity_max=0.0004,
)

_EPISODE_HORIZON_S = 21_600.0


def test_sample_small_n_satellites_rejects_n_below_one():
    with pytest.raises(ValueError):
        sample_small_n_satellites(0, seed=0, shell=_SHELL, capability_template=_CAPABILITY)


def test_sample_small_n_satellites_returns_exactly_n():
    satellites = sample_small_n_satellites(2, seed=0, shell=_SHELL, capability_template=_CAPABILITY)
    assert len(satellites) == 2


def test_sample_small_n_satellites_is_deterministic_under_seed():
    a = sample_small_n_satellites(2, seed=42, shell=_SHELL, capability_template=_CAPABILITY)
    b = sample_small_n_satellites(2, seed=42, shell=_SHELL, capability_template=_CAPABILITY)
    assert a[0].state.position_m == b[0].state.position_m
    assert a[1].state.position_m == b[1].state.position_m


def test_generate_small_n_conjunction_scenario_rejects_n_below_two():
    with pytest.raises(ValueError):
        generate_small_n_conjunction_scenario(
            1, seed=0, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=_EPISODE_HORIZON_S
        )


def test_multi_conjunction_never_sampled_below_four_satellites():
    for seed in range(200):
        scenario = generate_small_n_conjunction_scenario(
            2, seed=seed, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=_EPISODE_HORIZON_S
        )
        assert scenario.episode_type != "multi_conjunction"


def test_single_conjunction_episode_injects_exactly_one_pair_at_n_equals_two():
    for seed in range(50):
        scenario = generate_small_n_conjunction_scenario(
            2, seed=seed, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=_EPISODE_HORIZON_S
        )
        if scenario.episode_type == "single_conjunction":
            assert len(scenario.injected_conjunctions) == 1
            spec = scenario.injected_conjunctions[0]
            assert {spec.primary_index, spec.secondary_index} == {0, 1}
            return
    pytest.fail("no single_conjunction episode sampled in 50 seeds -- mix renormalization may be broken")


def test_benign_episode_injects_nothing():
    for seed in range(200):
        scenario = generate_small_n_conjunction_scenario(
            2, seed=seed, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=_EPISODE_HORIZON_S
        )
        if scenario.episode_type == "benign":
            assert scenario.injected_conjunctions == ()
            return
    pytest.fail("no benign episode sampled in 200 seeds")


def test_injected_conjunction_achieves_target_separation_closely():
    for seed in range(50):
        scenario = generate_small_n_conjunction_scenario(
            2, seed=seed, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=_EPISODE_HORIZON_S
        )
        if scenario.injected_conjunctions:
            spec = scenario.injected_conjunctions[0]
            tolerance_m = max(10.0, 0.15 * spec.target_d_min_m)
            assert abs(spec.achieved_d_min_m - spec.target_d_min_m) <= tolerance_m
            return
    pytest.fail("no injected conjunction sampled in 50 seeds")


def test_is_deterministic_under_seed():
    a = generate_small_n_conjunction_scenario(
        2, seed=7, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=_EPISODE_HORIZON_S
    )
    b = generate_small_n_conjunction_scenario(
        2, seed=7, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=_EPISODE_HORIZON_S
    )
    assert a.episode_type == b.episode_type
    assert a.satellites[0].state.position_m == b.satellites[0].state.position_m
