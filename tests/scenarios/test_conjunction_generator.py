"""Contract tests for orbit_guard.scenarios.conjunction_generator (PRD §17).

Injection mechanism (back-propagation from the encounter, not forward search):
  1. Round the target TCA onto the physics_dt grid: t*.
  2. Propagate the primary satellite forward to t* -> (r_i, v_i).
  3. Rotate v_i about r_hat_i by a sampled crossing angle -> a relative
     velocity delta_v (magnitude 2|v_i|sin(theta/2)), covering §17's
     "relative velocity / orbital-plane relationship" parameters.
  4. Offset r_j(t*) = r_i + d_target * n_hat, n_hat perpendicular to delta_v,
     so Delta_r . Delta_v = 0 at t* -- t* really is the local minimum.
  5. Two-body+J2 acceleration depends only on position (verified against
     orbit_guard.physics.dynamics.state_derivative), so it is time-reversible:
     back-propagate the secondary to t=0 by forward-propagating
     (r_j(t*), -v_j(t*)) for t* seconds and negating the final velocity.

This module never imports orbit_guard.conjunction (Part B, owned by a
different teammate and not yet implemented) -- these tests verify achieved
geometry with a test-local distance helper instead, exactly like Part B's
own test files verify TCA/collision without depending on anything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.scenarios.conjunction_generator import (
    DEFAULT_DISTRIBUTION_MIX,
    ConjunctionScenario,
    ConjunctionSpec,
    generate_conjunction_scenario,
    inject_conjunction,
    sample_episode_type,
)
from orbit_guard.scenarios.constellation import (
    fit_shell_distribution,
    load_tle_snapshot,
    sample_constellation,
)

_TLE_PATH = "data/TLE_snapshot_53deg_shell.csv"
_PHYSICS_DT_S = 10.0


def _shell():
    return fit_shell_distribution(load_tle_snapshot(_TLE_PATH))


def _capability() -> SatelliteCapability:
    return SatelliteCapability(
        max_delta_v_per_action_mps=0.1,
        remaining_delta_v_mps=2.0,
        maneuver_enabled=True,
        mission_deviation_limit=1.0,
    )


def _min_separation_over_window(traj_i: np.ndarray, traj_j: np.ndarray) -> tuple[int, float]:
    """Test-local discrete minimum-separation helper -- deliberately not
    imported from orbit_guard.conjunction (Part B), which doesn't exist yet."""
    distances = np.linalg.norm(traj_i[:, :3] - traj_j[:, :3], axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def _first_satellite(seed: int = 20260927):
    shell = _shell()
    cap = _capability()
    satellites = sample_constellation(15, seed=seed, shell=shell, capability_template=cap)
    return satellites[0], cap


# --- sample_episode_type ------------------------------------------------------

def test_sample_episode_type_is_deterministic_for_a_given_rng_state():
    assert sample_episode_type(np.random.default_rng(3)) == sample_episode_type(np.random.default_rng(3))


def test_sample_episode_type_matches_configured_proportions_statistically():
    counts = {"benign": 0, "single_conjunction": 0, "multi_conjunction": 0}
    rng = np.random.default_rng(0)
    n_draws = 5000
    for _ in range(n_draws):
        counts[sample_episode_type(rng)] += 1

    assert counts["benign"] / n_draws == pytest.approx(0.30, abs=0.03)
    assert counts["single_conjunction"] / n_draws == pytest.approx(0.50, abs=0.03)
    assert counts["multi_conjunction"] / n_draws == pytest.approx(0.20, abs=0.03)


def test_sample_episode_type_rejects_mix_not_summing_to_one():
    with pytest.raises(ValueError):
        sample_episode_type(np.random.default_rng(0), distribution_mix={"benign": 0.5, "single_conjunction": 0.6})


def test_sample_episode_type_only_returns_configured_labels():
    rng = np.random.default_rng(1)
    for _ in range(200):
        assert sample_episode_type(rng) in DEFAULT_DISTRIBUTION_MIX


# --- inject_conjunction --------------------------------------------------------

def test_inject_conjunction_achieves_target_separation_at_target_tca():
    primary, cap = _first_satellite()
    target_tca_s, target_d_min_m = 900.0, 150.0

    _secondary, achieved_tca_s, achieved_d_min_m = inject_conjunction(
        primary, cap, target_tca_s, target_d_min_m, crossing_angle_rad=np.radians(5.0), physics_dt_s=_PHYSICS_DT_S
    )

    assert achieved_tca_s == pytest.approx(target_tca_s, abs=_PHYSICS_DT_S)
    assert achieved_d_min_m == pytest.approx(target_d_min_m, rel=0.15)


def test_inject_conjunction_forward_propagation_confirms_achieved_geometry():
    # Independent re-verification: forward-propagate BOTH satellites from
    # t=0 (not trusting the module's own internal self-check) and confirm
    # the discrete minimum lands at achieved_tca_s within one dt, with a
    # separation close to achieved_d_min_m.
    primary, cap = _first_satellite()
    target_tca_s, target_d_min_m = 600.0, 120.0
    n_steps = round(target_tca_s / _PHYSICS_DT_S)

    secondary, achieved_tca_s, achieved_d_min_m = inject_conjunction(
        primary, cap, target_tca_s, target_d_min_m, crossing_angle_rad=np.radians(8.0), physics_dt_s=_PHYSICS_DT_S
    )

    traj_i = propagate(primary.state.as_vector(), _PHYSICS_DT_S, n_steps + 5, state_derivative)
    traj_j = propagate(secondary.state.as_vector(), _PHYSICS_DT_S, n_steps + 5, state_derivative)
    idx, d_min = _min_separation_over_window(traj_i, traj_j)

    assert idx * _PHYSICS_DT_S == pytest.approx(achieved_tca_s, abs=_PHYSICS_DT_S)
    assert d_min == pytest.approx(achieved_d_min_m, rel=0.2)


def test_inject_conjunction_can_hit_a_sub_hard_body_radius_collision_course():
    # A genuine ground-truth collision-course band must be reachable --
    # otherwise the RL collision penalty never fires in injected data.
    primary, cap = _first_satellite()

    _secondary, _achieved_tca_s, achieved_d_min_m = inject_conjunction(
        primary, cap, target_tca_s=500.0, target_d_min_m=3.0,
        crossing_angle_rad=np.radians(6.0), physics_dt_s=_PHYSICS_DT_S,
    )

    assert achieved_d_min_m < 10.0  # combined hard-body radius (PRD §18)


@pytest.mark.parametrize("bad_kwargs", [
    {"target_tca_s": 0.0, "target_d_min_m": 100.0, "crossing_angle_rad": 0.1, "physics_dt_s": 10.0},
    {"target_tca_s": 500.0, "target_d_min_m": -1.0, "crossing_angle_rad": 0.1, "physics_dt_s": 10.0},
    {"target_tca_s": 500.0, "target_d_min_m": 100.0, "crossing_angle_rad": 0.0, "physics_dt_s": 10.0},
    {"target_tca_s": 500.0, "target_d_min_m": 100.0, "crossing_angle_rad": 0.1, "physics_dt_s": 0.0},
])
def test_inject_conjunction_rejects_invalid_inputs(bad_kwargs):
    primary, cap = _first_satellite()
    with pytest.raises(ValueError):
        inject_conjunction(primary, cap, **bad_kwargs)


def test_inject_conjunction_rejects_tca_too_small_to_align_to_grid():
    # Positive but rounds to 0 steps at this physics_dt -- distinct from the
    # target_tca_s <= 0 case already covered above.
    primary, cap = _first_satellite()
    with pytest.raises(ValueError):
        inject_conjunction(
            primary, cap, target_tca_s=2.0, target_d_min_m=100.0,
            crossing_angle_rad=np.radians(5.0), physics_dt_s=10.0,
        )


def test_inject_conjunction_rejects_crossing_angle_too_small_but_positive():
    # Distinct from the crossing_angle_rad <= 0 case already covered above:
    # a positive but vanishingly small angle also can't be distinguished.
    primary, cap = _first_satellite()
    with pytest.raises(ValueError):
        inject_conjunction(
            primary, cap, target_tca_s=500.0, target_d_min_m=100.0,
            crossing_angle_rad=1e-12, physics_dt_s=10.0,
        )


# --- generate_conjunction_scenario ---------------------------------------------

def _find_seed_for_episode_type(target_type: str, mix=DEFAULT_DISTRIBUTION_MIX, search_range: int = 500) -> int:
    for candidate_seed in range(search_range):
        if sample_episode_type(np.random.default_rng(candidate_seed), mix) == target_type:
            return candidate_seed
    raise AssertionError(f"No seed in range({search_range}) produced episode type {target_type!r}")


def test_benign_episode_preserves_natural_sampling_unmodified():
    n, shell, cap, horizon = 15, _shell(), _capability(), 3600.0
    seed = _find_seed_for_episode_type("benign")

    scenario = generate_conjunction_scenario(n, seed, shell, cap, episode_horizon_s=horizon)
    expected_satellites = sample_constellation(n, seed, shell, cap)

    assert isinstance(scenario, ConjunctionScenario)
    assert scenario.episode_type == "benign"
    assert scenario.injected_conjunctions == ()
    assert scenario.satellites == expected_satellites


def test_single_conjunction_episode_injects_exactly_one_pair_and_keeps_n_fixed():
    n, shell, cap, horizon = 15, _shell(), _capability(), 3600.0
    seed = _find_seed_for_episode_type("single_conjunction")

    scenario = generate_conjunction_scenario(n, seed, shell, cap, episode_horizon_s=horizon)

    assert scenario.episode_type == "single_conjunction"
    assert len(scenario.satellites) == n
    assert len(scenario.injected_conjunctions) == 1
    spec = scenario.injected_conjunctions[0]
    assert isinstance(spec, ConjunctionSpec)
    assert spec.primary_index != spec.secondary_index
    assert spec.achieved_d_min_m == pytest.approx(spec.target_d_min_m, rel=0.2)


def test_multi_conjunction_episode_injects_disjoint_pairs():
    n, shell, cap, horizon = 15, _shell(), _capability(), 3600.0
    seed = _find_seed_for_episode_type("multi_conjunction")

    scenario = generate_conjunction_scenario(n, seed, shell, cap, episode_horizon_s=horizon)

    assert scenario.episode_type == "multi_conjunction"
    assert len(scenario.satellites) == n
    assert len(scenario.injected_conjunctions) == 2

    touched_indices = []
    for spec in scenario.injected_conjunctions:
        touched_indices.extend([spec.primary_index, spec.secondary_index])
    assert len(set(touched_indices)) == 4  # fully disjoint pairs


def test_generate_conjunction_scenario_is_deterministic_under_seed():
    n, shell, cap, horizon = 15, _shell(), _capability(), 3600.0
    seed = _find_seed_for_episode_type("single_conjunction")

    scenario_a = generate_conjunction_scenario(n, seed, shell, cap, episode_horizon_s=horizon)
    scenario_b = generate_conjunction_scenario(n, seed, shell, cap, episode_horizon_s=horizon)

    assert scenario_a.episode_type == scenario_b.episode_type
    assert scenario_a.satellites == scenario_b.satellites
    assert scenario_a.injected_conjunctions == scenario_b.injected_conjunctions


def test_generate_conjunction_scenario_differs_across_seeds():
    n, shell, cap, horizon = 15, _shell(), _capability(), 3600.0
    scenarios = [generate_conjunction_scenario(n, seed, shell, cap, episode_horizon_s=horizon) for seed in range(5)]

    distinct_satellite_sets = {scenario.satellites for scenario in scenarios}
    assert len(distinct_satellite_sets) > 1


def test_generate_conjunction_scenario_rejects_n_below_supported_range():
    shell, cap = _shell(), _capability()
    with pytest.raises(ValueError):
        generate_conjunction_scenario(3, 0, shell, cap, episode_horizon_s=3600.0)


def test_generate_conjunction_scenario_can_reach_the_collision_course_band():
    # Wires the sub-10m collision-course branch all the way through the
    # public generator entry point, not just via a direct inject_conjunction
    # call -- this is what makes the ground-truth collision penalty ever
    # fire on injected (not just naturally-sampled) episodes.
    n, shell, cap, horizon = 15, _shell(), _capability(), 3600.0
    for candidate_seed in range(300):
        if sample_episode_type(np.random.default_rng(candidate_seed)) != "single_conjunction":
            continue
        scenario = generate_conjunction_scenario(n, candidate_seed, shell, cap, episode_horizon_s=horizon)
        if scenario.injected_conjunctions[0].achieved_d_min_m < 10.0:
            return  # found a collision-course episode; branch exercised
    raise AssertionError("No seed in range(300) produced a collision-course injection")
