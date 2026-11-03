"""Rule-Based burn direction is antisymmetric within a conjunction pair (PRD §31 #2).

Each pair member sees the other in ITS OWN RTN frame. The two observations are built here from
real 3D circular-orbit states projected with the production frame transform, not by negating one
vector, so frame tilt (crossing planes) and chord curvature (long range) are both present.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.evaluation.baselines import rule_based_policy
from orbit_guard.physics.constants import MU_EARTH_M3_S2
from orbit_guard.physics.frames import eci_vector_to_rtn
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_RADIUS_M = 7_000_000.0
_MEAN_MOTION = np.sqrt(MU_EARTH_M3_S2 / _RADIUS_M**3)
_POSITION_SCALE_M = 5_000.0  # rl/observation.py RELATIVE_POSITION_SCALE_M
_OPPOSED = {ManeuverIntent.RAISE_ORBIT, ManeuverIntent.LOWER_ORBIT}


def _circular_state(radius_m, tilt_rad, phase_rad, time_s):
    """Circular orbit through +x at phase 0, its plane tilted by `tilt_rad` about the x axis."""
    x_hat = np.array([1.0, 0.0, 0.0])
    t_hat = np.array([0.0, np.cos(tilt_rad), np.sin(tilt_rad)])
    mean_motion = np.sqrt(MU_EARTH_M3_S2 / radius_m**3)
    angle = mean_motion * time_s + phase_rad
    position = radius_m * (np.cos(angle) * x_hat + np.sin(angle) * t_hat)
    velocity = radius_m * mean_motion * (-np.sin(angle) * x_hat + np.cos(angle) * t_hat)
    return position, velocity


def _observation(relative_position_obs, risk=0.9):
    neighbors = np.zeros((4, 10))
    neighbors[0, 0:3] = relative_position_obs
    neighbors[0, 6] = risk
    neighbors[0, 9] = 1.0
    own = np.zeros(10)
    own[6] = 1.0
    own[9] = 1.0
    return {"own": own, "neighbors": neighbors, "global_summary": np.zeros(4)}


def _pair_observations(state_i, state_j):
    (r_i, v_i), (r_j, v_j) = state_i, state_j
    seen_by_i = eci_vector_to_rtn(r_j - r_i, r_i, v_i) / _POSITION_SCALE_M
    seen_by_j = eci_vector_to_rtn(r_i - r_j, r_j, v_j) / _POSITION_SCALE_M
    return _observation(seen_by_i), _observation(seen_by_j)


def _pair_actions(state_i, state_j):
    obs_i, obs_j = _pair_observations(state_i, state_j)
    return ManeuverIntent(rule_based_policy("sat_0", obs_i)), ManeuverIntent(rule_based_policy("sat_1", obs_j))


def _assert_opposed(action_i, action_j):
    assert {action_i, action_j} == _OPPOSED, (action_i, action_j)


@pytest.mark.parametrize("crossing_deg", [0.5, 5.0, 15.0])
@pytest.mark.parametrize("seconds_before_tca", [1020.0, 600.0, 60.0])
@pytest.mark.parametrize(("miss_radial_m", "miss_phase_m"), [(100.0, 0.0), (0.0, 100.0), (-60.0, 80.0)])
def test_crossing_plane_pair_members_burn_opposite_ways(crossing_deg, seconds_before_tca, miss_radial_m, miss_phase_m):
    half = np.radians(crossing_deg) / 2.0
    time_s = -seconds_before_tca
    state_i = _circular_state(_RADIUS_M, +half, 0.0, time_s)
    state_j = _circular_state(_RADIUS_M + miss_radial_m, -half, miss_phase_m / _RADIUS_M, time_s)

    _assert_opposed(*_pair_actions(state_i, state_j))


@pytest.mark.parametrize("phase_offset_m", [2_000.0, -2_000.0, 40_000.0])
def test_coplanar_phased_pair_members_burn_opposite_ways(phase_offset_m):
    state_i = _circular_state(_RADIUS_M, 0.0, 0.0, 0.0)
    state_j = _circular_state(_RADIUS_M, 0.0, phase_offset_m / _RADIUS_M, 0.0)

    action_i, action_j = _pair_actions(state_i, state_j)

    _assert_opposed(action_i, action_j)
    # the frozen along-track mapping is unchanged: neighbour ahead -> prograde
    expected_i = ManeuverIntent.RAISE_ORBIT if phase_offset_m > 0 else ManeuverIntent.LOWER_ORBIT
    assert action_i == expected_i


@pytest.mark.parametrize(("radial_offset_m", "phase_offset_m"), [(2_000.0, 0.0), (-2_000.0, 0.0), (1_500.0, 300.0)])
def test_radial_offset_pair_members_burn_opposite_ways(radial_offset_m, phase_offset_m):
    state_i = _circular_state(_RADIUS_M, 0.0, 0.0, 0.0)
    state_j = _circular_state(_RADIUS_M + radial_offset_m, 0.0, phase_offset_m / _RADIUS_M, 0.0)

    action_i, action_j = _pair_actions(state_i, state_j)

    _assert_opposed(action_i, action_j)
    # the higher satellite raises, the lower one lowers
    expected_i = ManeuverIntent.LOWER_ORBIT if radial_offset_m > 0 else ManeuverIntent.RAISE_ORBIT
    assert action_i == expected_i


def test_review_probe_observation_pair_burns_opposite_ways():
    """Wave-4 probe p7 (validation seed 114021, pair (4, 6), step 196): R and T share a sign, only N flips."""
    sat4 = _observation(np.array([-24.54, +9.65, -257.65]))
    sat6 = _observation(np.array([-24.47, +9.72, +257.65]))

    _assert_opposed(ManeuverIntent(rule_based_policy("sat_4", sat4)), ManeuverIntent(rule_based_policy("sat_6", sat6)))


@pytest.mark.parametrize(
    ("relative_position", "expected"),
    [
        ((-0.4, 0.1, 0.1), ManeuverIntent.RAISE_ORBIT),  # radial-dominant, neighbour below -> go up
        ((+0.4, 0.1, 0.1), ManeuverIntent.LOWER_ORBIT),  # radial-dominant, neighbour above -> go down
        ((0.1, +0.4, -0.1), ManeuverIntent.RAISE_ORBIT),  # along-track-dominant, neighbour ahead
        ((0.1, -0.4, 0.1), ManeuverIntent.LOWER_ORBIT),  # along-track-dominant, neighbour behind
        ((-0.1, -0.1, +0.4), ManeuverIntent.RAISE_ORBIT),  # cross-track-dominant, N > 0
        ((0.1, 0.1, -0.4), ManeuverIntent.LOWER_ORBIT),  # cross-track-dominant, N < 0
    ],
)
def test_the_direction_follows_the_sign_of_the_dominant_component(relative_position, expected):
    assert rule_based_policy("sat_0", _observation(np.array(relative_position))) == expected


def test_exact_ties_resolve_radial_then_along_track_then_cross_track():
    assert rule_based_policy("sat_0", _observation(np.array([0.3, 0.3, 0.3]))) == ManeuverIntent.LOWER_ORBIT
    assert rule_based_policy("sat_0", _observation(np.array([0.1, -0.3, 0.3]))) == ManeuverIntent.LOWER_ORBIT


def test_a_neighbour_far_from_the_deadband_on_one_axis_is_never_sent_to_the_radial_fallback():
    """Old defect path: a crossing encounter with T ~ 0 must not make BOTH members burn +R."""
    action = rule_based_policy("sat_0", _observation(np.array([0.0, 0.0, 3.0])))

    assert action in _OPPOSED
