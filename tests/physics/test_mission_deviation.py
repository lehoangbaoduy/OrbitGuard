"""Tests for orbit_guard.physics.mission_deviation (PRD §25, §37; Phase 3 Part A).

Mission deviation compares a satellite's actual (possibly maneuvered) state
against a per-episode NO-MANEUVER SHADOW TRAJECTORY -- not a static t=0
reference -- because J2 short-period oscillation alone swings an
unmaneuvered satellite's semi-major axis by ~12,350m and eccentricity by
~0.0012 over one 6h episode (empirically measured, see
docs/OrbitGuard_progress.md Q9), both exceeding the PRD's own tolerance
bands. A static reference would flag an unmaneuvered satellite as having
violated its own mission tolerance, which is nonsensical.

This module is deliberately SOURCE-AGNOSTIC: it takes two plain 6-vectors
(whatever state and shadow-state the caller provides) and returns their
deviation. It does not care whether those vectors come from ground truth or
belief -- Part A's `rl/environment.py` decides that (see
docs/OrbitGuard_progress.md Q14: both the observation's `own_mission_deviation`
and the reward's mission term are computed from BELIEF state, never ground
truth, consistent with PRD §14's leakage-boundary philosophy -- ground
truth is reserved for collision determination and final metrics only).

Avoids `physics.frames.cartesian_to_coe` entirely -- that function raises
ValueError for near-circular orbits (this project's ~0.0002-eccentricity
shell is well within the danger zone once J2 perturbs it) and for the
argument-of-periapsis computation specifically, which mission deviation
never needs. Semi-major axis (vis-viva) and inclination (via the angular
momentum vector) are never singular for this use; the eccentricity-vector
MAGNITUDE (not its direction, which IS singular near-circular) is also
safe. Along-track phase deviation is measured as a literal RTN along-track
(T) distance in meters -- the same metric already used and empirically
verified in `tests/safety/test_maneuver_mapper.py`'s drift test -- not a
raw mean-anomaly angle difference, matching the PRD's own "along-track
phase deviation" wording and its meters-based tolerance band.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.physics.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.frames import coe_to_cartesian
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.mission_deviation import (
    MissionDeviationComponents,
    compute_mission_deviation,
    compute_weighted_mission_deviation,
)

_SEMI_MAJOR_AXIS_TOLERANCE_M = 2000.0
_ECCENTRICITY_TOLERANCE = 0.001
_INCLINATION_TOLERANCE_RAD = np.radians(0.05)
_ALONG_TRACK_TOLERANCE_M = 10_000.0


def _leo_state_vector(true_anomaly_rad: float = 0.0) -> np.ndarray:
    r, v = coe_to_cartesian(
        R_EARTH_EQUATORIAL_M + 463_000.0, 0.0002, np.radians(53.06),
        np.radians(10.0), np.radians(20.0), true_anomaly_rad,
    )
    return np.concatenate([r, v])


# --- Zero-deviation baseline ---------------------------------------------

def test_identical_state_and_shadow_gives_exactly_zero_deviation():
    state = _leo_state_vector()
    components = compute_mission_deviation(state, state)
    assert components.d_a == 0.0
    assert components.d_e == 0.0
    assert components.d_i == 0.0
    assert components.d_m == 0.0


def test_an_unmaneuvered_satellite_propagated_identically_to_its_shadow_stays_zero():
    # The literal shadow-trajectory construction: propagate the SAME initial
    # state twice with identical dynamics -- bitwise identical trajectories,
    # hence exactly zero deviation at every point, not just at t=0.
    state0 = _leo_state_vector()
    trajectory = propagate(state0, 10.0, 360, state_derivative)
    shadow_trajectory = propagate(state0, 10.0, 360, state_derivative)
    components = compute_mission_deviation(trajectory[-1], shadow_trajectory[-1])
    assert components.d_a == 0.0
    assert components.d_e == 0.0
    assert components.d_i == 0.0
    assert components.d_m == 0.0


# --- Semi-major axis deviation, verified independently via vis-viva -------

def test_semi_major_axis_deviation_matches_independent_vis_viva_calculation():
    state0 = _leo_state_vector()
    r0, v0 = state0[:3], state0[3:6]

    # Apply a small prograde (tangential) delta-v directly along the velocity
    # direction -- an independent, from-scratch vis-viva energy computation,
    # not calling any orbit_guard code, to avoid a tautological self-check.
    delta_v_mps = 0.1
    v_hat = v0 / np.linalg.norm(v0)
    v1 = v0 + delta_v_mps * v_hat
    maneuvered_state = np.concatenate([r0, v1])

    r_mag = np.linalg.norm(r0)
    energy0 = np.dot(v0, v0) / 2.0 - MU_EARTH_M3_S2 / r_mag
    energy1 = np.dot(v1, v1) / 2.0 - MU_EARTH_M3_S2 / r_mag
    a0 = -MU_EARTH_M3_S2 / (2.0 * energy0)
    a1 = -MU_EARTH_M3_S2 / (2.0 * energy1)
    expected_d_a = abs(a1 - a0) / _SEMI_MAJOR_AXIS_TOLERANCE_M

    components = compute_mission_deviation(maneuvered_state, state0)
    assert components.d_a == pytest.approx(expected_d_a, rel=1e-9)
    assert components.d_a > 0.0


# --- Along-track deviation, verified via direct RTN projection ------------

def test_along_track_deviation_is_the_rtn_t_projection_over_the_tolerance():
    state0 = _leo_state_vector()
    r0, v0 = state0[:3], state0[3:6]
    # Displace the "actual" position by a known 500m shift purely along the
    # shadow's own along-track (T) direction, verified independently.
    r_hat = r0 / np.linalg.norm(r0)
    h_vec = np.cross(r0, v0)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    shifted_r = r0 + 500.0 * t_hat
    actual_state = np.concatenate([shifted_r, v0])

    components = compute_mission_deviation(actual_state, state0)
    assert components.d_m == pytest.approx(500.0 / _ALONG_TRACK_TOLERANCE_M, rel=1e-9)


def test_pure_radial_displacement_contributes_zero_along_track_deviation():
    state0 = _leo_state_vector()
    r0, v0 = state0[:3], state0[3:6]
    r_hat = r0 / np.linalg.norm(r0)
    shifted_r = r0 + 300.0 * r_hat
    actual_state = np.concatenate([shifted_r, v0])

    components = compute_mission_deviation(actual_state, state0)
    assert components.d_m == pytest.approx(0.0, abs=1e-6)


# --- Eccentricity and inclination deviations ------------------------------

def test_eccentricity_deviation_is_the_absolute_difference_in_eccentricity_vector_magnitude():
    state0 = _leo_state_vector()
    r0, v0 = state0[:3], state0[3:6]

    def eccentricity_magnitude(r, v):
        r_norm, v_norm = np.linalg.norm(r), np.linalg.norm(v)
        e_vec = ((v_norm**2 - MU_EARTH_M3_S2 / r_norm) * r - np.dot(r, v) * v) / MU_EARTH_M3_S2
        return np.linalg.norm(e_vec)

    delta_v_mps = 0.1
    v_hat = v0 / np.linalg.norm(v0)
    v1 = v0 + delta_v_mps * v_hat
    maneuvered_state = np.concatenate([r0, v1])

    expected_d_e = abs(eccentricity_magnitude(r0, v1) - eccentricity_magnitude(r0, v0)) / _ECCENTRICITY_TOLERANCE
    components = compute_mission_deviation(maneuvered_state, state0)
    assert components.d_e == pytest.approx(expected_d_e, rel=1e-9)


def test_inclination_deviation_responds_to_a_cross_track_velocity_change():
    state0 = _leo_state_vector()
    r0, v0 = state0[:3], state0[3:6]
    h_vec = np.cross(r0, v0)
    n_hat = h_vec / np.linalg.norm(h_vec)
    # A small normal (cross-track) delta-v rotates the orbital plane,
    # changing inclination -- unlike a pure tangential or radial burn.
    v1 = v0 + 0.1 * n_hat
    maneuvered_state = np.concatenate([r0, v1])

    components = compute_mission_deviation(maneuvered_state, state0)
    assert components.d_i > 0.0


def test_pure_tangential_burn_does_not_change_inclination():
    state0 = _leo_state_vector()
    r0, v0 = state0[:3], state0[3:6]
    v_hat = v0 / np.linalg.norm(v0)
    v1 = v0 + 0.1 * v_hat
    maneuvered_state = np.concatenate([r0, v1])

    components = compute_mission_deviation(maneuvered_state, state0)
    assert components.d_i == pytest.approx(0.0, abs=1e-9)


# --- Validation ------------------------------------------------------------

@pytest.mark.parametrize("bad_kwargs", [
    {"semi_major_axis_tolerance_m": 0.0},
    {"semi_major_axis_tolerance_m": -1.0},
    {"eccentricity_tolerance": 0.0},
    {"inclination_tolerance_rad": -0.1},
    {"along_track_tolerance_m": 0.0},
])
def test_rejects_non_positive_tolerances(bad_kwargs):
    state0 = _leo_state_vector()
    with pytest.raises(ValueError):
        compute_mission_deviation(state0, state0, **bad_kwargs)


def test_is_deterministic():
    state0 = _leo_state_vector()
    state1 = _leo_state_vector(true_anomaly_rad=np.radians(30.0))
    c1 = compute_mission_deviation(state1, state0)
    c2 = compute_mission_deviation(state1, state0)
    assert c1 == c2


# --- Weighted scalar aggregate (used by both rl/observation.py and
#     rl/reward.py -- see docs/OrbitGuard_progress.md Q12/Q14) ------------

def test_weighted_mission_deviation_uses_equal_default_weights():
    components = MissionDeviationComponents(d_a=1.0, d_e=0.0, d_i=0.0, d_m=0.0)
    assert compute_weighted_mission_deviation(components) == pytest.approx(0.25)


def test_weighted_mission_deviation_matches_manual_weighted_sum():
    components = MissionDeviationComponents(d_a=0.4, d_e=0.2, d_i=0.1, d_m=0.3)
    weighted = compute_weighted_mission_deviation(
        components, q_a=0.1, q_e=0.2, q_i=0.3, q_m=0.4
    )
    expected = 0.1 * 0.4 + 0.2 * 0.2 + 0.3 * 0.1 + 0.4 * 0.3
    assert weighted == pytest.approx(expected)


def test_weighted_mission_deviation_of_all_zeros_is_zero():
    components = MissionDeviationComponents(d_a=0.0, d_e=0.0, d_i=0.0, d_m=0.0)
    assert compute_weighted_mission_deviation(components) == pytest.approx(0.0)
