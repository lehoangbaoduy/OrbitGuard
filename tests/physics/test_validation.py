"""PRD §12 physics validation suite -- Phase 1 exit criteria.

Five required checks, each a dedicated test (not folded together), plus the
"fail loudly" requirement:
1. Two-body circular-orbit sanity check.
2. Energy / angular-momentum conservation in two-body-only mode.
3. J2 perturbation sanity check (nodal-regression direction/rate).
4. Comparison against a trusted reference implementation.
5. Convergence check as the numerical timestep is reduced.
"""

import numpy as np
import pytest
from skyfield.keplerlib import propagate as skyfield_two_body_propagate

from orbit_guard.physics.constants import J2, MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from orbit_guard.physics.dynamics import state_derivative, two_body_state_derivative
from orbit_guard.physics.frames import cartesian_to_coe, coe_to_cartesian
from orbit_guard.physics.integrator import propagate

# Nominal N=15 benchmark shell parameters (PRD §16/§39): a~6838.6km, i~53.06deg,
# near-circular. e is kept small-but-nonzero so cartesian_to_coe's argument-of-
# periapsis/RAAN measurement stays well-defined (see frames.py's documented
# near-circular scope limit).
_SHELL_SEMI_MAJOR_AXIS_M = 6_838_588.0
_SHELL_ECCENTRICITY = 0.0005
_SHELL_INCLINATION_RAD = np.radians(53.06)


def _shell_state(raan_rad: float = 0.0, arg_periapsis_rad: float = 0.0, mean_anomaly_rad: float = 0.0):
    r, v = coe_to_cartesian(
        _SHELL_SEMI_MAJOR_AXIS_M,
        _SHELL_ECCENTRICITY,
        _SHELL_INCLINATION_RAD,
        raan_rad,
        arg_periapsis_rad,
        mean_anomaly_rad,
    )
    return np.concatenate([r, v])


# --- 1. Two-body circular-orbit sanity check --------------------------------


def test_two_body_circular_orbit_stays_circular_and_finite():
    radius_m = 6_878_000.0  # exact circle: e=0 is fine here, no cartesian_to_coe call
    r0, v0 = coe_to_cartesian(radius_m, 0.0, _SHELL_INCLINATION_RAD, 0.0, 0.0, 0.0)
    state0 = np.concatenate([r0, v0])

    period_s = 2 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)
    dt = 10.0
    n_steps = int(period_s // dt)

    trajectory = propagate(state0, dt, n_steps, two_body_state_derivative)
    assert np.all(np.isfinite(trajectory))

    radii = np.linalg.norm(trajectory[:, 0:3], axis=1)
    assert np.max(np.abs(radii - radius_m)) / radius_m < 1e-6


# --- 2. Energy / angular-momentum conservation (two-body-only mode) --------


def test_two_body_only_specific_energy_is_conserved_over_multiple_orbits():
    state0 = _shell_state(raan_rad=np.radians(30), arg_periapsis_rad=np.radians(15))
    period_s = 2 * np.pi * np.sqrt(_SHELL_SEMI_MAJOR_AXIS_M**3 / MU_EARTH_M3_S2)
    dt = 10.0
    n_steps = int(5 * period_s / dt)

    trajectory = propagate(state0, dt, n_steps, two_body_state_derivative)
    r = np.linalg.norm(trajectory[:, 0:3], axis=1)
    v = np.linalg.norm(trajectory[:, 3:6], axis=1)
    specific_energy = v**2 / 2.0 - MU_EARTH_M3_S2 / r

    relative_drift = (np.max(specific_energy) - np.min(specific_energy)) / abs(specific_energy[0])
    assert relative_drift < 1e-8


def test_two_body_only_angular_momentum_is_conserved_over_multiple_orbits():
    state0 = _shell_state(raan_rad=np.radians(30), arg_periapsis_rad=np.radians(15))
    period_s = 2 * np.pi * np.sqrt(_SHELL_SEMI_MAJOR_AXIS_M**3 / MU_EARTH_M3_S2)
    dt = 10.0
    n_steps = int(5 * period_s / dt)

    trajectory = propagate(state0, dt, n_steps, two_body_state_derivative)
    h_vectors = np.cross(trajectory[:, 0:3], trajectory[:, 3:6])
    h_magnitudes = np.linalg.norm(h_vectors, axis=1)

    relative_drift = (np.max(h_magnitudes) - np.min(h_magnitudes)) / h_magnitudes[0]
    assert relative_drift < 1e-8

    # Direction (orbit-plane orientation) must also stay fixed in two-body motion.
    h_unit_initial = h_vectors[0] / h_magnitudes[0]
    h_unit_final = h_vectors[-1] / h_magnitudes[-1]
    assert h_unit_final == pytest.approx(h_unit_initial, abs=1e-9)


# --- 3. J2 nodal-regression sanity check ------------------------------------


def test_j2_nodal_regression_matches_analytic_secular_rate_and_direction():
    raan0 = np.radians(45.0)
    state0 = _shell_state(raan_rad=raan0, arg_periapsis_rad=np.radians(30), mean_anomaly_rad=np.radians(10))

    dt = 10.0
    propagation_days = 10.0
    n_steps = int(propagation_days * 86400.0 / dt)

    trajectory = propagate(state0, dt, n_steps, state_derivative)  # full two-body + J2
    final_state = trajectory[-1]
    _, _, _, raan_final, _, _ = cartesian_to_coe(final_state[0:3], final_state[3:6])

    # RAAN is returned mod 2*pi (frames.cartesian_to_coe); unwrap the shortest
    # angular difference before computing a rate, since ~47deg of expected
    # drift here crosses the 0/2*pi boundary starting from raan0=45deg.
    raan_delta = ((raan_final - raan0 + np.pi) % (2 * np.pi)) - np.pi
    observed_rate_deg_per_day = np.degrees(raan_delta) / propagation_days

    # Independent analytic secular nodal-regression rate (Vallado):
    # dOmega/dt = -1.5 * n * J2 * (Re/p)^2 * cos(i)
    mean_motion_rad_s = np.sqrt(MU_EARTH_M3_S2 / _SHELL_SEMI_MAJOR_AXIS_M**3)
    p = _SHELL_SEMI_MAJOR_AXIS_M * (1 - _SHELL_ECCENTRICITY**2)
    analytic_rate_rad_s = (
        -1.5 * mean_motion_rad_s * J2 * (R_EARTH_EQUATORIAL_M / p) ** 2 * np.cos(_SHELL_INCLINATION_RAD)
    )
    analytic_rate_deg_per_day = np.degrees(analytic_rate_rad_s) * 86400.0

    assert analytic_rate_deg_per_day < 0  # westward/regressing at this inclination (<90deg)
    assert observed_rate_deg_per_day == pytest.approx(analytic_rate_deg_per_day, rel=0.02)


# --- 4. Comparison against a trusted reference implementation --------------


def test_two_body_propagation_matches_skyfield_reference_for_elliptical_orbit():
    # Skyfield's keplerlib.propagate is a SPICE-toolkit-based (prop2b.f) universal-
    # variable two-body propagator -- an independent, trusted implementation.
    state0 = _shell_state(raan_rad=np.radians(10), arg_periapsis_rad=np.radians(20), mean_anomaly_rad=np.radians(5))
    r0, v0 = state0[0:3], state0[3:6]

    dt = 10.0
    total_time_s = 6000.0
    n_steps = int(total_time_s / dt)

    ours = propagate(state0, dt, n_steps, two_body_state_derivative)[-1]

    reference_r, reference_v = skyfield_two_body_propagate(
        r0, v0, 0.0, np.array(total_time_s), MU_EARTH_M3_S2
    )

    assert ours[0:3] == pytest.approx(reference_r, abs=10.0)  # within 10 m
    assert ours[3:6] == pytest.approx(reference_v, abs=0.01)  # within 1 cm/s


# --- 5. Convergence check as the timestep shrinks (full two-body + J2) -----


def test_full_dynamics_convergence_error_shrinks_as_timestep_shrinks():
    state0 = _shell_state(raan_rad=np.radians(10), arg_periapsis_rad=np.radians(20), mean_anomaly_rad=np.radians(5))
    total_time_s = 6000.0

    fine_dt = 1.0
    reference_final = propagate(state0, fine_dt, int(total_time_s / fine_dt), state_derivative)[-1]

    errors = []
    for dt in (40.0, 20.0, 10.0):
        n_steps = int(total_time_s / dt)
        final = propagate(state0, dt, n_steps, state_derivative)[-1]
        errors.append(np.linalg.norm(final[0:3] - reference_final[0:3]))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < errors[0] / 10.0


# --- Fail loudly on non-finite/physically-invalid state ---------------------


def test_propagator_fails_loudly_rather_than_silently_propagating_invalid_state():
    degenerate_state = np.zeros(6)  # zero position: undefined acceleration
    with pytest.raises(ValueError):
        propagate(degenerate_state, dt=10.0, n_steps=5, derivative_fn=state_derivative)
