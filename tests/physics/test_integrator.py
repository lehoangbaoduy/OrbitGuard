import numpy as np
import pytest

from orbit_guard.physics.constants import MU_EARTH_M3_S2
from orbit_guard.physics.dynamics import state_derivative, two_body_state_derivative
from orbit_guard.physics.integrator import propagate, rk4_step


def _circular_orbit_state(radius_m: float) -> np.ndarray:
    """A circular orbit in the x-y plane: v = sqrt(mu/r), tangential."""
    speed = np.sqrt(MU_EARTH_M3_S2 / radius_m)
    return np.array([radius_m, 0.0, 0.0, 0.0, speed, 0.0])


def _analytic_circular_position(radius_m: float, t_s: float) -> np.ndarray:
    """Closed-form position of the same circular orbit at time t (independent of RK4)."""
    angular_rate = np.sqrt(MU_EARTH_M3_S2 / radius_m**3)
    theta = angular_rate * t_s
    return np.array([radius_m * np.cos(theta), radius_m * np.sin(theta), 0.0])


def test_rk4_step_raises_on_non_finite_result():
    bad_state = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises((ValueError, FloatingPointError)):
        rk4_step(bad_state, dt=10.0, derivative_fn=two_body_state_derivative)


def test_rk4_step_raises_when_update_itself_becomes_non_finite():
    # A finite input state whose derivative_fn returns a non-finite value:
    # exercises rk4_step's own post-update finiteness guard, distinct from a
    # derivative_fn that rejects a bad input up front.
    def _diverging_derivative(_state_vector: np.ndarray) -> np.ndarray:
        return np.array([np.inf, 0.0, 0.0, 0.0, 0.0, 0.0])

    finite_state = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7500.0, 0.0])
    with pytest.raises(ValueError):
        rk4_step(finite_state, dt=10.0, derivative_fn=_diverging_derivative)


def test_rk4_circular_orbit_preserves_radius_over_one_period():
    radius_m = 6_878_000.0  # ~500 km altitude
    state = _circular_orbit_state(radius_m)
    period_s = 2 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)
    dt = 10.0
    n_steps = int(period_s // dt)

    trajectory = propagate(state, dt, n_steps, two_body_state_derivative)
    radii = np.linalg.norm(trajectory[:, 0:3], axis=1)

    # A circular orbit should stay circular: radius should not drift by more
    # than a small fraction of its starting value over one full period.
    assert np.max(np.abs(radii - radius_m)) / radius_m < 1e-6


def test_rk4_circular_orbit_matches_analytic_solution_over_one_hour():
    radius_m = 6_878_000.0
    state = _circular_orbit_state(radius_m)
    dt = 10.0
    total_time_s = 3600.0  # divides evenly by dt, avoiding rounding artifacts
    n_steps = int(total_time_s / dt)

    trajectory = propagate(state, dt, n_steps, two_body_state_derivative)
    final_position = trajectory[-1, 0:3]
    expected_position = _analytic_circular_position(radius_m, total_time_s)

    assert final_position == pytest.approx(expected_position, abs=1.0)  # within 1 m


def test_rk4_convergence_error_shrinks_as_timestep_shrinks():
    radius_m = 6_878_000.0
    state = _circular_orbit_state(radius_m)
    total_time_s = 2400.0  # divides evenly by every dt candidate below

    errors = []
    for dt in (40.0, 20.0, 10.0):
        n_steps = int(total_time_s / dt)
        trajectory = propagate(state, dt, n_steps, two_body_state_derivative)
        final_position = trajectory[-1, 0:3]
        expected_position = _analytic_circular_position(radius_m, total_time_s)
        errors.append(np.linalg.norm(final_position - expected_position))

    # RK4 is 4th-order: halving dt should shrink error by roughly 16x per halving.
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < errors[0] / 50.0


def test_propagate_output_shape_matches_n_steps_plus_initial():
    state = _circular_orbit_state(6_878_000.0)
    trajectory = propagate(state, dt=10.0, n_steps=5, derivative_fn=two_body_state_derivative)
    assert trajectory.shape == (6, 6)
    assert trajectory[0] == pytest.approx(state)


def test_propagate_with_full_state_derivative_stays_finite_over_several_orbits():
    radius_m = 6_878_000.0
    state = _circular_orbit_state(radius_m)
    period_s = 2 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)
    dt = 10.0
    n_steps = round(3 * period_s / dt)

    trajectory = propagate(state, dt, n_steps, state_derivative)
    assert np.all(np.isfinite(trajectory))
