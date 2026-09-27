"""Contract tests for orbit_guard.conjunction.tca (PRD §19).

    t_TCA = argmin_t  ||r_i(t) - r_j(t)||         (over the lookahead window)
    d_min = ||r_i(t_TCA) - r_j(t_TCA)||

Trajectories are (n_steps+1, 6) arrays in the same [x,y,z,vx,vy,vz] layout
that orbit_guard.physics.integrator.propagate() already produces, so this
module must consume that format directly -- do not invent a new one.
"""

import numpy as np
import pytest
from orbit_guard.conjunction.tca import compute_tca


def _linear_trajectory(a0: np.ndarray, v: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Straight-line constant-velocity motion, sampled like propagate()'s output."""
    t = np.arange(n_steps + 1) * dt
    positions = a0[None, :] + np.outer(t, v)
    velocities = np.tile(v, (n_steps + 1, 1))
    return np.concatenate([positions, velocities], axis=1)


def test_compute_tca_matches_closed_form_linear_minimum_at_exact_sample():
    # Satellite i moves in +x at 10 m/s from (-1000, 50, 0); satellite j is
    # stationary at the origin. Closed-form minimum (calculus on
    # |Δa + Δv·t|^2) occurs at t* = -(Δa·Δv)/|Δv|^2 = 100 s, d_min = 50 m,
    # and t*=100 lands exactly on an integer-second sample -- no
    # interpolation is needed for this case to pass.
    dt = 1.0
    traj_i = _linear_trajectory(np.array([-1000.0, 50.0, 0.0]), np.array([10.0, 0.0, 0.0]), dt, 200)
    traj_j = _linear_trajectory(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 200)

    t_tca, d_min = compute_tca(traj_i, traj_j, dt)

    assert t_tca == pytest.approx(100.0, abs=dt)
    assert d_min == pytest.approx(50.0, abs=1.0)


def test_compute_tca_minimum_between_samples_is_found_within_one_timestep():
    # Same setup, but shifted so the true analytic minimum (t*=100.4s) falls
    # strictly between two 1-second samples. A pure discrete-argmin
    # implementation and an interpolating one should both land within one
    # dt of the true minimum and within a small margin of the true d_min.
    dt = 1.0
    a0_i = np.array([-1004.0, 50.0, 0.0])
    v_i = np.array([10.0, 0.0, 0.0])
    traj_i = _linear_trajectory(a0_i, v_i, dt, 200)
    traj_j = _linear_trajectory(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 200)

    true_t_star = 100.4
    true_d_min = 50.0

    t_tca, d_min = compute_tca(traj_i, traj_j, dt)

    assert t_tca == pytest.approx(true_t_star, abs=dt)
    assert d_min == pytest.approx(true_d_min, abs=1.0)


def test_compute_tca_minimum_at_window_boundary():
    # Two satellites moving apart from the very first sample: the minimum
    # separation is at t=0, not somewhere in the interior.
    dt = 10.0
    traj_i = _linear_trajectory(np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), dt, 50)
    traj_j = _linear_trajectory(np.array([0.0, 200.0, 0.0]), np.array([-5.0, 0.0, 0.0]), dt, 50)

    t_tca, d_min = compute_tca(traj_i, traj_j, dt)

    assert t_tca == pytest.approx(0.0, abs=dt)
    assert d_min == pytest.approx(200.0, abs=1.0)


def test_compute_tca_rejects_mismatched_trajectory_shapes():
    traj_i = _linear_trajectory(np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0, 100)
    traj_j = _linear_trajectory(np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0, 50)  # different length

    with pytest.raises(ValueError):
        compute_tca(traj_i, traj_j, dt=1.0)


def test_compute_tca_smoke_test_with_real_propagated_orbits():
    # Integration smoke test: two real, physically propagated near-identical
    # orbits (small phase offset) from Phase 1's actual propagator -- not a
    # synthetic straight line. This must run without error against real
    # orbit_guard.physics output and return sane, finite values.
    from orbit_guard.physics.dynamics import state_derivative
    from orbit_guard.physics.frames import coe_to_cartesian
    from orbit_guard.physics.integrator import propagate

    a = 6_838_588.0
    e = 0.0005
    i = np.radians(53.06)
    raan = np.radians(10.0)
    argp = np.radians(20.0)

    r1, v1 = coe_to_cartesian(a, e, i, raan, argp, np.radians(0.0))
    r2, v2 = coe_to_cartesian(a, e, i, raan, argp, np.radians(0.5))  # tiny phase offset

    dt = 10.0
    n_steps = 180  # 1800 s lookahead, matching PRD §17's default
    traj_i = propagate(np.concatenate([r1, v1]), dt, n_steps, state_derivative)
    traj_j = propagate(np.concatenate([r2, v2]), dt, n_steps, state_derivative)

    t_tca, d_min = compute_tca(traj_i, traj_j, dt)

    assert np.isfinite(t_tca) and np.isfinite(d_min)
    assert 0.0 <= t_tca <= n_steps * dt
    assert d_min >= 0.0
