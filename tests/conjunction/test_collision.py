"""Contract tests for orbit_guard.conjunction.collision (PRD §18).

Physical collision: ||r_i - r_j|| < R_i + R_j (STRICT inequality), determined
from ground truth, checked at every provided trajectory sample (which must be
physics_dt resolution, i.e. every 10s at minimum -- the caller is responsible
for passing a trajectory array at that resolution; this module does not
resample).
"""

import numpy as np
import pytest
from orbit_guard.conjunction.collision import CollisionResult, detect_collision


def _linear_trajectory(a0: np.ndarray, v: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    t = np.arange(n_steps + 1) * dt
    positions = a0[None, :] + np.outer(t, v)
    velocities = np.tile(v, (n_steps + 1, 1))
    return np.concatenate([positions, velocities], axis=1)


def test_detect_collision_returns_false_when_never_within_combined_radius():
    dt = 10.0
    traj_i = _linear_trajectory(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 20)
    traj_j = _linear_trajectory(np.array([0.0, 1000.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 20)

    result = detect_collision(traj_i, traj_j, dt, hard_body_radius_i_m=5.0, hard_body_radius_j_m=5.0)

    assert isinstance(result, CollisionResult)
    assert result.collided is False
    assert result.collision_time_s is None
    assert result.min_separation_m == pytest.approx(1000.0, abs=1e-6)


def test_detect_collision_true_when_trajectories_cross_within_combined_radius():
    # i stationary at origin; j starts 100m away closing at 20 m/s -- passes
    # through the 10m combined hard-body radius around t=4.5-5.0s of closing.
    dt = 1.0
    traj_i = _linear_trajectory(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 10)
    traj_j = _linear_trajectory(np.array([100.0, 0.0, 0.0]), np.array([-20.0, 0.0, 0.0]), dt, 10)

    result = detect_collision(traj_i, traj_j, dt, hard_body_radius_i_m=5.0, hard_body_radius_j_m=5.0)

    assert result.collided is True
    # j's position at sample t: 100 - 20*t. |100-20t| < 10  =>  4.5 < t < 5.5.
    # First sample (integer t) satisfying this is t=5.
    assert result.collision_time_s == pytest.approx(5.0, abs=dt)
    assert result.min_separation_m < 10.0


def test_detect_collision_reports_first_collision_time_not_last():
    # Construct a trajectory pair that is within the combined radius for
    # several consecutive samples; collision_time_s must be the FIRST one.
    dt = 1.0
    traj_i = _linear_trajectory(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 10)
    # j sits inside the combined radius for samples t=3..6, outside otherwise.
    positions = np.array([[20.0, 0, 0], [20, 0, 0], [20, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0], [3, 0, 0], [20, 0, 0], [20, 0, 0], [20, 0, 0], [20, 0, 0]])
    velocities = np.zeros_like(positions)
    traj_j = np.concatenate([positions, velocities], axis=1)

    result = detect_collision(traj_i, traj_j, dt, hard_body_radius_i_m=5.0, hard_body_radius_j_m=5.0)

    assert result.collided is True
    assert result.collision_time_s == pytest.approx(3.0, abs=1e-9)


def test_detect_collision_exact_boundary_separation_is_not_a_collision():
    # PRD §18 uses strict inequality: separation exactly equal to the
    # combined hard-body radius must NOT count as a collision.
    dt = 10.0
    traj_i = _linear_trajectory(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 5)
    traj_j = _linear_trajectory(np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), dt, 5)  # exactly 10m away

    result = detect_collision(traj_i, traj_j, dt, hard_body_radius_i_m=5.0, hard_body_radius_j_m=5.0)

    assert result.collided is False


def test_detect_collision_rejects_mismatched_trajectory_shapes():
    traj_i = _linear_trajectory(np.zeros(3), np.zeros(3), 1.0, 20)
    traj_j = _linear_trajectory(np.zeros(3), np.zeros(3), 1.0, 10)

    with pytest.raises(ValueError):
        detect_collision(traj_i, traj_j, dt=1.0, hard_body_radius_i_m=5.0, hard_body_radius_j_m=5.0)


def test_detect_collision_agrees_with_acceptance_test_a_benign_case():
    # Integration smoke test: real Phase 1 propagation of two independently
    # sampled, well-separated satellites must never collide (this mirrors
    # tests/acceptance/test_acceptance_a_benign_orbit.py's own inline check --
    # this module should agree with it once it exists).
    from pathlib import Path

    from orbit_guard.physics.dynamics import state_derivative
    from orbit_guard.physics.integrator import propagate
    from orbit_guard.physics.state import SatelliteCapability
    from orbit_guard.scenarios.constellation import (
        fit_shell_distribution,
        load_tle_snapshot,
        sample_constellation,
    )

    tle_path = Path(__file__).resolve().parents[2] / "data" / "TLE_snapshot_53deg_shell.csv"
    shell = fit_shell_distribution(load_tle_snapshot(tle_path))
    cap = SatelliteCapability(
        max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=2.0,
        maneuver_enabled=True, mission_deviation_limit=1.0,
    )
    satellites = sample_constellation(15, seed=20260922, shell=shell, capability_template=cap)

    dt = 10.0
    n_steps = 100
    trajectories = [
        propagate(sat.state.as_vector(), dt, n_steps, state_derivative) for sat in satellites
    ]

    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            result = detect_collision(
                trajectories[i], trajectories[j], dt, hard_body_radius_i_m=5.0, hard_body_radius_j_m=5.0
            )
            assert result.collided is False
