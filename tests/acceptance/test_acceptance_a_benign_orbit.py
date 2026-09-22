"""Acceptance Test A (PRD §38): a benign, no-conjunction episode never
spontaneously collides under the No-Maneuver baseline.

This is the Phase 1 exit gate (PRD §44). Collision detection itself is
formally Phase 2 scope (conjunction/collision.py, with sub-step refinement
and edge cases); here the literal PRD §18 distance formula is applied
directly and minimally, only to make this specific acceptance test
executable at Phase 1 exit as the PRD requires.
"""

from pathlib import Path

import numpy as np

from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.scenarios.constellation import (
    fit_shell_distribution,
    load_tle_snapshot,
    sample_constellation,
)

_TLE_PATH = Path(__file__).resolve().parents[2] / "data" / "TLE_snapshot_53deg_shell.csv"

# PRD §39 reference config defaults.
_PHYSICS_DT_S = 10.0
_EPISODE_HORIZON_S = 21600.0
_HARD_BODY_RADIUS_M = 5.0
_N_SATELLITES = 15


def test_acceptance_a_benign_orbit_no_maneuver_never_collides():
    tle_data = load_tle_snapshot(_TLE_PATH)
    shell = fit_shell_distribution(tle_data)
    capability = SatelliteCapability(
        max_delta_v_per_action_mps=0.1,
        remaining_delta_v_mps=2.0,
        maneuver_enabled=True,
        mission_deviation_limit=1.0,
    )

    # No conjunction injected: independent random sampling from the shell
    # distribution only (conjunction injection is Phase 2, §17).
    satellites = sample_constellation(
        _N_SATELLITES, seed=20260922, shell=shell, capability_template=capability
    )

    n_steps = int(_EPISODE_HORIZON_S / _PHYSICS_DT_S)
    positions = np.empty((n_steps + 1, _N_SATELLITES, 3))
    for idx, sat in enumerate(satellites):
        # No-Maneuver baseline: zero actions applied, full two-body + J2 dynamics.
        trajectory = propagate(sat.state.as_vector(), _PHYSICS_DT_S, n_steps, state_derivative)
        positions[:, idx, :] = trajectory[:, 0:3]

    assert np.all(np.isfinite(positions)), "propagator produced a non-finite state"

    combined_hard_body_radius_m = 2 * _HARD_BODY_RADIUS_M
    min_separation_m = np.inf
    for i in range(_N_SATELLITES):
        for j in range(i + 1, _N_SATELLITES):
            separation = np.linalg.norm(positions[:, i, :] - positions[:, j, :], axis=1)
            min_separation_m = min(min_separation_m, float(np.min(separation)))

    assert min_separation_m > combined_hard_body_radius_m, (
        f"spontaneous collision in a benign, no-conjunction episode: "
        f"min separation {min_separation_m:.1f} m <= combined hard-body radius "
        f"{combined_hard_body_radius_m:.1f} m"
    )
