"""Acceptance Test C (PRD §38): adaptive margin moves in the expected
direction, within bounds, as uncertainty/density/risk inputs change.

Deliberately does not import orbit_guard.conjunction (Part B, handed off to
a teammate and not yet implemented) -- risk is fed in as a plain float, the
same boundary the Phase 2 split established (docs/OrbitGuard_progress.md).
This exercises the real belief pipeline (§14) end to end, not synthetic
scalars, so it also doubles as an integration check across Part A's modules.
"""

from __future__ import annotations

import numpy as np

from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.safety.adaptive_margin import (
    compute_d_safe,
    compute_local_density,
    compute_sigma_norm,
)
from orbit_guard.scenarios.constellation import (
    fit_shell_distribution,
    load_tle_snapshot,
    sample_constellation,
)
from orbit_guard.sensing.belief import (
    GroundTruthState,
    combined_relative_sigma,
    form_belief,
    observe,
)
from orbit_guard.sensing.noise import NoiseModel

_TLE_PATH = "data/TLE_snapshot_53deg_shell.csv"
_D_MIN_M, _D_MAX_M = 200.0, 5000.0


def _capability() -> SatelliteCapability:
    return SatelliteCapability(
        max_delta_v_per_action_mps=0.1,
        remaining_delta_v_mps=2.0,
        maneuver_enabled=True,
        mission_deviation_limit=1.0,
    )


def _constellation(seed: int):
    shell = fit_shell_distribution(load_tle_snapshot(_TLE_PATH))
    return sample_constellation(15, seed=seed, shell=shell, capability_template=_capability())


def _beliefs(satellites, noise_model: NoiseModel, seed: int = 0):
    rng = np.random.default_rng(seed)
    beliefs = []
    for sat in satellites:
        truth = GroundTruthState(state_vector=sat.state.as_vector())
        beliefs.append(form_belief(observe(truth, rng, noise_model), noise_model))
    return beliefs


def test_d_safe_increases_with_uncertainty_holding_density_and_risk_fixed():
    satellites = _constellation(seed=1)
    beliefs_low = _beliefs(satellites, NoiseModel(position_sigma_m=10.0, velocity_sigma_mps=0.01))
    beliefs_high = _beliefs(satellites, NoiseModel(position_sigma_m=500.0, velocity_sigma_mps=0.5))

    sigma_rel_low = combined_relative_sigma(beliefs_low[0], beliefs_low[1])
    sigma_rel_high = combined_relative_sigma(beliefs_high[0], beliefs_high[1])
    assert sigma_rel_high > sigma_rel_low

    positions = np.array([b.state_vector[:3] for b in beliefs_low])
    midpoint = (positions[0] + positions[1]) / 2
    rho = compute_local_density(midpoint, positions, exclude_indices={0, 1})
    risk = 0.4

    d_safe_low = compute_d_safe(compute_sigma_norm(sigma_rel_low), rho, risk)
    d_safe_high = compute_d_safe(compute_sigma_norm(sigma_rel_high), rho, risk)

    assert d_safe_high >= d_safe_low
    assert _D_MIN_M <= d_safe_low <= _D_MAX_M
    assert _D_MIN_M <= d_safe_high <= _D_MAX_M


def test_d_safe_increases_with_local_density_holding_uncertainty_and_risk_fixed():
    # A naturally-sampled shell spans tens of thousands of km, so real
    # neighbors almost never fall within a 50 km density radius -- add
    # synthetic nearby points to exercise the "dense" case directly rather
    # than relying on coincidental natural clustering.
    satellites = _constellation(seed=2)
    beliefs = _beliefs(satellites, NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1))
    positions = np.array([b.state_vector[:3] for b in beliefs])
    midpoint = (positions[0] + positions[1]) / 2

    sparse_positions = positions[:2]
    nearby_neighbors = midpoint + np.array(
        [[1000.0, 0.0, 0.0], [-1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]]
    )
    dense_positions = np.vstack([positions[:2], nearby_neighbors])

    sparse_rho = compute_local_density(midpoint, sparse_positions, exclude_indices={0, 1})
    dense_rho = compute_local_density(midpoint, dense_positions, exclude_indices={0, 1})
    assert dense_rho > sparse_rho

    sigma_norm = compute_sigma_norm(combined_relative_sigma(beliefs[0], beliefs[1]))
    risk = 0.4

    d_safe_sparse = compute_d_safe(sigma_norm, sparse_rho, risk)
    d_safe_dense = compute_d_safe(sigma_norm, dense_rho, risk)

    assert d_safe_dense >= d_safe_sparse
    assert _D_MIN_M <= d_safe_sparse <= _D_MAX_M
    assert _D_MIN_M <= d_safe_dense <= _D_MAX_M


def test_d_safe_increases_with_risk_holding_uncertainty_and_density_fixed():
    satellites = _constellation(seed=3)
    beliefs = _beliefs(satellites, NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1))
    positions = np.array([b.state_vector[:3] for b in beliefs])
    midpoint = (positions[0] + positions[1]) / 2
    rho = compute_local_density(midpoint, positions, exclude_indices={0, 1})
    sigma_norm = compute_sigma_norm(combined_relative_sigma(beliefs[0], beliefs[1]))

    d_safe_low_risk = compute_d_safe(sigma_norm, rho, risk=0.05)
    d_safe_high_risk = compute_d_safe(sigma_norm, rho, risk=0.95)

    assert d_safe_high_risk >= d_safe_low_risk
    assert _D_MIN_M <= d_safe_low_risk <= _D_MAX_M
    assert _D_MIN_M <= d_safe_high_risk <= _D_MAX_M


def test_d_safe_is_symmetric_for_the_pair():
    satellites = _constellation(seed=4)
    beliefs = _beliefs(satellites, NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1))
    positions = np.array([b.state_vector[:3] for b in beliefs])
    midpoint = (positions[0] + positions[1]) / 2
    rho = compute_local_density(midpoint, positions, exclude_indices={0, 1})
    risk = 0.5

    sigma_norm_ij = compute_sigma_norm(combined_relative_sigma(beliefs[0], beliefs[1]))
    sigma_norm_ji = compute_sigma_norm(combined_relative_sigma(beliefs[1], beliefs[0]))
    assert sigma_norm_ij == sigma_norm_ji

    assert compute_d_safe(sigma_norm_ij, rho, risk) == compute_d_safe(sigma_norm_ji, rho, risk)
