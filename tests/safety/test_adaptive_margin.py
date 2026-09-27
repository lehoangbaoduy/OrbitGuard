"""Contract tests for orbit_guard.safety.adaptive_margin (PRD §22, Core Contribution #1).

    d_safe,ij = clip( d_base * [1 + alpha*Sigma_norm,ij + beta*rho_ij + gamma*R_ij], d_min, d_max )
    rho_ij = min(1, count_within(midpoint(i,j), d_density_ref) / rho_ref_count)

MVP defaults (configs/default.yaml `adaptive_margin:` block):
    d_base_m=1000, d_min_m=200, d_max_m=5000, alpha=1.0, beta=0.5, gamma=1.0,
    density_radius_km=50 (-> 50_000 m), density_ref_count=5

Note: `Sigma_norm,ij` itself (how the raw combined uncertainty is normalized)
is not numerically specified by the PRD beyond the symbol -- this module
implements the recommended provisional choice `sigma_rel_m / d_base_m`
(reuses the formula's own existing reference scale, dimensionless, no new
config parameter). See docs/OrbitGuard_progress.md §3 Q6 for the flagged
astro-review item.

Hard rule (§8 checklist items 5-7): d_safe is always pairwise, symmetric,
bounded to [d_min, d_max], and must never feed its own risk calculation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from orbit_guard.safety.adaptive_margin import (
    compute_d_safe,
    compute_local_density,
    compute_sigma_norm,
)

_D_BASE_M = 1000.0
_D_MIN_M = 200.0
_D_MAX_M = 5000.0
_ALPHA = 1.0
_BETA = 0.5
_GAMMA = 1.0
_DENSITY_RADIUS_M = 50_000.0
_DENSITY_REF_COUNT = 5.0


# --- compute_sigma_norm ------------------------------------------------------

def test_compute_sigma_norm_divides_by_reference_scale():
    assert compute_sigma_norm(141.42, d_base_m=1000.0) == pytest.approx(0.14142, rel=1e-4)


def test_compute_sigma_norm_rejects_negative_sigma_rel():
    with pytest.raises(ValueError):
        compute_sigma_norm(-1.0, d_base_m=1000.0)


def test_compute_sigma_norm_rejects_nonpositive_d_base():
    with pytest.raises(ValueError):
        compute_sigma_norm(100.0, d_base_m=0.0)


# --- compute_local_density ---------------------------------------------------

def test_compute_local_density_counts_only_others_within_radius():
    midpoint = np.array([0.0, 0.0, 0.0])
    positions = np.array([
        [0.0, 0.0, 0.0],       # excluded (this is satellite i itself, index 0)
        [10_000.0, 0.0, 0.0],  # 10 km -- within 50 km radius
        [20_000.0, 0.0, 0.0],  # 20 km -- within
        [200_000.0, 0.0, 0.0],  # 200 km -- outside
    ])

    rho = compute_local_density(
        midpoint, positions, exclude_indices={0},
        density_radius_m=_DENSITY_RADIUS_M, density_ref_count=_DENSITY_REF_COUNT,
    )

    assert rho == pytest.approx(2 / 5)  # 2 within radius / ref_count=5


def test_compute_local_density_excludes_both_pair_members():
    midpoint = np.array([0.0, 0.0, 0.0])
    positions = np.array([
        [1_000.0, 0.0, 0.0],  # i
        [-1_000.0, 0.0, 0.0],  # j
        [5_000.0, 0.0, 0.0],  # a genuine third neighbor
    ])

    rho = compute_local_density(
        midpoint, positions, exclude_indices={0, 1},
        density_radius_m=_DENSITY_RADIUS_M, density_ref_count=_DENSITY_REF_COUNT,
    )

    assert rho == pytest.approx(1 / 5)


def test_compute_local_density_saturates_at_one():
    midpoint = np.array([0.0, 0.0, 0.0])
    positions = np.array([[i * 100.0, 0.0, 0.0] for i in range(1, 11)])  # 10 close neighbors

    rho = compute_local_density(
        midpoint, positions, exclude_indices=set(),
        density_radius_m=_DENSITY_RADIUS_M, density_ref_count=_DENSITY_REF_COUNT,
    )

    assert rho == 1.0


def test_compute_local_density_unit_boundary_km_to_m():
    # density_radius_km: 50 in configs/default.yaml means 50_000 m -- a
    # satellite at 49 km must count, one at 51 km must not (strict <).
    midpoint = np.array([0.0, 0.0, 0.0])
    positions = np.array([
        [49_000.0, 0.0, 0.0],
        [51_000.0, 0.0, 0.0],
    ])

    rho = compute_local_density(
        midpoint, positions, exclude_indices=set(),
        density_radius_m=50_000.0, density_ref_count=_DENSITY_REF_COUNT,
    )

    assert rho == pytest.approx(1 / 5)


def test_compute_local_density_exact_boundary_is_excluded():
    midpoint = np.array([0.0, 0.0, 0.0])
    positions = np.array([[50_000.0, 0.0, 0.0]])  # exactly at the radius

    rho = compute_local_density(
        midpoint, positions, exclude_indices=set(),
        density_radius_m=50_000.0, density_ref_count=_DENSITY_REF_COUNT,
    )

    assert rho == 0.0


def test_compute_local_density_midpoint_order_is_symmetric():
    pos_i = np.array([0.0, 0.0, 0.0])
    pos_j = np.array([10_000.0, 0.0, 0.0])
    others = np.array([[5_000.0, 1_000.0, 0.0]])

    midpoint_ij = (pos_i + pos_j) / 2
    midpoint_ji = (pos_j + pos_i) / 2

    rho_ij = compute_local_density(midpoint_ij, others, set(), _DENSITY_RADIUS_M, _DENSITY_REF_COUNT)
    rho_ji = compute_local_density(midpoint_ji, others, set(), _DENSITY_RADIUS_M, _DENSITY_REF_COUNT)

    assert rho_ij == rho_ji


def test_compute_local_density_rejects_invalid_parameters():
    midpoint = np.zeros(3)
    positions = np.zeros((1, 3))
    with pytest.raises(ValueError):
        compute_local_density(midpoint, positions, set(), density_radius_m=0.0, density_ref_count=_DENSITY_REF_COUNT)
    with pytest.raises(ValueError):
        compute_local_density(midpoint, positions, set(), density_radius_m=_DENSITY_RADIUS_M, density_ref_count=0.0)


# --- compute_d_safe -----------------------------------------------------------

def test_compute_d_safe_matches_formula_when_unclipped():
    sigma_norm, rho, risk = 0.5, 0.4, 0.3
    expected = _D_BASE_M * (1 + _ALPHA * sigma_norm + _BETA * rho + _GAMMA * risk)
    assert _D_MIN_M < expected < _D_MAX_M  # sanity: this case shouldn't hit either clip

    d_safe = compute_d_safe(sigma_norm, rho, risk)

    assert d_safe == pytest.approx(expected)


def test_compute_d_safe_default_parameters_match_reference_config():
    d_safe = compute_d_safe(0.5, 0.4, 0.3)
    explicit = compute_d_safe(
        0.5, 0.4, 0.3,
        d_base_m=_D_BASE_M, d_min_m=_D_MIN_M, d_max_m=_D_MAX_M,
        alpha=_ALPHA, beta=_BETA, gamma=_GAMMA,
    )
    assert d_safe == explicit


def test_compute_d_safe_clips_to_d_max_for_extreme_inputs():
    d_safe = compute_d_safe(sigma_norm=10.0, rho=1.0, risk=1.0)
    assert d_safe == _D_MAX_M


def test_compute_d_safe_clips_to_d_min_when_d_base_is_below_it():
    # Only reachable with a non-default config where d_base < d_min -- still
    # must be honored, since clip(...) must bound BOTH sides unconditionally.
    d_safe = compute_d_safe(0.0, 0.0, 0.0, d_base_m=100.0, d_min_m=_D_MIN_M, d_max_m=_D_MAX_M)
    assert d_safe == _D_MIN_M


def test_compute_d_safe_is_always_within_bounds_for_random_inputs():
    rng = np.random.default_rng(123)
    for _ in range(200):
        sigma_norm, rho, risk = rng.uniform(0, 20, size=3)
        d_safe = compute_d_safe(float(sigma_norm), float(rho), float(risk))
        assert _D_MIN_M <= d_safe <= _D_MAX_M


def test_compute_d_safe_is_non_decreasing_in_sigma_norm():
    values = [compute_d_safe(s, 0.2, 0.2) for s in [0.0, 0.5, 1.0, 2.0, 5.0, 20.0]]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_compute_d_safe_is_non_decreasing_in_rho():
    values = [compute_d_safe(0.2, r, 0.2) for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_compute_d_safe_is_non_decreasing_in_risk():
    values = [compute_d_safe(0.2, 0.2, r) for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_compute_d_safe_is_deterministic():
    assert compute_d_safe(0.3, 0.3, 0.3) == compute_d_safe(0.3, 0.3, 0.3)


def test_compute_d_safe_rejects_negative_inputs():
    with pytest.raises(ValueError):
        compute_d_safe(-0.1, 0.2, 0.2)
    with pytest.raises(ValueError):
        compute_d_safe(0.2, -0.1, 0.2)
    with pytest.raises(ValueError):
        compute_d_safe(0.2, 0.2, -0.1)


def test_compute_d_safe_rejects_d_min_greater_than_d_max():
    with pytest.raises(ValueError):
        compute_d_safe(0.2, 0.2, 0.2, d_min_m=5000.0, d_max_m=200.0)


def test_compute_d_safe_signature_has_no_self_referential_input():
    # Processing-order guard (PRD §22, checklist item 6): d_safe must never
    # be an input to its own calculation.
    params = set(inspect.signature(compute_d_safe).parameters)
    forbidden = {"d_safe", "d_safe_m", "margin", "previous_d_safe", "prior_margin"}
    assert not (params & forbidden)
