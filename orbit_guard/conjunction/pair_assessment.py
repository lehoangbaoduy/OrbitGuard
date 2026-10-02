"""Belief-based assessment of one satellite pair or of every candidate pair (PRD §19-22).

Shared by the RL environment (observations, rewards) and the safety shield
(candidate-maneuver consequence prediction), so both score a conjunction with
literally the same geometry -> risk -> adaptive-margin pipeline.

Entirely belief-based: `predict_belief` raises `TypeError` for anything but an
`AgentBelief`, so a ground-truth leak fails loudly here.

The safety margin is injected as a `MarginFn(sigma_rel_m, rho, risk) -> d_safe_m`
so the Fixed-Margin baseline (PRD §31) is a swap of one callable, not a code
fork. `make_adaptive_margin` builds the PRD §22 margin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from orbit_guard.conjunction.risk import risk_proxy as compute_risk_proxy
from orbit_guard.conjunction.screening import generate_candidate_pairs
from orbit_guard.conjunction.tca import compute_tca
from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.safety.adaptive_margin import (
    MarginFn,
    compute_local_density,
    make_adaptive_margin,
)
from orbit_guard.sensing.belief import (
    AgentBelief,
    PredictedState,
    combined_relative_sigma,
    predict_belief,
)

DEFAULT_DENSITY_RADIUS_M = 50_000.0
DEFAULT_DENSITY_REF_COUNT = 5.0


@dataclass(frozen=True)
class PairAssessment:
    """One candidate pair's belief-predicted conjunction assessment."""

    i: int
    j: int
    d_min_m: float
    risk_proxy: float
    sigma_rel_m: float
    d_safe_m: float


def prediction_steps(lookahead_s: float, physics_dt_s: float) -> int:
    return max(1, round(lookahead_s / physics_dt_s))


def predict_all(beliefs: Sequence[AgentBelief], lookahead_s: float, physics_dt_s: float) -> list[PredictedState]:
    """Propagate every belief `lookahead_s` forward (no maneuvers)."""
    n_steps = prediction_steps(lookahead_s, physics_dt_s)
    return [predict_belief(b, physics_dt_s, n_steps, state_derivative) for b in beliefs]


def pair_density(
    i: int,
    j: int,
    positions_now_m: np.ndarray,
    density_radius_m: float = DEFAULT_DENSITY_RADIUS_M,
    density_ref_count: float = DEFAULT_DENSITY_REF_COUNT,
) -> float:
    """rho_ij around the pair's current midpoint (belief positions, PRD §22)."""
    midpoint = (positions_now_m[i] + positions_now_m[j]) / 2.0
    return compute_local_density(
        midpoint, positions_now_m, exclude_indices={i, j},
        density_radius_m=density_radius_m, density_ref_count=density_ref_count,
    )


def assess_pair(
    i: int,
    j: int,
    trajectory_i: np.ndarray,
    trajectory_j: np.ndarray,
    physics_dt_s: float,
    sigma_rel_m: float,
    rho: float,
    margin_fn: MarginFn,
) -> PairAssessment:
    """Predicted TCA/d_min -> risk proxy -> margin for two (possibly maneuvered) trajectories."""
    _t_tca_s, d_min_m = compute_tca(trajectory_i, trajectory_j, physics_dt_s)
    risk = compute_risk_proxy(d_min_m, sigma_rel_m)
    d_safe_m = margin_fn(sigma_rel_m, rho, risk)
    return PairAssessment(i=i, j=j, d_min_m=d_min_m, risk_proxy=risk, sigma_rel_m=sigma_rel_m, d_safe_m=d_safe_m)


def assess_all_pairs(
    beliefs: Sequence[AgentBelief],
    lookahead_s: float,
    physics_dt_s: float,
    density_radius_m: float = DEFAULT_DENSITY_RADIUS_M,
    density_ref_count: float = DEFAULT_DENSITY_REF_COUNT,
    d_base_m: float = 1000.0,
    d_min_m: float = 200.0,
    d_max_m: float = 5000.0,
    *,
    margin_fn: MarginFn | None = None,
    predicted: Sequence[PredictedState] | None = None,
) -> list[PairAssessment]:
    """Assess every candidate pair; `predicted` reuses an earlier `predict_all`, `margin_fn` overrides the adaptive default."""
    if predicted is None:
        predicted = predict_all(beliefs, lookahead_s, physics_dt_s)
    if margin_fn is None:
        margin_fn = make_adaptive_margin(d_base_m=d_base_m, d_min_m=d_min_m, d_max_m=d_max_m)
    positions_now = np.array([b.state_vector[:3] for b in beliefs])

    return [
        assess_pair(
            i, j, predicted[i].trajectory, predicted[j].trajectory, physics_dt_s,
            combined_relative_sigma(beliefs[i], beliefs[j]),
            pair_density(i, j, positions_now, density_radius_m, density_ref_count),
            margin_fn,
        )
        for i, j in generate_candidate_pairs(len(beliefs))
    ]
