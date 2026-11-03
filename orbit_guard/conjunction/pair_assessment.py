"""Belief-based assessment of one satellite pair or of every candidate pair (PRD §19-22).

Shared by the RL environment (observations, rewards) and the safety shield
(candidate-maneuver consequence prediction), so both score a conjunction with
literally the same geometry -> risk -> adaptive-margin pipeline.

`assess_proposal` is the ONE reward estimator of both shield arms (CORE-2, PRD
§25/§28, Q18/Q25): the predicted outcome of a RAW joint proposal from the
decision-time beliefs. Burns are mapped from each satellite's belief state by
the frozen §23 mapper, sized by that satellite's own per-action limit and never
capability-capped (an infeasible request is scored as requested); satellites
proposing NO_MANEUVER, including non-controllable ones, reuse their baseline
prediction. The shield predicts its raw proposal with the same two functions
(`predict_proposal_trajectories`, `assess_trajectories`), so the no-shield and
shielded environments score R_safety and the coordination term identically.

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

from orbit_guard.conjunction.risk import DEFAULT_EPSILON_M, DEFAULT_K, DEFAULT_Z0
from orbit_guard.conjunction.risk import risk_proxy as compute_risk_proxy
from orbit_guard.conjunction.screening import generate_candidate_pairs
from orbit_guard.conjunction.tca import compute_tca
from orbit_guard.physics.dynamics import state_derivative_batch
from orbit_guard.physics.integrator import propagate_batch
from orbit_guard.safety.adaptive_margin import (
    MarginFn,
    compute_local_density,
    make_adaptive_margin,
)
from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v
from orbit_guard.sensing.belief import (
    AgentBelief,
    PredictedState,
    combined_relative_sigma,
    predict_beliefs_batch,
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
    return predict_beliefs_batch(beliefs, physics_dt_s, prediction_steps(lookahead_s, physics_dt_s), state_derivative_batch)


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
    *,
    risk_epsilon_m: float = DEFAULT_EPSILON_M,
    risk_z0: float = DEFAULT_Z0,
    risk_k: float = DEFAULT_K,
) -> PairAssessment:
    """Predicted TCA/d_min -> risk proxy -> margin for two (possibly maneuvered) trajectories."""
    _t_tca_s, d_min_m = compute_tca(trajectory_i, trajectory_j, physics_dt_s)
    risk = compute_risk_proxy(d_min_m, sigma_rel_m, risk_epsilon_m, risk_z0, risk_k)
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
    risk_epsilon_m: float = DEFAULT_EPSILON_M,
    risk_z0: float = DEFAULT_Z0,
    risk_k: float = DEFAULT_K,
) -> list[PairAssessment]:
    """Assess every candidate pair; `predicted` reuses an earlier `predict_all`, `margin_fn` overrides the adaptive default."""
    if predicted is None:
        predicted = predict_all(beliefs, lookahead_s, physics_dt_s)
    if margin_fn is None:
        margin_fn = make_adaptive_margin(d_base_m=d_base_m, d_min_m=d_min_m, d_max_m=d_max_m)
    return assess_trajectories(
        beliefs, [p.trajectory for p in predicted], physics_dt_s, margin_fn,
        density_radius_m=density_radius_m, density_ref_count=density_ref_count,
        risk_epsilon_m=risk_epsilon_m, risk_z0=risk_z0, risk_k=risk_k,
    )


def assess_trajectories(
    beliefs: Sequence[AgentBelief],
    trajectories: Sequence[np.ndarray],
    physics_dt_s: float,
    margin_fn: MarginFn,
    *,
    density_radius_m: float = DEFAULT_DENSITY_RADIUS_M,
    density_ref_count: float = DEFAULT_DENSITY_REF_COUNT,
    risk_epsilon_m: float = DEFAULT_EPSILON_M,
    risk_z0: float = DEFAULT_Z0,
    risk_k: float = DEFAULT_K,
) -> list[PairAssessment]:
    """Assess every candidate pair on given predicted trajectories; sigma and density come from the beliefs now."""
    if len(trajectories) != len(beliefs):
        raise ValueError(f"need one trajectory per belief ({len(beliefs)}); got {len(trajectories)}")
    positions_now = np.array([b.state_vector[:3] for b in beliefs])
    return [
        assess_pair(
            i, j, trajectories[i], trajectories[j], physics_dt_s,
            combined_relative_sigma(beliefs[i], beliefs[j]),
            pair_density(i, j, positions_now, density_radius_m, density_ref_count),
            margin_fn,
            risk_epsilon_m=risk_epsilon_m, risk_z0=risk_z0, risk_k=risk_k,
        )
        for i, j in generate_candidate_pairs(len(beliefs))
    ]


def intent_delta_v(
    belief: AgentBelief, intent: int, max_delta_v_per_action_mps: float, phase_fraction: float
) -> np.ndarray:
    """The ECI burn `intent` maps to from the BELIEF state (PRD §23), sized by the satellite's own per-action limit."""
    if not isinstance(belief, AgentBelief):
        raise TypeError(
            f"intent_delta_v() requires an AgentBelief, got {type(belief).__name__} -- "
            "predicted burns use the estimated state, never ground truth (PRD §14)"
        )
    state = belief.state_vector
    return map_intent_to_delta_v(
        intent, state[:3], state[3:6], base_delta_v_mps=max_delta_v_per_action_mps, phase_fraction=phase_fraction
    )


def predict_proposal_trajectories(
    beliefs: Sequence[AgentBelief],
    intents: Sequence[int],
    max_delta_v_per_action_mps: Sequence[float],
    *,
    phase_fraction: float,
    lookahead_s: float,
    physics_dt_s: float,
    baseline_predictions: Sequence[PredictedState],
) -> list[np.ndarray]:
    """Each satellite's predicted trajectory under the raw joint proposal `intents`.

    NO_MANEUVER satellites reuse their baseline prediction; every burning satellite is propagated from
    belief + `intent_delta_v` in ONE vectorized call, in satellite-index order. No capability check is
    applied: this is the outcome of what was proposed, not of what would be allowed.
    """
    n = len(beliefs)
    if len(intents) != n or len(max_delta_v_per_action_mps) != n or len(baseline_predictions) != n:
        raise ValueError(
            f"need one intent, per-action limit and baseline prediction per satellite ({n}); got "
            f"{len(intents)}, {len(max_delta_v_per_action_mps)}, {len(baseline_predictions)}"
        )
    resolved = [ManeuverIntent(intent) for intent in intents]  # raises ValueError on an invalid intent
    moving = [i for i, intent in enumerate(resolved) if intent is not ManeuverIntent.NO_MANEUVER]
    trajectories = [prediction.trajectory for prediction in baseline_predictions]
    if not moving:
        return trajectories
    burns = {i: intent_delta_v(beliefs[i], resolved[i], max_delta_v_per_action_mps[i], phase_fraction) for i in moving}
    starts = np.array([
        np.concatenate([beliefs[i].state_vector[:3], beliefs[i].state_vector[3:6] + burns[i]]) for i in moving
    ])
    propagated = propagate_batch(
        starts, physics_dt_s, prediction_steps(lookahead_s, physics_dt_s), state_derivative_batch
    )
    for i, trajectory in zip(moving, propagated, strict=True):
        trajectories[i] = trajectory
    return trajectories


def assess_proposal(
    beliefs: Sequence[AgentBelief],
    intents: Sequence[int],
    max_delta_v_per_action_mps: Sequence[float],
    *,
    phase_fraction: float,
    lookahead_s: float,
    physics_dt_s: float,
    margin_fn: MarginFn,
    baseline_predictions: Sequence[PredictedState],
    density_radius_m: float = DEFAULT_DENSITY_RADIUS_M,
    density_ref_count: float = DEFAULT_DENSITY_REF_COUNT,
    risk_epsilon_m: float = DEFAULT_EPSILON_M,
    risk_z0: float = DEFAULT_Z0,
    risk_k: float = DEFAULT_K,
) -> list[PairAssessment]:
    """Predicted pair assessments of the raw joint proposal: the one R_safety/coordination estimator (CORE-2)."""
    trajectories = predict_proposal_trajectories(
        beliefs, intents, max_delta_v_per_action_mps, phase_fraction=phase_fraction,
        lookahead_s=lookahead_s, physics_dt_s=physics_dt_s, baseline_predictions=baseline_predictions,
    )
    return assess_trajectories(
        beliefs, trajectories, physics_dt_s, margin_fn,
        density_radius_m=density_radius_m, density_ref_count=density_ref_count,
        risk_epsilon_m=risk_epsilon_m, risk_z0=risk_z0, risk_k=risk_k,
    )
