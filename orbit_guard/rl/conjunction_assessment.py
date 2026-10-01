"""Per-control-step, belief-based conjunction assessment (PRD §19-22, §24).

Wraps Phase 2's ground-truth-free pipeline (screening -> predicted TCA/d_min
-> risk proxy -> adaptive margin) into what `rl/environment.py` needs each
control step: a `PairAssessment` per candidate pair, and the finished
`NeighborCandidate` list for one agent's observation (Part B, PRD §24).

Entirely belief-based by construction -- every input here is an `AgentBelief`
(Phase 2's sensing/belief.py leakage-boundary type), never a `GroundTruthState`;
`predict_belief` itself raises `TypeError` on anything else, so a leakage
mistake at this layer fails loudly rather than silently reaching the policy
or reward through this module. Ground truth is used ONLY for the physical
collision check in `rl/environment.py`, never here (docs/OrbitGuard_progress.md
Q14/Q15).

Relative geometry in each `NeighborCandidate` is taken at the CURRENT instant
(not projected forward to the predicted TCA) and expressed in the observing
agent's own current RTN frame (docs/OrbitGuard_progress.md Q16/Q10) -- this
is the raw "what does my neighbor look like right now" perceptual input;
`d_min_m`/`risk_proxy` separately carry the forward-looking predicted threat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from orbit_guard.conjunction.risk import risk_proxy as compute_risk_proxy
from orbit_guard.conjunction.screening import generate_candidate_pairs
from orbit_guard.conjunction.tca import compute_tca
from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.frames import eci_vector_to_rtn
from orbit_guard.rl.observation import NeighborCandidate
from orbit_guard.safety.adaptive_margin import compute_d_safe, compute_local_density, compute_sigma_norm
from orbit_guard.sensing.belief import AgentBelief, combined_relative_sigma, predict_belief


@dataclass(frozen=True)
class PairAssessment:
    """One candidate pair's belief-predicted conjunction assessment."""

    i: int
    j: int
    d_min_m: float
    risk_proxy: float
    sigma_rel_m: float
    d_safe_m: float


def assess_all_pairs(
    beliefs: list[AgentBelief],
    lookahead_s: float,
    physics_dt_s: float,
    density_radius_m: float = 50_000.0,
    density_ref_count: float = 5.0,
    d_base_m: float = 1000.0,
    d_min_m: float = 200.0,
    d_max_m: float = 5000.0,
) -> list[PairAssessment]:
    """Predict every belief `lookahead_s` forward and assess all candidate pairs."""
    n = len(beliefs)
    n_steps = max(1, round(lookahead_s / physics_dt_s))
    predicted = [predict_belief(b, physics_dt_s, n_steps, state_derivative) for b in beliefs]
    positions_now = np.array([b.state_vector[:3] for b in beliefs])

    assessments = []
    for i, j in generate_candidate_pairs(n):
        _t_tca_s, d_min_pred_m = compute_tca(predicted[i].trajectory, predicted[j].trajectory, physics_dt_s)
        sigma_rel_m = combined_relative_sigma(beliefs[i], beliefs[j])
        risk = compute_risk_proxy(d_min_pred_m, sigma_rel_m)

        midpoint = (positions_now[i] + positions_now[j]) / 2.0
        sigma_norm = compute_sigma_norm(sigma_rel_m, d_base_m=d_base_m)
        rho = compute_local_density(
            midpoint, positions_now, exclude_indices={i, j},
            density_radius_m=density_radius_m, density_ref_count=density_ref_count,
        )
        d_safe = compute_d_safe(sigma_norm, rho, risk, d_base_m=d_base_m, d_min_m=d_min_m, d_max_m=d_max_m)

        assessments.append(
            PairAssessment(i=i, j=j, d_min_m=d_min_pred_m, risk_proxy=risk, sigma_rel_m=sigma_rel_m, d_safe_m=d_safe)
        )
    return assessments


def build_neighbor_candidates(
    agent_index: int, beliefs: list[AgentBelief], assessments: list[PairAssessment]
) -> list[NeighborCandidate]:
    """`NeighborCandidate` list for one agent -- relative vectors in ITS OWN current RTN frame."""
    own_position_m = beliefs[agent_index].state_vector[:3]
    own_velocity_mps = beliefs[agent_index].state_vector[3:6]

    candidates = []
    for assessment in assessments:
        if agent_index not in (assessment.i, assessment.j):
            continue
        other_index = assessment.j if assessment.i == agent_index else assessment.i
        rel_pos_eci = beliefs[other_index].state_vector[:3] - own_position_m
        rel_vel_eci = beliefs[other_index].state_vector[3:6] - own_velocity_mps
        candidates.append(
            NeighborCandidate(
                relative_position_m=eci_vector_to_rtn(rel_pos_eci, own_position_m, own_velocity_mps),
                relative_velocity_mps=eci_vector_to_rtn(rel_vel_eci, own_position_m, own_velocity_mps),
                d_min_m=assessment.d_min_m,
                risk_proxy=assessment.risk_proxy,
                d_safe_m=assessment.d_safe_m,
                uncertainty_summary_m=assessment.sigma_rel_m,
            )
        )
    return candidates
