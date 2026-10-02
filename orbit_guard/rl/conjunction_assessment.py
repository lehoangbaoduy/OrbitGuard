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

from orbit_guard.conjunction.pair_assessment import PairAssessment, assess_all_pairs
from orbit_guard.physics.frames import eci_vector_to_rtn
from orbit_guard.rl.observation import NeighborCandidate
from orbit_guard.sensing.belief import AgentBelief

__all__ = ["PairAssessment", "assess_all_pairs", "build_neighbor_candidates"]


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
