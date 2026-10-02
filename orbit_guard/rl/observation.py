"""Fixed-size, per-agent, decentralized observation (PRD §24).

Pure function of plain floats/arrays: no gymnasium/pettingzoo import, no frame
transforms. The caller (`rl/environment.py`) supplies neighbor geometry already
expressed in the OWNING agent's own RTN frame, and `own_mission_deviation`
already normalized (the scalar D_mission of PRD §25).

Layout (index order is part of the contract):
  own            (10,): [pos(3), vel(3), remaining_dv, mission_deviation,
                         max_dv_per_action, maneuver_enabled]
  neighbors   (k, 10) : [rel_pos(3), rel_vel(3), risk_proxy, d_safe,
                         uncertainty, neighbor_valid]
  global_summary (4,) : [threatening_neighbor_count, max_risk, total_risk,
                         min_separation_norm]

Neighbors are ranked by `risk_proxy` descending, ties broken by `d_min_m`
ascending -- never by satellite index, which would leak persistent identity
and violate PRD §24's anonymity requirement. The padding flag is deliberately
named `neighbor_valid`, not "mask" (PRD §27's action masking is unused here).
Global-summary statistics are computed over ALL candidates, not just the top-k.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from orbit_guard.physics.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M

POSITION_SCALE_M = R_EARTH_EQUATORIAL_M
VELOCITY_SCALE_MPS = float(np.sqrt(MU_EARTH_M3_S2 / R_EARTH_EQUATORIAL_M))
RELATIVE_POSITION_SCALE_M = 5000.0
RELATIVE_VELOCITY_SCALE_MPS = 1000.0
DELTA_V_BUDGET_SCALE_MPS = 2.0
MAX_DELTA_V_PER_ACTION_SCALE_MPS = 0.1
MINIMUM_SEPARATION_SCALE_M = 5000.0
THREATENING_RISK_THRESHOLD = 0.5

_OWN_DIM = 10
_NEIGHBOR_DIM = 10
_GLOBAL_SUMMARY_DIM = 4


@dataclass(frozen=True)
class NeighborCandidate:
    """One candidate neighbor, geometry in the OWNING agent's own RTN frame."""

    relative_position_m: np.ndarray
    relative_velocity_mps: np.ndarray
    d_min_m: float
    risk_proxy: float
    d_safe_m: float
    uncertainty_summary_m: float


def _own_vector(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    remaining_delta_v_mps: float,
    mission_deviation: float,
    max_delta_v_per_action_mps: float,
    maneuver_enabled: bool,
) -> np.ndarray:
    own = np.zeros(_OWN_DIM)
    own[0:3] = np.asarray(position_m) / POSITION_SCALE_M
    own[3:6] = np.asarray(velocity_mps) / VELOCITY_SCALE_MPS
    own[6] = remaining_delta_v_mps / DELTA_V_BUDGET_SCALE_MPS
    own[7] = mission_deviation
    own[8] = max_delta_v_per_action_mps / MAX_DELTA_V_PER_ACTION_SCALE_MPS
    own[9] = 1.0 if maneuver_enabled else 0.0
    return own


def _neighbor_matrix(ranked_top_k: Sequence[NeighborCandidate], k_neighbors: int) -> np.ndarray:
    neighbors = np.zeros((k_neighbors, _NEIGHBOR_DIM))
    for slot, candidate in enumerate(ranked_top_k):
        neighbors[slot, 0:3] = np.asarray(candidate.relative_position_m) / RELATIVE_POSITION_SCALE_M
        neighbors[slot, 3:6] = np.asarray(candidate.relative_velocity_mps) / RELATIVE_VELOCITY_SCALE_MPS
        neighbors[slot, 6] = candidate.risk_proxy
        neighbors[slot, 7] = candidate.d_safe_m / RELATIVE_POSITION_SCALE_M
        neighbors[slot, 8] = candidate.uncertainty_summary_m / RELATIVE_POSITION_SCALE_M
        neighbors[slot, 9] = 1.0
    return neighbors


def _global_summary(candidates: Sequence[NeighborCandidate]) -> np.ndarray:
    if not candidates:
        return np.array([0.0, 0.0, 0.0, 1.0])
    risks = [c.risk_proxy for c in candidates]
    threatening_count = float(sum(1 for risk in risks if risk > THREATENING_RISK_THRESHOLD))
    min_separation_norm = min(c.d_min_m for c in candidates) / MINIMUM_SEPARATION_SCALE_M
    return np.array([threatening_count, max(risks), sum(risks), min_separation_norm])


def build_observation(
    own_position_m: np.ndarray,
    own_velocity_mps: np.ndarray,
    own_remaining_delta_v_mps: float,
    own_mission_deviation: float,
    own_max_delta_v_per_action_mps: float,
    own_maneuver_enabled: bool,
    neighbor_candidates: Sequence[NeighborCandidate],
    k_neighbors: int = 4,
) -> dict[str, np.ndarray]:
    """Build one agent's observation dict (float64 arrays; the caller casts to float32)."""
    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")

    ranked = sorted(neighbor_candidates, key=lambda c: (-c.risk_proxy, c.d_min_m))
    return {
        "own": _own_vector(
            own_position_m,
            own_velocity_mps,
            own_remaining_delta_v_mps,
            own_mission_deviation,
            own_max_delta_v_per_action_mps,
            own_maneuver_enabled,
        ),
        "neighbors": _neighbor_matrix(ranked[:k_neighbors], k_neighbors),
        "global_summary": _global_summary(neighbor_candidates),
    }
