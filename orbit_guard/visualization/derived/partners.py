"""Each satellite's closest live conjunction partner at every step (inspector "CLOSEST", EC-08)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.visualization.derived.status import (
    STATUS_COLLISION,
    pair_status_matrix,
    retired_matrix,
)
from orbit_guard.visualization.replay import EpisodeTrace


@dataclass(frozen=True)
class ClosestPartner:
    d_min_m: np.ndarray  # (T+1, N), NaN where the satellite has no live partner
    d_safe_m: np.ndarray  # (T+1, N), d_safe of that same pair
    partner: np.ndarray  # (T+1, N), partner index or -1


def closest_partner(trace: EpisodeTrace, pair_status: np.ndarray | None = None) -> ClosestPartner:
    """Closest pair by predicted d_min, ignoring collided pairs and pairs with a retired satellite (stale after impact)."""
    status = pair_status_matrix(trace) if pair_status is None else pair_status
    retired = retired_matrix(trace, status)
    steps, agents = trace.num_steps + 1, trace.num_agents
    closest = np.full((steps, agents), np.inf)
    d_safe = np.full((steps, agents), np.nan)
    partner = np.full((steps, agents), -1, dtype=np.int16)
    for pair, (i, j) in enumerate(trace.pair_index):
        live = (status[:, pair] != STATUS_COLLISION) & ~retired[:, i] & ~retired[:, j]
        for own, other in ((i, j), (j, i)):
            better = live & (trace.d_min_m[:, pair] < closest[:, own])
            closest[better, own] = trace.d_min_m[better, pair]
            d_safe[better, own] = trace.d_safe_m[better, pair]
            partner[better, own] = other
    closest[np.isinf(closest)] = np.nan
    return ClosestPartner(closest, d_safe, partner)
