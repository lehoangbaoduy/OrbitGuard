"""Thruster capability flags (EC-07): offline and limited-delta-v satellites."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from orbit_guard.visualization.replay import EpisodeTrace

LIMITED_EPS_MPS = 1e-9


@dataclass(frozen=True)
class Capability:
    offline: np.ndarray  # (N,) maneuver_enabled is false
    limited: np.ndarray  # (N,) thrusters work but the capability degradation cut the budget


def capability_flags(trace: EpisodeTrace, budget_mps: float, degraded: Sequence[bool] | None = None) -> Capability:
    """`degraded` is the recorded per-satellite mask (trace meta) of budgets the capability degradation cut.

    Traces recorded before that mask existed fall back to the old rule: a starting tank below `budget_mps`, the
    nominal per-satellite budget (from trace meta, never a literal). That rule is only right for fixed budgets.
    """
    enabled = np.asarray(trace.maneuver_enabled, dtype=bool)
    if degraded is not None:
        return Capability(offline=~enabled, limited=enabled & np.asarray(degraded, dtype=bool))
    return Capability(offline=~enabled, limited=enabled & (trace.remaining_dv_mps[0] < budget_mps - LIMITED_EPS_MPS))
