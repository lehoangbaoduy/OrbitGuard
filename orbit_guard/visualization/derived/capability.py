"""Thruster capability flags (EC-07): offline and limited-delta-v satellites."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.visualization.replay import EpisodeTrace

LIMITED_EPS_MPS = 1e-9


@dataclass(frozen=True)
class Capability:
    offline: np.ndarray  # (N,) maneuver_enabled is false
    limited: np.ndarray  # (N,) thrusters work but the starting tank is below the nominal budget


def capability_flags(trace: EpisodeTrace, budget_mps: float) -> Capability:
    """`budget_mps` is the nominal per-satellite budget (from trace meta, never a literal)."""
    enabled = np.asarray(trace.maneuver_enabled, dtype=bool)
    return Capability(offline=~enabled, limited=enabled & (trace.remaining_dv_mps[0] < budget_mps - LIMITED_EPS_MPS))
