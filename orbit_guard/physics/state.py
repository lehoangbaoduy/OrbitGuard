"""Per-satellite numerical state and capability model (PRD §10, §13).

Cartesian state is the numerical truth; orbital elements are derived
diagnostics computed on demand (see orbit_guard.physics.frames), never an
independently propagated truth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SatelliteState:
    """x_i = [r_x, r_y, r_z, v_x, v_y, v_z] in the OG-ECI frame (meters, m/s)."""

    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]

    def as_vector(self) -> np.ndarray:
        """Return a fresh 6-vector [r, v]; mutating it never affects this state."""
        return np.array([*self.position_m, *self.velocity_mps], dtype=np.float64)

    @staticmethod
    def from_vector(vector: np.ndarray) -> SatelliteState:
        return SatelliteState(
            position_m=(float(vector[0]), float(vector[1]), float(vector[2])),
            velocity_mps=(float(vector[3]), float(vector[4]), float(vector[5])),
        )


@dataclass(frozen=True)
class SatelliteCapability:
    """Explicit capability vector (PRD §13) — a remaining-Δv budget, not physical propellant."""

    max_delta_v_per_action_mps: float
    remaining_delta_v_mps: float
    maneuver_enabled: bool
    mission_deviation_limit: float

    def with_remaining_delta_v(self, remaining_delta_v_mps: float) -> SatelliteCapability:
        return replace(self, remaining_delta_v_mps=remaining_delta_v_mps)
