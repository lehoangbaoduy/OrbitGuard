"""Deterministic maneuver-intent -> delta-v mapping (PRD §23).

Six discrete intents map to an RTN-frame delta-v, converted to OG-ECI via the
frozen Phase 1 frame transform. This is the ONLY place the intent->direction
table lives; nothing else in the codebase may reinterpret it.

    | Intent            | RTN direction         | Magnitude              |
    |-------------------|-----------------------|------------------------|
    | NO_MANEUVER       | --                    | 0                      |
    | RAISE_ORBIT       | +T (prograde)         | base_delta_v           |
    | LOWER_ORBIT       | -T (retrograde)       | base_delta_v           |
    | PHASE_FORWARD     | -T (retrograde, fine) | phase_fraction * base  |
    | PHASE_BACKWARD    | +T (prograde, fine)   | phase_fraction * base  |
    | RADIAL_ADJUSTMENT | +R (radially outward) | base_delta_v           |

A prograde burn raises semi-major axis, so the satellite's orbital period
grows and it drifts BACKWARD relative to an unmaneuvered neighbor -- which is
why PHASE_FORWARD is retrograde (-T) despite the intuitive-sounding name.

This is a pure geometric mapping: capability checks (remaining budget,
maneuver_enabled) belong to the environment, not here.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from orbit_guard.physics.frames import rtn_vector_to_eci


class ManeuverIntent(IntEnum):
    NO_MANEUVER = 0
    RAISE_ORBIT = 1
    LOWER_ORBIT = 2
    PHASE_FORWARD = 3
    PHASE_BACKWARD = 4
    RADIAL_ADJUSTMENT = 5


_RADIAL_AXIS = 0
_TANGENTIAL_AXIS = 1

# intent -> (RTN axis, sign, scales_with_phase_fraction)
_RTN_TABLE: dict[ManeuverIntent, tuple[int, float, bool]] = {
    ManeuverIntent.RAISE_ORBIT: (_TANGENTIAL_AXIS, +1.0, False),
    ManeuverIntent.LOWER_ORBIT: (_TANGENTIAL_AXIS, -1.0, False),
    ManeuverIntent.PHASE_FORWARD: (_TANGENTIAL_AXIS, -1.0, True),
    ManeuverIntent.PHASE_BACKWARD: (_TANGENTIAL_AXIS, +1.0, True),
    ManeuverIntent.RADIAL_ADJUSTMENT: (_RADIAL_AXIS, +1.0, False),
}


def map_intent_to_delta_v(
    intent: ManeuverIntent | int,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    base_delta_v_mps: float = 0.1,
    phase_fraction: float = 0.5,
) -> np.ndarray:
    """Return the 3-vector delta-v for `intent` in OG-ECI (same frame as the inputs)."""
    resolved_intent = ManeuverIntent(intent)  # raises ValueError on an invalid value
    if base_delta_v_mps < 0:
        raise ValueError(f"base_delta_v_mps must be >= 0, got {base_delta_v_mps}")
    if phase_fraction < 0:
        raise ValueError(f"phase_fraction must be >= 0, got {phase_fraction}")

    if resolved_intent is ManeuverIntent.NO_MANEUVER:
        return np.zeros(3)

    axis, sign, is_phase_burn = _RTN_TABLE[resolved_intent]
    magnitude_mps = base_delta_v_mps * (phase_fraction if is_phase_burn else 1.0)
    delta_v_rtn = np.zeros(3)
    delta_v_rtn[axis] = sign * magnitude_mps
    return rtn_vector_to_eci(delta_v_rtn, position_m, velocity_mps)
