"""Mission deviation against a no-maneuver shadow trajectory (PRD §25).

Compares a satellite's actual state to a same-instant NO-MANEUVER SHADOW
state (the same initial condition propagated with zero applied delta-v),
not a static t=0 reference -- J2 short-period oscillation alone swings an
unmaneuvered satellite's semi-major axis by ~12,350m and eccentricity by
~0.0012 over one 6h episode, both exceeding the tolerance bands below (see
docs/OrbitGuard_progress.md Q9 for the empirical measurement). A static
reference would flag an unmaneuvered satellite as off-mission, which is
meaningless.

Deliberately avoids `physics.frames.cartesian_to_coe`, which raises
ValueError for near-circular orbits when computing argument of periapsis --
irrelevant here, since mission deviation never needs argument of periapsis,
only semi-major axis, eccentricity MAGNITUDE, inclination, and an
along-track distance, none of which are singular for this project's shell.

This module is source-agnostic: callers decide whether `state`/`shadow_state`
come from ground truth or belief (see docs/OrbitGuard_progress.md Q14 --
`rl/environment.py` uses belief for both, consistently with PRD §14's
leakage-boundary philosophy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.physics.constants import MU_EARTH_M3_S2
from orbit_guard.physics.frames import eci_vector_to_rtn

_DEFAULT_SEMI_MAJOR_AXIS_TOLERANCE_M = 2000.0
_DEFAULT_ECCENTRICITY_TOLERANCE = 0.001
_DEFAULT_INCLINATION_TOLERANCE_RAD = np.radians(0.05)
_DEFAULT_ALONG_TRACK_TOLERANCE_M = 10_000.0

_DEFAULT_Q_A = 0.25
_DEFAULT_Q_E = 0.25
_DEFAULT_Q_I = 0.25
_DEFAULT_Q_M = 0.25


@dataclass(frozen=True)
class MissionDeviationComponents:
    """Four normalized, non-negative orbital-element deviation magnitudes (PRD §25)."""

    d_a: float
    d_e: float
    d_i: float
    d_m: float


def _semi_major_axis_m(position_m: np.ndarray, velocity_mps: np.ndarray) -> float:
    r = np.linalg.norm(position_m)
    v = np.linalg.norm(velocity_mps)
    return 1.0 / (2.0 / r - v**2 / MU_EARTH_M3_S2)


def _eccentricity_magnitude(position_m: np.ndarray, velocity_mps: np.ndarray) -> float:
    r_vec, v_vec = position_m, velocity_mps
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    e_vec = ((v**2 - MU_EARTH_M3_S2 / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU_EARTH_M3_S2
    return float(np.linalg.norm(e_vec))


def _inclination_rad(position_m: np.ndarray, velocity_mps: np.ndarray) -> float:
    h_vec = np.cross(position_m, velocity_mps)
    h = np.linalg.norm(h_vec)
    return float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)))


def compute_mission_deviation(
    state: np.ndarray,
    shadow_state: np.ndarray,
    semi_major_axis_tolerance_m: float = _DEFAULT_SEMI_MAJOR_AXIS_TOLERANCE_M,
    eccentricity_tolerance: float = _DEFAULT_ECCENTRICITY_TOLERANCE,
    inclination_tolerance_rad: float = _DEFAULT_INCLINATION_TOLERANCE_RAD,
    along_track_tolerance_m: float = _DEFAULT_ALONG_TRACK_TOLERANCE_M,
) -> MissionDeviationComponents:
    """Four normalized deviations between `state` and `shadow_state`, same instant.

    `state`/`shadow_state` are 6-vectors [x, y, z, vx, vy, vz] in OG-ECI, SI
    units. `d_m` (along-track phase deviation) is the shadow-frame RTN
    along-track (T) distance between the two positions, divided by
    `along_track_tolerance_m` -- not a raw mean-anomaly angle difference.
    """
    for name, value in [
        ("semi_major_axis_tolerance_m", semi_major_axis_tolerance_m),
        ("eccentricity_tolerance", eccentricity_tolerance),
        ("inclination_tolerance_rad", inclination_tolerance_rad),
        ("along_track_tolerance_m", along_track_tolerance_m),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    r = np.asarray(state[:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)
    r_shadow = np.asarray(shadow_state[:3], dtype=np.float64)
    v_shadow = np.asarray(shadow_state[3:6], dtype=np.float64)

    a, a_shadow = _semi_major_axis_m(r, v), _semi_major_axis_m(r_shadow, v_shadow)
    e, e_shadow = _eccentricity_magnitude(r, v), _eccentricity_magnitude(r_shadow, v_shadow)
    i, i_shadow = _inclination_rad(r, v), _inclination_rad(r_shadow, v_shadow)
    along_track_offset_m = eci_vector_to_rtn(r - r_shadow, r_shadow, v_shadow)[1]

    return MissionDeviationComponents(
        d_a=abs(a - a_shadow) / semi_major_axis_tolerance_m,
        d_e=abs(e - e_shadow) / eccentricity_tolerance,
        d_i=abs(i - i_shadow) / inclination_tolerance_rad,
        d_m=abs(along_track_offset_m) / along_track_tolerance_m,
    )


def compute_weighted_mission_deviation(
    components: MissionDeviationComponents,
    q_a: float = _DEFAULT_Q_A,
    q_e: float = _DEFAULT_Q_E,
    q_i: float = _DEFAULT_Q_I,
    q_m: float = _DEFAULT_Q_M,
) -> float:
    """D_mission = q_a*d_a + q_e*d_e + q_i*d_i + q_m*d_m (PRD §25).

    This scalar feeds BOTH `rl/observation.py`'s `own_mission_deviation`
    (unscaled) and, negated, matches `rl/reward.py::compute_mission_term`'s
    output -- the caller (`rl/environment.py`) must use the same
    `components`/weights for both so the two stay equal by construction
    (docs/OrbitGuard_progress.md Q12).
    """
    return q_a * components.d_a + q_e * components.d_e + q_i * components.d_i + q_m * components.d_m
