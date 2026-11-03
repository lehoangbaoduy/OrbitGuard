"""Two-body + J2 acceleration model (PRD §10). No drag, SRP, third-body, or attitude in the MVP."""

from __future__ import annotations

import numpy as np

from orbit_guard.physics.constants import J2, MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M


def _require_finite_nonzero_position(position_m: np.ndarray) -> float:
    if not np.all(np.isfinite(position_m)):
        raise ValueError(f"Non-finite position: {position_m}")
    r = float(np.linalg.norm(position_m))
    if r <= 0.0:
        raise ValueError(f"Degenerate (zero-norm) position: {position_m}")
    return r


def _require_physically_valid_state(state_vector: np.ndarray) -> None:
    """Fail loudly on a non-finite state or a position inside Earth's equatorial radius (PRD §12)."""
    if not np.all(np.isfinite(state_vector)):
        raise ValueError(f"Non-finite state encountered: {state_vector}")
    radius_m = float(np.linalg.norm(state_vector[0:3]))
    if radius_m < R_EARTH_EQUATORIAL_M:
        raise ValueError(
            f"Physically invalid state: position radius {radius_m:.1f} m is below Earth's "
            f"equatorial radius {R_EARTH_EQUATORIAL_M:.1f} m"
        )


def two_body_acceleration(
    position_m: np.ndarray, mu: float = MU_EARTH_M3_S2
) -> np.ndarray:
    """Point-mass gravitational acceleration: a = -mu * r / |r|^3."""
    r = _require_finite_nonzero_position(position_m)
    return -mu * position_m / r**3


def j2_acceleration(
    position_m: np.ndarray,
    mu: float = MU_EARTH_M3_S2,
    j2: float = J2,
    r_eq: float = R_EARTH_EQUATORIAL_M,
) -> np.ndarray:
    """Earth J2 zonal-harmonic perturbing acceleration (standard closed form, Vallado/Curtis)."""
    r = _require_finite_nonzero_position(position_m)
    x, y, z = position_m
    z_over_r_sq = (z / r) ** 2
    coeff = -1.5 * j2 * mu * r_eq**2 / r**5
    return np.array(
        [
            coeff * x * (1 - 5 * z_over_r_sq),
            coeff * y * (1 - 5 * z_over_r_sq),
            coeff * z * (3 - 5 * z_over_r_sq),
        ]
    )


def total_acceleration(position_m: np.ndarray) -> np.ndarray:
    """r-double-dot = a_2body + a_J2 (PRD §10)."""
    return two_body_acceleration(position_m) + j2_acceleration(position_m)


def state_derivative(state_vector: np.ndarray) -> np.ndarray:
    """d/dt [r, v] = [v, a(r)]. Fails loudly on a non-finite or below-surface state (PRD §12)."""
    _require_physically_valid_state(state_vector)
    position = state_vector[0:3]
    velocity = state_vector[3:6]
    acceleration = total_acceleration(position)
    return np.concatenate([velocity, acceleration])


def two_body_state_derivative(state_vector: np.ndarray) -> np.ndarray:
    """Two-body-only d/dt [r, v] = [v, a_2body(r)] -- for the §12 conservation/sanity checks
    that must isolate two-body behavior from the J2 perturbation."""
    _require_physically_valid_state(state_vector)
    position = state_vector[0:3]
    velocity = state_vector[3:6]
    return np.concatenate([velocity, two_body_acceleration(position)])


def state_derivative_batch(states: np.ndarray) -> np.ndarray:
    """Vectorized `state_derivative` over rows of a (B, 6) array (same model, same failure checks)."""
    if not np.all(np.isfinite(states)):
        raise ValueError(f"Non-finite state encountered: {states}")
    position = states[:, 0:3]
    velocity = states[:, 3:6]
    r = np.linalg.norm(position, axis=1)
    if np.any(r <= 0.0):
        raise ValueError(f"Degenerate (zero-norm) position in batch: {position}")
    if np.any(r < R_EARTH_EQUATORIAL_M):
        raise ValueError(
            f"Physically invalid state: position radius {float(r.min()):.1f} m is below Earth's "
            f"equatorial radius {R_EARTH_EQUATORIAL_M:.1f} m"
        )
    z_over_r_sq = (position[:, 2] / r) ** 2
    j2_coeff = -1.5 * J2 * MU_EARTH_M3_S2 * R_EARTH_EQUATORIAL_M**2 / r**5
    j2_acceleration_batch = np.stack(
        [
            j2_coeff * position[:, 0] * (1 - 5 * z_over_r_sq),
            j2_coeff * position[:, 1] * (1 - 5 * z_over_r_sq),
            j2_coeff * position[:, 2] * (3 - 5 * z_over_r_sq),
        ],
        axis=1,
    )
    two_body_batch = -MU_EARTH_M3_S2 * position / (r**3)[:, None]
    return np.concatenate([velocity, two_body_batch + j2_acceleration_batch], axis=1)
