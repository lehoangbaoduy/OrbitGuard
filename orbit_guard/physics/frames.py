"""Frame conventions and transforms (PRD §9).

**Frame convention (design decision, pending Astro sign-off — see
docs/OrbitGuard_progress.md §3 Q2):** the internal "OG-ECI" frame is defined
as whatever equatorial, Earth-centered, non-rotating frame the classical
orbital elements (COE) -> Cartesian conversion below produces (a standard
Vallado-style COE2RV). This is treated as TEME-equivalent, matching the
native frame of the TLE-derived elements in the frozen data snapshot.

Deliberately, no epoch-based propagation or TEME<->J2000 conversion is
performed anywhere in this simulator: per PRD §16, the TLE snapshot is used
only to fit *sampling distributions* over orbital elements, and each episode
seeds fresh synthetic satellites directly via `coe_to_cartesian` at a single
simulator-internal t=0 (see orbit_guard.scenarios.constellation). Real-world
epoch alignment across the ~3-day TLE pull spread is therefore irrelevant to
the physics: no satellite is ever propagated from its real TLE epoch.

RTN (Radial / Transverse / Normal) is the local maneuver frame: R points
radially outward from Earth's center, N is along the orbit angular-momentum
vector (orbit-normal), and T = N x R completes a right-handed (R, T, N)
triad -- T is the along-track/transverse direction.
"""

from __future__ import annotations

import numpy as np

from orbit_guard.physics.constants import MU_EARTH_M3_S2

_KEPLER_TOLERANCE = 1e-12
_KEPLER_MAX_ITERATIONS = 100
_MIN_NODE_VECTOR_NORM = 1e-8
_MIN_ECCENTRICITY_VECTOR_NORM = 1e-8


def solve_kepler_equation(
    mean_anomaly_rad: float, eccentricity: float, tolerance: float = _KEPLER_TOLERANCE
) -> float:
    """Solve M = E - e*sin(E) for E via Newton-Raphson."""
    M = mean_anomaly_rad
    E = M if eccentricity < 0.8 else np.pi
    for _ in range(_KEPLER_MAX_ITERATIONS):
        delta = (E - eccentricity * np.sin(E) - M) / (1 - eccentricity * np.cos(E))
        E = E - delta
        if abs(delta) < tolerance:
            break
    return E


def coe_to_cartesian(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_rad: float,
    raan_rad: float,
    arg_periapsis_rad: float,
    mean_anomaly_rad: float,
    mu: float = MU_EARTH_M3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard COE2RV (Vallado): classical orbital elements -> (position_m, velocity_mps)."""
    e = eccentricity
    a = semi_major_axis_m
    E = solve_kepler_equation(mean_anomaly_rad, e)
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0), np.sqrt(1 - e) * np.cos(E / 2.0)
    )
    r_mag = a * (1 - e * np.cos(E))
    p = a * (1 - e**2)

    r_pqw = r_mag * np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array(
        [-np.sin(true_anomaly), e + np.cos(true_anomaly), 0.0]
    )

    rotation = _perifocal_to_eci_rotation(inclination_rad, raan_rad, arg_periapsis_rad)
    return rotation @ r_pqw, rotation @ v_pqw


def _perifocal_to_eci_rotation(
    inclination_rad: float, raan_rad: float, arg_periapsis_rad: float
) -> np.ndarray:
    """Explicit direction-cosine matrix, perifocal (PQW) -> OG-ECI (Vallado eq. 2-25/2-26)."""
    cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)
    cos_i, sin_i = np.cos(inclination_rad), np.sin(inclination_rad)
    cos_argp, sin_argp = np.cos(arg_periapsis_rad), np.sin(arg_periapsis_rad)

    return np.array(
        [
            [
                cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
                -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
                sin_raan * sin_i,
            ],
            [
                sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
                -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
                -cos_raan * sin_i,
            ],
            [
                sin_argp * sin_i,
                cos_argp * sin_i,
                cos_i,
            ],
        ]
    )


def cartesian_to_coe(
    position_m: np.ndarray, velocity_mps: np.ndarray, mu: float = MU_EARTH_M3_S2
) -> tuple[float, float, float, float, float, float]:
    """Inverse of coe_to_cartesian. Diagnostic only (PRD §10) -- not an independent truth.

    Raises ValueError on near-equatorial or near-parabolic/hyperbolic orbits,
    where the standard algorithm is singular (node vector or shape undefined)
    -- OrbitGuard's fixed 53-degree shell never approaches either case, so
    this is a documented scope limit, not a gap that needs handling.
    """
    r_vec = np.asarray(position_m, dtype=np.float64)
    v_vec = np.asarray(velocity_mps, dtype=np.float64)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    node_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    n = np.linalg.norm(node_vec)
    if n < _MIN_NODE_VECTOR_NORM:
        raise ValueError(
            "cartesian_to_coe is undefined for a near-equatorial orbit (node vector ~0); "
            "not supported for OrbitGuard's fixed ~53-degree shell."
        )

    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    energy = v**2 / 2.0 - mu / r
    if abs(energy) < 1e-12:
        raise ValueError("cartesian_to_coe is undefined for a near-parabolic orbit.")
    a = -mu / (2.0 * energy)

    inclination = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    raan = np.arccos(np.clip(node_vec[0] / n, -1.0, 1.0))
    if node_vec[1] < 0:
        raan = 2 * np.pi - raan

    if e < _MIN_ECCENTRICITY_VECTOR_NORM:
        raise ValueError(
            "cartesian_to_coe is undefined for a near-circular orbit (eccentricity vector ~0); "
            "argument of periapsis is not meaningful."
        )

    arg_periapsis = np.arccos(np.clip(np.dot(node_vec, e_vec) / (n * e), -1.0, 1.0))
    if e_vec[2] < 0:
        arg_periapsis = 2 * np.pi - arg_periapsis

    cos_true_anomaly = np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)
    true_anomaly = np.arccos(cos_true_anomaly)
    if np.dot(r_vec, v_vec) < 0:
        true_anomaly = 2 * np.pi - true_anomaly

    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1 - e) * np.sin(true_anomaly / 2.0), np.sqrt(1 + e) * np.cos(true_anomaly / 2.0)
    )
    mean_anomaly = eccentric_anomaly - e * np.sin(eccentric_anomaly)
    mean_anomaly = mean_anomaly % (2 * np.pi)

    return a, e, inclination, raan, arg_periapsis, mean_anomaly


def _rtn_basis(position_m: np.ndarray, velocity_mps: np.ndarray) -> np.ndarray:
    """3x3 matrix whose rows are the (R, T, N) unit basis vectors, expressed in OG-ECI."""
    r_hat = position_m / np.linalg.norm(position_m)
    h_vec = np.cross(position_m, velocity_mps)
    n_hat = h_vec / np.linalg.norm(h_vec)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([r_hat, t_hat, n_hat])


def eci_vector_to_rtn(
    vector_eci: np.ndarray, position_m: np.ndarray, velocity_mps: np.ndarray
) -> np.ndarray:
    """Rotate a vector (e.g. Δv) from OG-ECI into the local RTN frame at this state."""
    return _rtn_basis(position_m, velocity_mps) @ vector_eci


def rtn_vector_to_eci(
    vector_rtn: np.ndarray, position_m: np.ndarray, velocity_mps: np.ndarray
) -> np.ndarray:
    """Rotate a vector from the local RTN frame back into OG-ECI at this state."""
    return _rtn_basis(position_m, velocity_mps).T @ vector_rtn
