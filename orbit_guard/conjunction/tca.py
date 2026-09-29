"""Time of Closest Approach (TCA) and minimum separation computation (PRD §19)."""

from __future__ import annotations

import numpy as np


def compute_tca(
    trajectory_i: np.ndarray,
    trajectory_j: np.ndarray,
    dt: float,
    *,
    refine: bool = True,
) -> tuple[float, float]:
    """Compute time of closest approach and minimum separation over sampled trajectories.

    Locates the discrete minimum across the sampled trajectory window and, when
    ``refine=True`` (default), performs local quadratic relative-motion interpolation
    around the discrete minimum sample using the relative state vector
    ``[Δr, Δv]`` to resolve sub-timestep closest approach within the lookahead window.

    Args:
        trajectory_i: State history of satellite i with shape (n_steps + 1, 6),
            columns [x, y, z, vx, vy, vz] in SI units (m, m/s).
        trajectory_j: State history of satellite j with shape (n_steps + 1, 6),
            columns [x, y, z, vx, vy, vz] in SI units (m, m/s).
        dt: Time step between consecutive rows in seconds.
        refine: Whether to apply local sub-timestep interpolation around the discrete
            minimum. Defaults to True.

    Returns:
        Tuple (t_tca_s, d_min_m) where t_tca_s is the time in seconds from row 0
        at which minimum separation occurs, and d_min_m is the Euclidean separation
        distance in meters at that instant.

    Raises:
        ValueError: If trajectory shapes do not match, have invalid dimensions,
            or dt is non-positive.
    """
    if (
        trajectory_i.shape != trajectory_j.shape
        or trajectory_i.ndim != 2
        or trajectory_i.shape[1] < 6
        or trajectory_i.shape[0] == 0
        or dt <= 0.0
    ):
        raise ValueError(
            f"Invalid trajectories or timestep: shapes {trajectory_i.shape} and "
            f"{trajectory_j.shape}, dt={dt}."
        )

    pos_diff_m = trajectory_i[:, :3] - trajectory_j[:, :3]
    vel_diff_mps = trajectory_i[:, 3:6] - trajectory_j[:, 3:6]
    distances_m = np.linalg.norm(pos_diff_m, axis=1)
    min_idx = int(np.argmin(distances_m))

    r_k_m = pos_diff_m[min_idx]
    v_k_mps = vel_diff_mps[min_idx]
    v_norm_sq = float(np.dot(v_k_mps, v_k_mps))

    tau_min_s = -dt if (refine and min_idx > 0) else 0.0
    tau_max_s = dt if (refine and min_idx < len(distances_m) - 1) else 0.0
    tau_raw_s = -float(np.dot(r_k_m, v_k_mps)) / max(v_norm_sq, 1e-12)
    tau_star_s = float(np.clip(tau_raw_s, tau_min_s, tau_max_s))

    t_tca_s = float(min_idx * dt + tau_star_s)
    d_interp_m = float(np.linalg.norm(r_k_m + v_k_mps * tau_star_s))
    d_min_m = min(float(distances_m[min_idx]), d_interp_m)
    return t_tca_s, d_min_m
