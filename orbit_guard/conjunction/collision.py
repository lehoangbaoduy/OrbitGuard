"""Ground-truth physical collision detection (PRD §18)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.conjunction.tca import compute_tca


@dataclass(frozen=True)
class CollisionResult:
    """Immutable result of a ground-truth pairwise collision check.

    Attributes:
        collided: True iff separation falls strictly below combined hard-body radius.
        collision_time_s: First sampled time (in seconds from t=0) of collision,
            or sub-timestep TCA if collision occurs strictly between samples,
            or None if collided is False.
        min_separation_m: Minimum Euclidean separation in meters across the window.
    """

    collided: bool
    collision_time_s: float | None
    min_separation_m: float


def detect_collision(
    trajectory_i: np.ndarray,
    trajectory_j: np.ndarray,
    dt: float,
    hard_body_radius_i_m: float,
    hard_body_radius_j_m: float,
) -> CollisionResult:
    """Check whether two sampled trajectories experience a physical collision.

    A collision occurs if and only if ||r_i(t) - r_j(t)|| < R_i + R_j (strict
    inequality) at any sampled timestep or refined closest-approach instant.

    Args:
        trajectory_i: State history of satellite i, shape (n_steps + 1, 6) in SI units.
        trajectory_j: State history of satellite j, shape (n_steps + 1, 6) in SI units.
        dt: Time step between consecutive samples in seconds.
        hard_body_radius_i_m: Hard-body radius of satellite i in meters.
        hard_body_radius_j_m: Hard-body radius of satellite j in meters.

    Returns:
        CollisionResult indicating whether a collision occurred, the first collision
        time in seconds (or None), and the minimum separation distance in meters.

    Raises:
        ValueError: If trajectory shapes mismatch, dimensions are invalid, dt <= 0,
            or either hard-body radius is negative.
    """
    if (
        trajectory_i.shape != trajectory_j.shape
        or hard_body_radius_i_m < 0.0
        or hard_body_radius_j_m < 0.0
    ):
        raise ValueError(
            f"Invalid inputs for detect_collision: shapes {trajectory_i.shape} and "
            f"{trajectory_j.shape}, radii=({hard_body_radius_i_m}, {hard_body_radius_j_m})."
        )

    t_tca_s, min_separation_m = compute_tca(trajectory_i, trajectory_j, dt)
    pos_diff_m = trajectory_i[:, :3] - trajectory_j[:, :3]
    distances_m = np.linalg.norm(pos_diff_m, axis=1)
    combined_radius_m = float(hard_body_radius_i_m + hard_body_radius_j_m)

    collision_indices = np.flatnonzero(distances_m < combined_radius_m)
    if min_separation_m < combined_radius_m:
        first_collision_time_s = (
            float(int(collision_indices[0]) * dt)
            if collision_indices.size > 0
            else t_tca_s
        )
        return CollisionResult(
            collided=True,
            collision_time_s=first_collision_time_s,
            min_separation_m=min_separation_m,
        )

    return CollisionResult(
        collided=False,
        collision_time_s=None,
        min_separation_m=min_separation_m,
    )
