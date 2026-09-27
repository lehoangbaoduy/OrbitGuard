"""Adaptive safety margin -- PRD §22, Core Contribution #1.

    d_safe,ij = clip( d_base * [1 + alpha*Sigma_norm,ij + beta*rho_ij + gamma*R_ij], d_min, d_max )
    rho_ij     = min(1, count_within(midpoint(i,j), d_density_ref) / rho_ref_count)

Processing order (PRD §22, never violated): predict -> TCA/d_min -> risk
proxy -> uncertainty + local density -> d_safe. `compute_d_safe` therefore
takes the three already-computed scalars (sigma_norm, rho, risk) and nothing
that could feed d_safe back into itself.

`Sigma_norm,ij` (the normalized-uncertainty term) is not numerically defined
by the PRD beyond the symbol. This module uses the provisional choice
`sigma_rel_m / d_base_m` -- see docs/OrbitGuard_progress.md §3 Q6.
"""

from __future__ import annotations

import numpy as np


def compute_sigma_norm(sigma_rel_m: float, d_base_m: float = 1000.0) -> float:
    """Dimensionless, provisional normalization of the combined relative uncertainty."""
    if sigma_rel_m < 0:
        raise ValueError(f"sigma_rel_m must be >= 0, got {sigma_rel_m}")
    if d_base_m <= 0:
        raise ValueError(f"d_base_m must be > 0, got {d_base_m}")
    return sigma_rel_m / d_base_m


def compute_local_density(
    midpoint_m: np.ndarray,
    all_positions_m: np.ndarray,
    exclude_indices: set[int],
    density_radius_m: float = 50_000.0,
    density_ref_count: float = 5.0,
) -> float:
    """rho_ij: fraction (capped at 1) of other satellites within density_radius_m of the pair's midpoint.

    `all_positions_m` holds every satellite's position (shape (M, 3)) --
    document at the call site that these must be belief positions, not
    ground truth, when used inside the RL-facing pipeline. `exclude_indices`
    must contain both pair members (i, j) so a satellite never counts itself
    or its pair partner as a "neighbor".
    """
    if density_radius_m <= 0:
        raise ValueError(f"density_radius_m must be > 0, got {density_radius_m}")
    if density_ref_count <= 0:
        raise ValueError(f"density_ref_count must be > 0, got {density_ref_count}")

    count = 0
    for idx in range(all_positions_m.shape[0]):
        if idx in exclude_indices:
            continue
        distance = float(np.linalg.norm(all_positions_m[idx] - midpoint_m))
        if distance < density_radius_m:
            count += 1

    return float(min(1.0, count / density_ref_count))


def compute_d_safe(
    sigma_norm: float,
    rho: float,
    risk: float,
    d_base_m: float = 1000.0,
    d_min_m: float = 200.0,
    d_max_m: float = 5000.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> float:
    """Pairwise, symmetric, bounded adaptive margin (PRD §22)."""
    if sigma_norm < 0:
        raise ValueError(f"sigma_norm must be >= 0, got {sigma_norm}")
    if rho < 0:
        raise ValueError(f"rho must be >= 0, got {rho}")
    if risk < 0:
        raise ValueError(f"risk must be >= 0, got {risk}")
    if d_min_m > d_max_m:
        raise ValueError(f"d_min_m ({d_min_m}) must be <= d_max_m ({d_max_m})")

    raw = d_base_m * (1.0 + alpha * sigma_norm + beta * rho + gamma * risk)
    return float(np.clip(raw, d_min_m, d_max_m))
