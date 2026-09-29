"""Bounded conjunction risk proxy computation (PRD §21)."""

from __future__ import annotations

from scipy.special import expit


def risk_proxy(
    d_min_m: float,
    sigma_rel_m: float,
    epsilon_m: float = 1.0,
    z0: float = 3.0,
    k: float = 1.5,
) -> float:
    """Compute bounded conjunction risk proxy R_proxy in [0, 1].

    Evaluates:
        z = d_min_m / (sigma_rel_m + epsilon_m)
        R_proxy = expit(k * (z0 - z))

    Args:
        d_min_m: Minimum separation distance at TCA in meters (>= 0).
        sigma_rel_m: Combined relative position uncertainty in meters (>= 0).
        epsilon_m: Regularization constant in meters (> 0). Defaults to 1.0.
        z0: Normalized miss-distance crossover threshold (>= 0). Defaults to 3.0.
        k: Logistic steepness parameter (>= 0). Defaults to 1.5.

    Returns:
        Bounded risk proxy value R_proxy in [0.0, 1.0].

    Raises:
        ValueError: If any input parameter is negative (or epsilon_m <= 0).
    """
    if d_min_m < 0.0 or sigma_rel_m < 0.0 or epsilon_m <= 0.0 or z0 < 0.0 or k < 0.0:
        raise ValueError(
            "All inputs to risk_proxy must be non-negative (and epsilon_m > 0)."
        )

    z_score = d_min_m / (sigma_rel_m + epsilon_m)
    r_proxy = float(expit(k * (z0 - z_score)))
    return r_proxy
