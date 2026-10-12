"""Reward calculation functions for multi-agent collision avoidance (PRD §25).

Asymmetric sign convention:
    R_i = w1*R_safety,i + w2*R_mission,i - w3*||dv_i|| - w4*C_coordination,i
          [+ collision_penalty if collision_occurred]

- R_safety and R_mission are reward-signed (higher = better) and added (+w1, +w2).
- ||dv|| and C_coordination are raw non-negative magnitudes and subtracted (-w3, -w4).
- compute_reward is the sole assembler of final signs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def compute_pairwise_margin(
    d_min_m: float,
    d_safe_m: float,
    epsilon_m: float = 1.0,
) -> float:
    """Compute the normalized pairwise safety margin m_ij between two satellites.

    Formula:
        m_ij = (d_min_m - d_safe_m) / (d_safe_m + epsilon_m)

    Parameters:
        d_min_m: Minimum predicted separation distance at TCA in meters.
        d_safe_m: Adaptive safety bubble radius in meters.
        epsilon_m: Regularization distance in meters to prevent division by zero.

    Returns:
        Pairwise safety margin (dimensionless float). Positive indicates
        separation exceeds d_safe_m, zero indicates exactly d_safe_m,
        negative indicates encroachment inside the safety bubble.
    """
    if d_safe_m < 0.0:
        raise ValueError(f"d_safe_m must be >= 0, got {d_safe_m}")
    if epsilon_m <= 0.0:
        raise ValueError(f"epsilon_m must be > 0, got {epsilon_m}")
    return float((d_min_m - d_safe_m) / (d_safe_m + epsilon_m))


def compute_safety_term(
    margins: Sequence[float],
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> float:
    """Compute the clipped worst-case safety term across all pairwise margins.

    Uses the MINIMUM (worst-case) margin so that a single dangerous encounter
    dominates the safety signal rather than being washed out by benign pairs.
    If margins is empty (no active conjunctions), returns clip_max (maximally safe).

    Parameters:
        margins: Sequence of pairwise safety margins m_ij.
        clip_min: Lower bound for clipping (default: -1.0).
        clip_max: Upper bound for clipping (default: 1.0).

    Returns:
        Clipped minimum margin or clip_max if empty (dimensionless float).
    """
    if clip_min > clip_max:
        raise ValueError(f"clip_min ({clip_min}) cannot exceed clip_max ({clip_max})")
    if not margins:
        return float(clip_max)
    worst_margin = min(margins)
    return float(max(clip_min, min(clip_max, worst_margin)))


def compute_mission_term(
    d_a: float,
    d_e: float,
    d_i: float,
    d_m: float,
    q_a: float = 0.25,
    q_e: float = 0.25,
    q_i: float = 0.25,
    q_m: float = 0.25,
) -> float:
    """Compute the mission reward term as the negative weighted orbital element deviation.

    Formula:
        R_mission = -(q_a * d_a + q_e * d_e + q_i * d_i + q_m * d_m)

    Parameters:
        d_a: Normalized semi-major axis deviation (>= 0).
        d_e: Normalized eccentricity deviation (>= 0).
        d_i: Normalized inclination deviation (>= 0).
        d_m: Normalized mean anomaly deviation (>= 0).
        q_a, q_e, q_i, q_m: Element weighting coefficients (default: 0.25 each).

    Returns:
        Negative weighted sum of deviations (<= 0, dimensionless float).

    Raises:
        ValueError: If any deviation component is negative or any weight is negative.
    """
    if d_a < 0.0 or d_e < 0.0 or d_i < 0.0 or d_m < 0.0:
        raise ValueError(
            f"Mission deviation components must be non-negative, got "
            f"d_a={d_a}, d_e={d_e}, d_i={d_i}, d_m={d_m}"
        )
    if q_a < 0.0 or q_e < 0.0 or q_i < 0.0 or q_m < 0.0:
        raise ValueError(
            f"Mission weights must be non-negative, got "
            f"q_a={q_a}, q_e={q_e}, q_i={q_i}, q_m={q_m}"
        )
    d_mission = q_a * d_a + q_e * d_e + q_i * d_i + q_m * d_m
    return float(-d_mission)


def compute_delta_v_term(delta_v_mps: Sequence[float]) -> float:
    """Compute the raw Euclidean norm of the delta-v maneuver vector.

    Returns ||delta_v_mps|| as a raw non-negative magnitude in m/s.
    This is NOT pre-negated; subtraction happens during reward assembly.

    Parameters:
        delta_v_mps: Delta-v vector components in m/s.

    Returns:
        Raw Euclidean norm ||delta_v_mps|| in m/s (>= 0.0).
    """
    return float(math.sqrt(sum(x * x for x in delta_v_mps)))


def compute_coordination_term(total_risk_pre: float, total_risk_post: float) -> float:
    """Compute the one-sided constellation-wide risk degradation.

    Formula:
        C_coordination = max(0.0, total_risk_post - total_risk_pre)

    Only positive risk degradation is penalized; risk reductions score 0.0
    and are not double-rewarded. Output is raw and left unnormalized.

    Parameters:
        total_risk_pre: Constellation-wide total risk before joint maneuver.
        total_risk_post: Constellation-wide total risk after joint maneuver.

    Returns:
        One-sided positive risk degradation (>= 0.0, dimensionless float).
    """
    return float(max(0.0, total_risk_post - total_risk_pre))


def compute_reward(
    safety_term: float,
    mission_term: float,
    delta_v_term: float,
    coordination_term: float,
    collision_occurred: bool,
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 0.1,
    w4: float = 0.5,
    collision_penalty: float = -100.0,
) -> float:
    """Assemble the per-agent dense reward from individual terms.

    Formula:
        R_i = w1*safety_term + w2*mission_term - w3*delta_v_term - w4*coordination_term
              [+ collision_penalty if collision_occurred]

    Note on sign convention:
        - safety_term and mission_term are reward-signed (higher = better)
          and added with +w1 and +w2.
        - delta_v_term and coordination_term are raw non-negative magnitudes
          and subtracted with -w3 and -w4.
        - collision_penalty is added if collision_occurred is True.

    Parameters:
        safety_term: Clipped minimum pairwise margin.
        mission_term: Negative weighted deviation R_mission = -D_mission.
        delta_v_term: Raw Euclidean norm ||delta_v|| in m/s.
        coordination_term: Raw positive risk degradation max(0, post - pre).
        collision_occurred: Whether a collision occurred for this satellite.
        w1: Weight for safety term (default: 1.0).
        w2: Weight for mission term (default: 0.3).
        w3: Weight for delta-v term (default: 0.1).
        w4: Weight for coordination term (default: 0.5).
        collision_penalty: Terminal collision penalty (default: -100.0).

    Returns:
        Total assembled reward (float).
    """
    reward = (
        w1 * safety_term
        + w2 * mission_term
        - w3 * delta_v_term
        - w4 * coordination_term
    )
    if collision_occurred:
        reward += collision_penalty
    return float(reward)
