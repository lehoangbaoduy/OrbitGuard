"""Per-agent dense reward (PRD §25), as pure framework-free functions.

    R_i = w1*R_safety + w2*R_mission - w3*||dv|| - w4*C_coordination
          [+ collision_penalty if a ground-truth collision occurred]

The sign convention is asymmetric on purpose, mirroring the PRD formula:
  * `compute_safety_term` / `compute_mission_term` return values that are
    ALREADY reward-signed (higher is better) and are ADDED in `compute_reward`.
    `R_mission = -D_mission` is an explicit freeze (the PRD never states it).
  * `compute_delta_v_term` / `compute_coordination_term` return RAW
    non-negative magnitudes and are SUBTRACTED in `compute_reward`. They are
    never pre-negated, which would double the sign.

`R_safety` uses the MINIMUM pairwise margin, not the mean, so one dangerous
encounter dominates instead of being diluted by benign ones.
`compute_coordination_term` stays raw and unnormalized (frozen spec, Q13); the
clipping PRD §25 requires before training is the separate
`compute_clipped_coordination_term`, min(clip, raw term), with the clip set by
`reward.coordination_clip` in configs/default.yaml (1.0; CORE-2). Each pair's
risk is in [0, 1], but the raw term sums the degradation over up to N(N-1)/2
pairs; clipping at 1 keeps it O(1) like the other bounded terms, so one step
cannot dominate the gradient. Its weight remains `compute_reward`'s `w4`.

Never imports the environment, training code, or any RL framework.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def compute_pairwise_margin(d_min_m: float, d_safe_m: float, epsilon_m: float = 1.0) -> float:
    """m_ij = (d_min - d_safe) / (d_safe + epsilon): negative inside the adaptive margin."""
    if d_safe_m < 0:
        raise ValueError(f"d_safe_m must be >= 0, got {d_safe_m}")
    if epsilon_m <= 0:
        raise ValueError(f"epsilon_m must be > 0, got {epsilon_m}")
    return (d_min_m - d_safe_m) / (d_safe_m + epsilon_m)


def compute_safety_term(margins: Sequence[float], clip_min: float = -1.0, clip_max: float = 1.0) -> float:
    """clip(min(margins)); maximally safe (clip_max) when there are no active pairs."""
    if clip_min > clip_max:
        raise ValueError(f"clip_min ({clip_min}) must be <= clip_max ({clip_max})")
    if not margins:
        return clip_max
    return max(clip_min, min(clip_max, min(margins)))


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
    """R_mission = -D_mission = -(q_a*d_a + q_e*d_e + q_i*d_i + q_m*d_m)."""
    for name, value in (("d_a", d_a), ("d_e", d_e), ("d_i", d_i), ("d_m", d_m)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    return -(q_a * d_a + q_e * d_e + q_i * d_i + q_m * d_m)


def compute_delta_v_term(delta_v_mps: Sequence[float]) -> float:
    """Raw Euclidean norm of the applied delta-v (not negated)."""
    return math.sqrt(sum(component**2 for component in delta_v_mps))


def compute_coordination_term(total_risk_pre: float, total_risk_post: float) -> float:
    """One-sided risk degradation, raw and unnormalized: improvements score 0."""
    return max(0.0, total_risk_post - total_risk_pre)


def compute_clipped_coordination_term(total_risk_pre: float, total_risk_post: float, clip: float) -> float:
    """min(clip, one-sided risk degradation): the coordination term the environment rewards with (PRD §25)."""
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError(f"clip must be finite and > 0, got {clip}")
    return min(clip, compute_coordination_term(total_risk_pre, total_risk_post))


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
    """Assemble the final per-agent reward; the only place term signs are combined."""
    reward = w1 * safety_term + w2 * mission_term - w3 * delta_v_term - w4 * coordination_term
    if collision_occurred:
        reward += collision_penalty
    return reward
