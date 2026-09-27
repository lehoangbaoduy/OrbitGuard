"""Constellation conjunction injection (PRD §17).

Injects a controlled, targeted close approach between two satellites in an
otherwise naturally-sampled constellation (`orbit_guard.scenarios.constellation`),
via back-propagation from the encounter rather than a forward search:

    1. Round the target TCA onto the physics_dt grid: t*.
    2. Propagate the primary satellite forward to t* -> (r_i, v_i).
    3. Rotate v_i about r_hat_i by a sampled crossing angle to build a
       relative velocity delta_v = 2|v_i|*sin(theta/2) -- this is §17's
       "relative velocity / orbital-plane relationship" parameter.
    4. Place r_j(t*) = r_i + d_target * n_hat, n_hat perpendicular to
       delta_v, so Delta_r . Delta_v = 0 at t* -- t* really is the local
       minimum of ||Delta_r(t)||, not just an approximate one.
    5. Two-body+J2 acceleration depends only on position (never velocity),
       so the dynamics are time-reversible: back-propagate the secondary to
       t=0 by forward-propagating (r_j(t*), -v_j(t*)) for t* seconds and
       negating the resulting velocity.

This module never imports orbit_guard.conjunction (Part B: screening/TCA/
collision/risk, owned by a different teammate and not yet implemented) --
achieved geometry is self-checked here with a direct distance computation.

Provisional astro parameters (flagged for real review, see
docs/OrbitGuard_progress.md §3 Q8): crossing angle range, near-miss and
collision-course d_min bands, and the collision-course injection fraction.
The PRD specifies the *training-mix* proportions (§17: 30/50/20 benign/
single/multi) exactly; it does not specify these geometric ranges.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.state import SatelliteCapability, SatelliteState
from orbit_guard.scenarios.constellation import (
    GeneratedSatellite,
    ShellDistribution,
    sample_constellation,
)

DEFAULT_DISTRIBUTION_MIX: Mapping[str, float] = {
    "benign": 0.30,
    "single_conjunction": 0.50,
    "multi_conjunction": 0.20,
}

_EPISODE_TYPE_CONJUNCTION_COUNT = {"benign": 0, "single_conjunction": 1, "multi_conjunction": 2}

_CROSSING_ANGLE_RANGE_RAD = (np.radians(0.5), np.radians(15.0))
_NEAR_MISS_D_MIN_RANGE_M = (50.0, 300.0)
_COLLISION_COURSE_D_MIN_RANGE_M = (1.0, 8.0)
_COLLISION_COURSE_FRACTION = 0.2


@dataclass(frozen=True)
class ConjunctionSpec:
    primary_index: int
    secondary_index: int
    target_tca_s: float
    achieved_tca_s: float
    target_d_min_m: float
    achieved_d_min_m: float
    crossing_angle_rad: float


@dataclass(frozen=True)
class ConjunctionScenario:
    satellites: tuple[GeneratedSatellite, ...]
    episode_type: str
    injected_conjunctions: tuple[ConjunctionSpec, ...]


def _rotate_vector(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotate v by angle_rad about a unit axis."""
    axis = axis / np.linalg.norm(axis)
    return (
        v * np.cos(angle_rad)
        + np.cross(axis, v) * np.sin(angle_rad)
        + axis * np.dot(axis, v) * (1 - np.cos(angle_rad))
    )


def _perpendicular_unit_vector(v: np.ndarray, fallback_axis: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to v, preferring `fallback_axis`'s plane."""
    for axis in (fallback_axis, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
        candidate = np.cross(v, axis)
        norm = np.linalg.norm(candidate)
        if norm > 1e-6:
            return candidate / norm
    raise RuntimeError("Could not construct a perpendicular vector -- v is degenerate")


def sample_episode_type(rng: np.random.Generator, distribution_mix: Mapping[str, float] = DEFAULT_DISTRIBUTION_MIX) -> str:
    total = sum(distribution_mix.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"distribution_mix must sum to 1.0, got {total}")
    labels = list(distribution_mix.keys())
    probs = list(distribution_mix.values())
    return str(rng.choice(labels, p=probs))


def inject_conjunction(
    primary: GeneratedSatellite,
    capability_template: SatelliteCapability,
    target_tca_s: float,
    target_d_min_m: float,
    crossing_angle_rad: float,
    physics_dt_s: float,
) -> tuple[GeneratedSatellite, float, float]:
    """Build a secondary satellite engineered to approach `primary` at the target geometry.

    Returns (secondary_satellite, achieved_tca_s, achieved_d_min_m).
    """
    if target_tca_s <= 0:
        raise ValueError(f"target_tca_s must be > 0, got {target_tca_s}")
    if target_d_min_m < 0:
        raise ValueError(f"target_d_min_m must be >= 0, got {target_d_min_m}")
    if crossing_angle_rad <= 0:
        raise ValueError(f"crossing_angle_rad must be > 0, got {crossing_angle_rad}")
    if physics_dt_s <= 0:
        raise ValueError(f"physics_dt_s must be > 0, got {physics_dt_s}")

    n_steps = round(target_tca_s / physics_dt_s)
    if n_steps < 1:
        raise ValueError("target_tca_s too small relative to physics_dt_s to align to the physics grid")
    t_star = n_steps * physics_dt_s

    primary_traj = propagate(primary.state.as_vector(), physics_dt_s, n_steps, state_derivative)
    r_i, v_i = primary_traj[-1, :3], primary_traj[-1, 3:]

    r_hat = r_i / np.linalg.norm(r_i)
    v_rot = _rotate_vector(v_i, r_hat, crossing_angle_rad)
    delta_v = v_rot - v_i
    if np.linalg.norm(delta_v) < 1e-6:
        raise ValueError("crossing_angle_rad too small to produce a distinguishable relative velocity")

    n_hat = _perpendicular_unit_vector(delta_v, r_hat)
    r_j_star = r_i + target_d_min_m * n_hat
    v_j_star = v_i + delta_v

    # Time-reversal trick: acceleration is position-only, so forward-propagating
    # the negated-velocity state for t_star seconds and negating the result
    # yields the state at t=0 that reaches (r_j_star, v_j_star) at t_star.
    aux_initial = np.concatenate([r_j_star, -v_j_star])
    aux_traj = propagate(aux_initial, physics_dt_s, n_steps, state_derivative)
    r_j0 = aux_traj[-1, :3]
    v_j0 = -aux_traj[-1, 3:]

    check_traj = propagate(np.concatenate([r_j0, v_j0]), physics_dt_s, n_steps, state_derivative)
    r_j_check = check_traj[-1, :3]
    achieved_d_min_m = float(np.linalg.norm(r_j_check - r_i))

    tolerance_m = max(10.0, 0.15 * target_d_min_m)
    if abs(achieved_d_min_m - target_d_min_m) > tolerance_m:
        raise RuntimeError(
            f"Conjunction injection failed self-check: achieved {achieved_d_min_m:.2f}m, "
            f"target {target_d_min_m:.2f}m (tolerance {tolerance_m:.2f}m)"
        )

    secondary = GeneratedSatellite(
        state=SatelliteState(position_m=tuple(r_j0), velocity_mps=tuple(v_j0)),
        capability=capability_template,
    )
    return secondary, t_star, achieved_d_min_m


def generate_conjunction_scenario(
    n: int,
    seed: int,
    shell: ShellDistribution,
    capability_template: SatelliteCapability,
    episode_horizon_s: float,
    physics_dt_s: float = 10.0,
    distribution_mix: Mapping[str, float] = DEFAULT_DISTRIBUTION_MIX,
) -> ConjunctionScenario:
    """Sample a fresh constellation and layer a randomized conjunction mix on top (PRD §17).

    `sample_constellation`'s own N-range check ([10, 20]) is relied on here
    rather than duplicated -- it already guarantees enough satellites for
    up to 2 disjoint injected pairs (multi_conjunction needs 4 of them).
    """
    satellites = list(sample_constellation(n, seed=seed, shell=shell, capability_template=capability_template))

    rng = np.random.default_rng(seed)
    episode_type = sample_episode_type(rng, distribution_mix)
    n_conjunctions = _EPISODE_TYPE_CONJUNCTION_COUNT[episode_type]

    pool = list(range(n))
    rng.shuffle(pool)

    conjunctions = []
    for k in range(n_conjunctions):
        primary_index = pool[2 * k]
        secondary_index = pool[2 * k + 1]

        target_tca_s = float(rng.uniform(0.1 * episode_horizon_s, 0.9 * episode_horizon_s))
        if rng.random() < _COLLISION_COURSE_FRACTION:
            target_d_min_m = float(rng.uniform(*_COLLISION_COURSE_D_MIN_RANGE_M))
        else:
            target_d_min_m = float(rng.uniform(*_NEAR_MISS_D_MIN_RANGE_M))
        crossing_angle_rad = float(rng.uniform(*_CROSSING_ANGLE_RANGE_RAD))

        secondary_sat, achieved_tca_s, achieved_d_min_m = inject_conjunction(
            satellites[primary_index],
            capability_template,
            target_tca_s,
            target_d_min_m,
            crossing_angle_rad,
            physics_dt_s,
        )
        satellites[secondary_index] = secondary_sat

        conjunctions.append(
            ConjunctionSpec(
                primary_index=primary_index,
                secondary_index=secondary_index,
                target_tca_s=target_tca_s,
                achieved_tca_s=achieved_tca_s,
                target_d_min_m=target_d_min_m,
                achieved_d_min_m=achieved_d_min_m,
                crossing_angle_rad=crossing_angle_rad,
            )
        )

    return ConjunctionScenario(
        satellites=tuple(satellites),
        episode_type=episode_type,
        injected_conjunctions=tuple(conjunctions),
    )
