"""Small-N (N < 10) scenario construction for Phase 3's 2-agent sanity gate.

`sample_constellation` and `generate_conjunction_scenario` (Phase 2) both
enforce `MIN_SUPPORTED_N = 10` -- correct for the project's primary N=10-20
benchmark, but PRD §44 Phase 3 explicitly requires a 2-SATELLITE sanity
check before scaling up at all. Neither Phase 2 function is modified here
(their own floor and tests stay exactly as frozen); this module reuses the
same per-satellite sampling statistics (`constellation._sample_one_satellite`)
and the same conjunction-injection geometry (`inject_conjunction`) without
that floor, specifically for `rl/environment.py`'s small-N construction.

`multi_conjunction` needs 4 disjoint satellites (2 injected pairs) and is
therefore unsupported below N=4 -- the distribution mix is renormalized
over the remaining episode types in that case, proportionally, so the
relative weighting of what remains is unchanged (e.g. the default
30/50/20 mix with multi_conjunction dropped becomes 37.5/62.5 benign/single).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.scenarios.conjunction_generator import (
    _COLLISION_COURSE_D_MIN_RANGE_M,
    _COLLISION_COURSE_FRACTION,
    _CROSSING_ANGLE_RANGE_RAD,
    _NEAR_MISS_D_MIN_RANGE_M,
    DEFAULT_DISTRIBUTION_MIX,
    ConjunctionScenario,
    ConjunctionSpec,
    inject_conjunction,
    sample_episode_type,
)
from orbit_guard.scenarios.constellation import ShellDistribution, _sample_one_satellite

_MIN_N_FOR_MULTI_CONJUNCTION = 4


def _renormalize_mix_without_multi_conjunction(
    distribution_mix: Mapping[str, float],
) -> dict[str, float]:
    remaining = {k: v for k, v in distribution_mix.items() if k != "multi_conjunction"}
    total = sum(remaining.values())
    if total <= 0:
        raise ValueError("distribution_mix has no mass left after dropping multi_conjunction")
    return {k: v / total for k, v in remaining.items()}


def sample_small_n_satellites(
    n: int, seed: int, shell: ShellDistribution, capability_template: SatelliteCapability
):
    """N (>= 1) satellites via the same per-satellite statistics as `sample_constellation`."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    return tuple(_sample_one_satellite(rng, shell, capability_template) for _ in range(n))


def generate_small_n_conjunction_scenario(
    n: int,
    seed: int,
    shell: ShellDistribution,
    capability_template: SatelliteCapability,
    episode_horizon_s: float,
    physics_dt_s: float = 10.0,
    distribution_mix: Mapping[str, float] = DEFAULT_DISTRIBUTION_MIX,
) -> ConjunctionScenario:
    """Small-N counterpart of `conjunction_generator.generate_conjunction_scenario`.

    Identical episode-type/injection logic, just without the N>=10 floor.
    `multi_conjunction` is dropped (and the mix renormalized) below N=4.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 for a conjunction scenario, got {n}")

    effective_mix = (
        _renormalize_mix_without_multi_conjunction(distribution_mix)
        if n < _MIN_N_FOR_MULTI_CONJUNCTION
        else distribution_mix
    )

    satellites = list(sample_small_n_satellites(n, seed, shell, capability_template))

    rng = np.random.default_rng(seed)
    episode_type = sample_episode_type(rng, effective_mix)
    n_conjunctions = {"benign": 0, "single_conjunction": 1, "multi_conjunction": 2}[episode_type]

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
