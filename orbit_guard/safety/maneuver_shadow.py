"""Maneuvered shadows: the noise-free mission deviation a satellite's OWN burns caused (shield MISSION_LIMIT input).

The base environment keeps one no-maneuver shadow per satellite (PRD §25): the
noisy reset belief snapshot, propagated with zero delta-v. Its belief-vs-shadow
deviation feeds R_mission and the observation (Q14), and it is mostly SENSING
NOISE: under the noise_increase shift, satellites that never burned exceed the
1.0 mission limit from noise alone. The shield's hard limit must not read it.

A maneuvered shadow starts from the same shadow state the first time the
satellite actually burns. It is then propagated exactly like the shadow, with
each applied burn added at the start of its control step. The accumulated
maneuver deviation is `compute_weighted_mission_deviation(maneuvered, shadow)`.
Both states are deterministic given the reset snapshot, and no belief or ground
truth enters after reset. Each burn is re-mapped from the maneuvered-shadow
state (its own RTN frame), using the approved intent, and scaled to the APPLIED
norm. The applied norm is the budget-capped value, and it is 0 for a disabled
satellite. The applied ECI vector is never used, because the env maps it from
the true state.

A satellite that has never burned has no maneuvered shadow (None), so its
deviation is exactly 0.0, not a roundoff residue. All functions are pure: they
return new tuples and never modify their inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from orbit_guard.physics.dynamics import state_derivative_batch
from orbit_guard.physics.integrator import propagate_batch
from orbit_guard.physics.mission_deviation import (
    compute_mission_deviation,
    compute_weighted_mission_deviation,
)
from orbit_guard.safety.maneuver_mapper import map_intent_to_delta_v

ManeuveredShadows = tuple[np.ndarray | None, ...]

_ZERO_BURN_MPS = 1.0e-12


def no_maneuver_shadows(num_agents: int) -> ManeuveredShadows:
    """The reset value: no satellite has burned yet."""
    return (None,) * num_agents


def _burn(state: np.ndarray, intent: int, applied_mps: float, max_delta_v_mps: float, phase_fraction: float) -> np.ndarray:
    """`intent` mapped from `state`'s own frame, rescaled to the applied norm."""
    direction = map_intent_to_delta_v(
        intent, state[:3], state[3:6], base_delta_v_mps=max_delta_v_mps, phase_fraction=phase_fraction
    )
    norm = float(np.linalg.norm(direction))
    if norm < _ZERO_BURN_MPS:
        return np.zeros(3)
    return direction * (applied_mps / norm)


def advance_maneuvered_shadows(
    maneuvered: ManeuveredShadows,
    shadow_states: Sequence[np.ndarray],
    approved_intents: Sequence[int],
    applied_delta_v_mps: Sequence[float],
    max_delta_v_per_action_mps: Sequence[float],
    *,
    phase_fraction: float,
    physics_dt_s: float,
    substeps: int,
) -> ManeuveredShadows:
    """Advance every maneuvered shadow by one control step, applying this step's burns at its start.

    `shadow_states` are the no-maneuver shadows at the START of the step. A satellite's first burn starts its
    maneuvered shadow from that state. A satellite with no maneuvered shadow and no applied burn stays None.
    """
    n = len(maneuvered)
    if not (len(shadow_states) == len(approved_intents) == len(applied_delta_v_mps) == len(max_delta_v_per_action_mps) == n):
        raise ValueError(f"every per-satellite input needs {n} entries")
    if any(float(applied) < 0.0 for applied in applied_delta_v_mps):
        raise ValueError("applied delta-v norms must be >= 0")

    starts: dict[int, np.ndarray] = {}
    for idx in range(n):
        applied = float(applied_delta_v_mps[idx])
        if maneuvered[idx] is None and applied < _ZERO_BURN_MPS:
            continue
        base = np.array(shadow_states[idx] if maneuvered[idx] is None else maneuvered[idx], dtype=np.float64)
        if applied >= _ZERO_BURN_MPS:
            base[3:6] += _burn(base, int(approved_intents[idx]), applied, float(max_delta_v_per_action_mps[idx]), phase_fraction)
        starts[idx] = base
    if not starts:
        return tuple(maneuvered)

    order = sorted(starts)
    trajectories = propagate_batch(np.array([starts[idx] for idx in order]), physics_dt_s, substeps, state_derivative_batch)
    advanced = {idx: trajectory[-1].copy() for idx, trajectory in zip(order, trajectories, strict=True)}
    return tuple(advanced.get(idx, maneuvered[idx]) for idx in range(n))


def maneuver_mission_deviations(
    maneuvered: ManeuveredShadows,
    shadow_states: Sequence[np.ndarray],
    mission_weights: Mapping[str, float],
) -> tuple[float, ...]:
    """Weighted mission deviation (PRD §25) of each maneuvered shadow from its no-maneuver shadow, same instant."""
    if len(maneuvered) != len(shadow_states):
        raise ValueError(f"need one shadow state per satellite ({len(maneuvered)}); got {len(shadow_states)}")
    return tuple(
        0.0 if state is None
        else compute_weighted_mission_deviation(compute_mission_deviation(state, shadow), **mission_weights)
        for state, shadow in zip(maneuvered, shadow_states, strict=True)
    )
