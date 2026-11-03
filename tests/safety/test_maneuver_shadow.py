"""Tests for orbit_guard.safety.maneuver_shadow: the noise-free maneuver deviation the shield's MISSION_LIMIT uses.

Real physics: the maneuvered shadow is the reset shadow propagated with the
satellite's own applied burns, so mocking propagation would test nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.physics.dynamics import state_derivative_batch
from orbit_guard.physics.integrator import propagate_batch
from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v
from orbit_guard.safety.maneuver_shadow import (
    advance_maneuvered_shadows,
    maneuver_mission_deviations,
    no_maneuver_shadows,
)

_DT_S = 10.0
_SUBSTEPS = 6
_MAX_DV = 0.1
_WEIGHTS = {"q_a": 0.25, "q_e": 0.25, "q_i": 0.25, "q_m": 0.25}
_R_M = 6_371_000.0 + 463_000.0
_V_MPS = float(np.sqrt(3.986004418e14 / _R_M))


def _shadow_states(n: int) -> tuple[np.ndarray, ...]:
    """Circular LEO states, spread in phase."""
    states = []
    for k in range(n):
        angle = 0.3 * k
        position = _R_M * np.array([np.cos(angle), np.sin(angle), 0.0])
        velocity = _V_MPS * np.array([-np.sin(angle), np.cos(angle) * 0.6, np.cos(angle) * 0.8])
        states.append(np.concatenate([position, velocity]))
    return tuple(states)


def _advance_shadows(shadows):
    trajectories = propagate_batch(np.array(shadows), _DT_S, _SUBSTEPS, state_derivative_batch)
    return tuple(trajectory[-1] for trajectory in trajectories)


def _advance(maneuvered, shadows, intents, applied):
    return advance_maneuvered_shadows(
        maneuvered, shadows, intents, applied, (_MAX_DV,) * len(shadows),
        phase_fraction=0.5, physics_dt_s=_DT_S, substeps=_SUBSTEPS,
    )


def _run(schedule, n: int = 3):
    """Advance shadows and maneuvered shadows through `schedule` = [(intents, applied norms), ...]."""
    shadows = _shadow_states(n)
    maneuvered = no_maneuver_shadows(n)
    for intents, applied in schedule:
        maneuvered = _advance(maneuvered, shadows, intents, applied)
        shadows = _advance_shadows(shadows)
    return maneuvered, shadows


def test_zero_burns_give_exactly_zero_maneuver_deviation():
    schedule = [((0, 0, 0), (0.0, 0.0, 0.0))] * 5

    maneuvered, shadows = _run(schedule)

    assert maneuvered == (None, None, None)
    assert maneuver_mission_deviations(maneuvered, shadows, _WEIGHTS) == (0.0, 0.0, 0.0)


def test_a_disabled_or_budget_exhausted_request_with_zero_applied_burn_adds_no_deviation():
    schedule = [((int(ManeuverIntent.RAISE_ORBIT), 0, 0), (0.0, 0.0, 0.0))]

    maneuvered, shadows = _run(schedule)

    assert maneuver_mission_deviations(maneuvered, shadows, _WEIGHTS) == (0.0, 0.0, 0.0)


def test_a_burn_moves_only_that_satellite_off_its_shadow():
    raise_orbit = int(ManeuverIntent.RAISE_ORBIT)
    schedule = [((0, raise_orbit, 0), (0.0, _MAX_DV, 0.0))] + [((0, 0, 0), (0.0, 0.0, 0.0))] * 3

    maneuvered, shadows = _run(schedule)
    deviations = maneuver_mission_deviations(maneuvered, shadows, _WEIGHTS)

    assert deviations[0] == 0.0 and deviations[2] == 0.0
    assert deviations[1] > 0.0
    assert maneuvered[0] is None and maneuvered[2] is None


def test_the_burn_is_mapped_from_the_shadow_state_and_scaled_to_the_applied_norm():
    shadows = _shadow_states(1)
    intent = int(ManeuverIntent.PHASE_FORWARD)
    applied = 0.02  # e.g. the budget cap cut the 0.05 m/s phase burn

    maneuvered = _advance(no_maneuver_shadows(1), shadows, (intent,), (applied,))

    direction = map_intent_to_delta_v(intent, shadows[0][:3], shadows[0][3:6], base_delta_v_mps=_MAX_DV, phase_fraction=0.5)
    start = shadows[0].copy()
    start[3:6] += direction * (applied / np.linalg.norm(direction))
    expected = propagate_batch(start[None, :], _DT_S, _SUBSTEPS, state_derivative_batch)[0, -1]
    np.testing.assert_array_equal(maneuvered[0], expected)


def test_accumulated_burns_grow_the_deviation():
    raise_orbit = int(ManeuverIntent.RAISE_ORBIT)
    one = _run([((raise_orbit,), (_MAX_DV,))], n=1)
    five = _run([((raise_orbit,), (_MAX_DV,))] * 5, n=1)

    (deviation_one,) = maneuver_mission_deviations(*one, _WEIGHTS)
    (deviation_five,) = maneuver_mission_deviations(*five, _WEIGHTS)

    assert deviation_five > 3.0 * deviation_one > 0.0


def test_inputs_are_not_mutated():
    shadows = _shadow_states(2)
    shadow_copies = tuple(s.copy() for s in shadows)
    maneuvered = _advance(no_maneuver_shadows(2), shadows, (int(ManeuverIntent.RAISE_ORBIT), 0), (_MAX_DV, 0.0))
    maneuvered_copy = maneuvered[0].copy()

    _advance(maneuvered, shadows, (int(ManeuverIntent.LOWER_ORBIT), 0), (_MAX_DV, 0.0))

    for state, copy in zip(shadows, shadow_copies, strict=True):
        np.testing.assert_array_equal(state, copy)
    np.testing.assert_array_equal(maneuvered[0], maneuvered_copy)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        _advance(no_maneuver_shadows(2), _shadow_states(3), (0, 0, 0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        maneuver_mission_deviations(no_maneuver_shadows(2), _shadow_states(3), _WEIGHTS)


def test_negative_applied_norm_is_rejected():
    with pytest.raises(ValueError):
        _advance(no_maneuver_shadows(1), _shadow_states(1), (int(ManeuverIntent.RAISE_ORBIT),), (-0.1,))
