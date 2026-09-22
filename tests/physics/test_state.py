from dataclasses import FrozenInstanceError

import pytest

from orbit_guard.physics.state import SatelliteCapability, SatelliteState


def test_satellite_state_round_trips_through_vector():
    state = SatelliteState(
        position_m=(7000000.0, 0.0, 0.0),
        velocity_mps=(0.0, 7500.0, 0.0),
    )
    vec = state.as_vector()
    assert vec.shape == (6,)
    restored = SatelliteState.from_vector(vec)
    assert restored.position_m == pytest.approx(state.position_m)
    assert restored.velocity_mps == pytest.approx(state.velocity_mps)


def test_satellite_state_is_immutable():
    state = SatelliteState(position_m=(1.0, 2.0, 3.0), velocity_mps=(4.0, 5.0, 6.0))
    with pytest.raises(FrozenInstanceError):
        state.position_m = (0.0, 0.0, 0.0)


def test_satellite_state_as_vector_does_not_alias_internal_tuples():
    state = SatelliteState(position_m=(1.0, 2.0, 3.0), velocity_mps=(4.0, 5.0, 6.0))
    vec = state.as_vector()
    vec[0] = 999.0
    assert state.position_m == (1.0, 2.0, 3.0)


def test_satellite_capability_with_remaining_delta_v_returns_new_instance():
    cap = SatelliteCapability(
        max_delta_v_per_action_mps=0.1,
        remaining_delta_v_mps=2.0,
        maneuver_enabled=True,
        mission_deviation_limit=1.0,
    )
    updated = cap.with_remaining_delta_v(1.5)
    assert updated is not cap
    assert updated.remaining_delta_v_mps == 1.5
    assert cap.remaining_delta_v_mps == 2.0  # original untouched


def test_satellite_capability_is_immutable():
    cap = SatelliteCapability(
        max_delta_v_per_action_mps=0.1,
        remaining_delta_v_mps=2.0,
        maneuver_enabled=True,
        mission_deviation_limit=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        cap.remaining_delta_v_mps = 0.0
