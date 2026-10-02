"""Batched propagation must agree with the single-state reference (PRD §10-12)."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.physics.dynamics import state_derivative, state_derivative_batch
from orbit_guard.physics.integrator import propagate, propagate_batch


def _states(count: int = 5) -> np.ndarray:
    rng = np.random.default_rng(7)
    radius_m = 6_840_000.0 + rng.uniform(-5_000, 5_000, count)
    angle = rng.uniform(0, 2 * np.pi, count)
    speed_mps = 7_600.0 + rng.uniform(-10, 10, count)
    inclination = np.radians(53.0)
    positions = np.stack([radius_m * np.cos(angle), radius_m * np.sin(angle) * np.cos(inclination), radius_m * np.sin(angle) * np.sin(inclination)], axis=1)
    velocities = np.stack([-speed_mps * np.sin(angle), speed_mps * np.cos(angle) * np.cos(inclination), speed_mps * np.cos(angle) * np.sin(inclination)], axis=1)
    return np.concatenate([positions, velocities], axis=1)


def test_batch_derivative_matches_the_single_state_derivative_row_by_row():
    states = _states()

    batched = state_derivative_batch(states)

    for row, state in zip(batched, states, strict=True):
        np.testing.assert_allclose(row, state_derivative(state), rtol=1e-12, atol=1e-12)


def test_batch_propagation_matches_single_state_propagation():
    states = _states()

    batched = propagate_batch(states, 10.0, 180, state_derivative_batch)

    assert batched.shape == (5, 181, 6)
    for trajectory, state in zip(batched, states, strict=True):
        np.testing.assert_allclose(trajectory, propagate(state, 10.0, 180, state_derivative), rtol=1e-9, atol=1e-6)


def test_batch_propagation_does_not_mutate_its_input():
    states = _states()
    before = states.copy()

    propagate_batch(states, 10.0, 10, state_derivative_batch)

    np.testing.assert_array_equal(states, before)


def test_batch_propagation_of_a_single_row_matches_the_reference():
    state = _states(1)

    batched = propagate_batch(state, 10.0, 30, state_derivative_batch)

    np.testing.assert_allclose(batched[0], propagate(state[0], 10.0, 30, state_derivative), rtol=1e-9, atol=1e-6)


def test_batch_derivative_fails_loudly_on_non_finite_or_degenerate_states():
    states = _states(2)
    bad = states.copy()
    bad[1, 0] = np.nan
    with pytest.raises(ValueError):
        state_derivative_batch(bad)

    degenerate = states.copy()
    degenerate[0, :3] = 0.0
    with pytest.raises(ValueError):
        state_derivative_batch(degenerate)
