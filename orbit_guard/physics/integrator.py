"""Fixed-step RK4 integrator with configurable dt (PRD §11-12).

Fails loudly (raises) if any propagated state becomes non-finite, rather than
silently continuing with an invalid state.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

StateDerivative = Callable[[np.ndarray], np.ndarray]


def rk4_step(
    state_vector: np.ndarray, dt: float, derivative_fn: StateDerivative
) -> np.ndarray:
    """One fixed-step classical RK4 update. Returns a new array; never mutates the input."""
    k1 = derivative_fn(state_vector)
    k2 = derivative_fn(state_vector + 0.5 * dt * k1)
    k3 = derivative_fn(state_vector + 0.5 * dt * k2)
    k4 = derivative_fn(state_vector + dt * k3)
    new_state = state_vector + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.all(np.isfinite(new_state)):
        raise ValueError(f"RK4 step produced a non-finite state: {new_state}")
    return new_state


def propagate(
    state_vector: np.ndarray,
    dt: float,
    n_steps: int,
    derivative_fn: StateDerivative,
) -> np.ndarray:
    """Propagate for n_steps, returning the trajectory including the initial state.

    Shape: (n_steps + 1, len(state_vector)).
    """
    trajectory = np.empty((n_steps + 1, state_vector.shape[0]), dtype=np.float64)
    trajectory[0] = state_vector
    current = state_vector
    for step in range(1, n_steps + 1):
        current = rk4_step(current, dt, derivative_fn)
        trajectory[step] = current
    return trajectory
