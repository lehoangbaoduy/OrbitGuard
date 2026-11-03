"""The centralized critic's global state (PRD §28: critic-only, never read by the actor).

Layout, for N agents whose flattened local observation has `slot_dim` entries:

    [slot_0 .. slot_{N-1} | alive_0 .. alive_{N-1} | time_remaining]     (N * slot_dim + N + 1 entries)

* ``slot_i`` -- agent i's last flattened local observation, in `possible_agents` order; ZEROED once the
  agent has collided (terminated), instead of repeating its stale last observation.
* ``alive_i`` -- 1.0 until agent i collides, then 0.0. Truncation at the horizon is not death: a survivor
  keeps its slot and its bit in the final state.
* ``time_remaining`` -- ``1 - steps_taken / horizon_steps``: 1, 1-1/H, ..., 1/H at the H decision steps
  and 0 after the last one.

Why time is here: Ray 2.58's MultiAgentEpisode marks an episode terminated once every agent is done, so
each agent's last row at the horizon is learned as TERMINATED and V(s_T+1) is never bootstrapped. With
time remaining in the critic's input that forced terminal is a Markov finite-horizon terminal. The actor
observation (PRD §24, frozen) does not change.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces


def global_state_dim(num_agents: int, slot_dim: int) -> int:
    if num_agents < 1 or slot_dim < 1:
        raise ValueError(f"need num_agents >= 1 and slot_dim >= 1, got {num_agents}, {slot_dim}")
    return num_agents * slot_dim + num_agents + 1


def global_state_space(num_agents: int, slot_dim: int) -> spaces.Box:
    return spaces.Box(low=-np.inf, high=np.inf, shape=(global_state_dim(num_agents, slot_dim),), dtype=np.float32)


def time_remaining(steps_taken: int, horizon_steps: int) -> float:
    """``1 - steps_taken / horizon_steps``, clipped to [0, 1]."""
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")
    return float(min(1.0, max(0.0, 1.0 - steps_taken / horizon_steps)))


def assemble_global_state(slots: Sequence[np.ndarray], alive: Sequence[bool], remaining: float) -> np.ndarray:
    """Concatenate the slots (a dead agent's zeroed), the alive mask and the time remaining, as float32."""
    if len(slots) != len(alive):
        raise ValueError(f"{len(slots)} slots but {len(alive)} alive flags")
    mask = np.asarray(alive, dtype=np.float32)
    parts = [
        np.asarray(slot, dtype=np.float32) if is_alive else np.zeros(np.shape(slot), dtype=np.float32)
        for slot, is_alive in zip(slots, alive, strict=True)
    ]
    return np.concatenate([*parts, mask, np.array([remaining], dtype=np.float32)]).astype(np.float32)
