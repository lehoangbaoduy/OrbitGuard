"""Whole-episode status and event matrices (vectorised twins of `replay.frame_at`).

`frame_at` stays the source of display semantics; tests/evaluation/test_derived_status.py proves these matrices
equal it at every step. Codes index the name tuples below (the same order as the UI's severity ranking).
"""

from __future__ import annotations

import numpy as np

from orbit_guard.visualization.replay import (
    BURN_EPSILON_MPS,
    HIGH_RISK_THRESHOLD,
    EpisodeTrace,
)

PAIR_STATUS_NAMES = ("nominal", "inside_d_safe", "high_risk", "collision")
SAT_STATUS_NAMES = ("nominal", "inside_d_safe", "high_risk", "collided")
EVENT_NAMES = ("none", "maneuver", "shield_intervention", "collision")
STATUS_NOMINAL, STATUS_INSIDE, STATUS_HIGH_RISK, STATUS_COLLISION = 0, 1, 2, 3
EVENT_NONE, EVENT_MANEUVER, EVENT_SHIELD, EVENT_COLLISION = 0, 1, 2, 3


def pair_status_matrix(trace: EpisodeTrace) -> np.ndarray:
    """(T+1, P) pair status codes; a collision stays a collision for the rest of the replay."""
    status = np.zeros((trace.num_steps + 1, trace.num_pairs), dtype=np.uint8)
    status[trace.d_min_m < trace.d_safe_m] = STATUS_INSIDE
    status[trace.risk > HIGH_RISK_THRESHOLD] = STATUS_HIGH_RISK
    if trace.num_steps:
        collided_by_step = np.logical_or.accumulate(trace.collided_pair, axis=0)  # collided_pair[t] shows up in frame t+1
        status[1:][collided_by_step] = STATUS_COLLISION
    return status


def _scatter_max(shape: tuple[int, int], pair_index: np.ndarray, values: np.ndarray, *, boolean: bool) -> np.ndarray:
    """Per satellite, the worst of `values` over the pairs it belongs to."""
    out = np.zeros(shape, dtype=bool if boolean else np.uint8)
    if not len(pair_index):  # a single satellite has no pairs
        return out
    ufunc = np.logical_or if boolean else np.maximum
    for column in (0, 1):
        ufunc.at(out, (slice(None), pair_index[:, column]), values)
    return out


def satellite_status_matrix(trace: EpisodeTrace, pair_status: np.ndarray | None = None) -> np.ndarray:
    """(T+1, N) satellite status codes: the worst status of any pair the satellite belongs to."""
    pairs = pair_status_matrix(trace) if pair_status is None else pair_status
    return _scatter_max((trace.num_steps + 1, trace.num_agents), trace.pair_index, pairs, boolean=False)


def satellite_event_matrix(trace: EpisodeTrace) -> np.ndarray:
    """(T+1, N) event codes for the step INTO each frame: collision > shield intervention > maneuver."""
    events = np.zeros((trace.num_steps + 1, trace.num_agents), dtype=np.uint8)
    if not trace.num_steps:
        return events
    step_events = np.where(trace.intervened, EVENT_SHIELD, np.where(trace.applied_dv_mps > BURN_EPSILON_MPS, EVENT_MANEUVER, EVENT_NONE))
    colliding = _scatter_max((trace.num_steps, trace.num_agents), trace.pair_index, trace.collided_pair, boolean=True)
    step_events = np.where(colliding, EVENT_COLLISION, step_events)
    events[1:] = step_events
    return events


def retired_matrix(trace: EpisodeTrace, pair_status: np.ndarray | None = None) -> np.ndarray:
    """(T+1, N) True from the frame a satellite is in a collided pair (it is retired from then on).

    This is deliberately not `~trace.alive`: the environment empties `agents` when the episode is truncated, so
    `alive` is False for every satellite on the final frame although nothing collided.
    """
    return satellite_status_matrix(trace, pair_status) == STATUS_COLLISION
