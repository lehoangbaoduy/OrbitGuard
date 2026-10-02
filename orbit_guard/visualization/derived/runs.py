"""Event runs: consecutive events of one satellite merged into a single mark (EC-10, EC-11)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.visualization.derived.status import (
    EVENT_COLLISION,
    EVENT_MANEUVER,
    EVENT_SHIELD,
)

RUN_MERGE_GAP_STEPS = 3  # events up to this many steps apart belong to one run


@dataclass(frozen=True)
class Run:
    type: int
    sats: tuple[int, ...]
    start: int
    end: int
    count: int


def _satellite_runs(events: np.ndarray, event_type: int, max_gap: int) -> list[Run]:
    runs: list[Run] = []
    for satellite in range(events.shape[1]):
        current: Run | None = None
        for step in np.nonzero(events[:, satellite] == event_type)[0].tolist():
            if current is not None and step - current.end <= max_gap:
                current = Run(event_type, (satellite,), current.start, step, current.count + 1)
                runs[-1] = current
            else:
                current = Run(event_type, (satellite,), step, step, 1)
                runs.append(current)
    return sorted(runs, key=lambda run: (run.start, run.sats))


def _collision_runs(pair_index: np.ndarray, collided_pair: np.ndarray) -> list[Run]:
    runs = []
    for pair in range(collided_pair.shape[1]):
        hits = np.nonzero(collided_pair[:, pair])[0]
        if hits.size:
            step = int(hits[0]) + 1  # a hit in step t is shown on frame t + 1
            runs.append(Run(EVENT_COLLISION, (int(pair_index[pair, 0]), int(pair_index[pair, 1])), step, step, 1))
    return sorted(runs, key=lambda run: (run.start, run.sats))


def event_runs(
    events: np.ndarray, pair_index: np.ndarray, collided_pair: np.ndarray, max_gap: int = RUN_MERGE_GAP_STEPS
) -> dict[int, list[Run]]:
    """{event type: runs sorted by start}; collisions are one run per collided pair."""
    return {
        EVENT_MANEUVER: _satellite_runs(events, EVENT_MANEUVER, max_gap),
        EVENT_SHIELD: _satellite_runs(events, EVENT_SHIELD, max_gap),
        EVENT_COLLISION: _collision_runs(pair_index, collided_pair),
    }
