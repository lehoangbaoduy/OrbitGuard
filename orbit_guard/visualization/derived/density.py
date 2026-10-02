"""Per-step event density: how many satellites show each event type at each step (EC-10)."""

from __future__ import annotations

import numpy as np

from orbit_guard.visualization.derived.status import (
    EVENT_COLLISION,
    EVENT_MANEUVER,
    EVENT_SHIELD,
)


def event_density(events: np.ndarray) -> dict[int, np.ndarray]:
    """{event type: (T+1,) count of satellites with that event at each step}."""
    return {
        event_type: (events == event_type).sum(axis=1).astype(np.int32)
        for event_type in (EVENT_MANEUVER, EVENT_SHIELD, EVENT_COLLISION)
    }
