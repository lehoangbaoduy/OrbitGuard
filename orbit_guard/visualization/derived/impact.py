"""Where the replay draws a satellite after it collides: stopped at the impact point (display only).

In the simulation both bodies of a collision keep orbiting as debris, so their recorded positions run on. The replay
shows them where they hit instead. The impact point of a pair is its midpoint at the closest approach within the
colliding step, with both positions linearly interpolated between the step's two frames (an orbital arc of one control
step bends only a few km, far below the drawing scale). From the next frame on, both satellites sit at that point.
The trace itself is never modified; every total and distance still comes from the recorded data.
"""

from __future__ import annotations

import numpy as np

from orbit_guard.visualization.replay import EpisodeTrace


def _closest_approach_midpoint(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    relative, change = a0 - b0, (a1 - a0) - (b1 - b0)
    span = float(change @ change)
    s = 0.0 if span == 0.0 else float(np.clip(-(relative @ change) / span, 0.0, 1.0))
    return ((a0 + s * (a1 - a0)) + (b0 + s * (b1 - b0))) / 2.0


def impact_display_positions(trace: EpisodeTrace) -> np.ndarray:
    """(T+1, N, 3) positions in metres to draw: recorded ones, except each collided satellite stays at its first impact
    point from the frame after that collision. A new array; the trace is untouched."""
    shown = np.array(trace.positions_m, dtype=float, copy=True)
    stopped: set[int] = set()
    steps, pairs = np.nonzero(trace.collided_pair)
    for step, pair in sorted(zip(steps.tolist(), pairs.tolist())):
        i, j = (int(k) for k in trace.pair_index[pair])
        moving = {i, j} - stopped  # a satellite in several collisions stops at its first one
        if not moving:
            continue
        p = trace.positions_m
        point = _closest_approach_midpoint(p[step, i], p[step + 1, i], p[step, j], p[step + 1, j])
        for k in moving:
            shown[step + 1 :, k] = point
        stopped |= moving
    return shown
