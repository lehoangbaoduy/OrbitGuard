"""Running metrics for the summary strip's "T+n" lines.

Each series ends exactly on the episode summary the environment recorded (tests/evaluation/test_derived_series.py
pins that for every fixture), so the strip's total and its running value are one number seen two ways.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.visualization.replay import BURN_EPSILON_MPS, EpisodeTrace


@dataclass(frozen=True)
class RunningMetrics:
    delta_v_mps: np.ndarray  # (T+1,) cumulative applied delta-v
    maneuvers: np.ndarray  # (T+1,) cumulative applied burns (includes burns the shield rewrote)
    shield: np.ndarray | None  # (T+1,) cumulative interventions; None when the shield is off ("—", never 0)
    collisions: np.ndarray  # (T+1,) distinct collided pairs so far
    mean_deviation: np.ndarray  # (T+1,) running mean mission deviation; NaN before the first sample


def _with_zero(series: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=series.dtype), series])


def running_metrics(trace: EpisodeTrace, *, shield_enabled: bool) -> RunningMetrics:
    applied = trace.applied_dv_mps
    ever_collided = np.logical_or.accumulate(trace.collided_pair, axis=0) if trace.num_steps else trace.collided_pair
    # The environment averages deviation over agents still alive at each of steps 0..T-1.
    samples = np.cumsum((trace.mission_deviation[:-1] * trace.alive[:-1]).sum(axis=1))
    counts = np.cumsum(trace.alive[:-1].sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_deviation = np.where(counts > 0, samples / np.maximum(counts, 1), np.nan)
    return RunningMetrics(
        delta_v_mps=_with_zero(np.cumsum(applied.sum(axis=1))),
        maneuvers=_with_zero(np.cumsum((applied > BURN_EPSILON_MPS).sum(axis=1)).astype(np.int64)),
        shield=_with_zero(np.cumsum(trace.intervened.sum(axis=1)).astype(np.int64)) if shield_enabled else None,
        collisions=_with_zero(ever_collided.sum(axis=1).astype(np.int64)),
        mean_deviation=np.concatenate([[np.nan], mean_deviation]),
    )
