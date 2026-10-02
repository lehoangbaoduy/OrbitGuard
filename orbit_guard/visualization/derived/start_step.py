"""Where a replay opens: derived from the data, never a hand-picked constant (decision D7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from orbit_guard.visualization.derived.status import STATUS_COLLISION, retired_matrix
from orbit_guard.visualization.replay import EpisodeTrace


def peak_risk_step(trace: EpisodeTrace, pair_status: np.ndarray) -> int | None:
    """Step of the highest live-pair risk (pairs with a retired satellite no longer count); None if risk is zero throughout."""
    if not trace.num_pairs:
        return None
    retired = retired_matrix(trace, pair_status)
    stale = (pair_status == STATUS_COLLISION) | retired[:, trace.pair_index[:, 0]] | retired[:, trace.pair_index[:, 1]]
    risk = np.where(stale, 0.0, trace.risk).max(axis=1)
    return int(risk.argmax()) if risk.max() > 0 else None


def default_start_step(trace: EpisodeTrace, pair_status: np.ndarray, key_steps: Sequence[int]) -> int:
    """Peak-risk step, else the first key step, else the beginning."""
    peak = peak_risk_step(trace, pair_status)
    if peak is not None:
        return peak
    return int(key_steps[0]) if key_steps else 0
