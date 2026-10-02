"""Steps worth jumping to with previous/next event (EC-23)."""

from __future__ import annotations

import numpy as np

from orbit_guard.visualization.derived.runs import Run
from orbit_guard.visualization.derived.status import EVENT_MANEUVER, EVENT_SHIELD

SPARSE_RUN_LIMIT = 80  # run starts are key steps only when there are this few runs (a dense episode would flood)
DEPLETED_EPS_MPS = 1e-9


def conjunction_entries(pair_status: np.ndarray) -> list[int]:
    """Steps at which any pair's status rises (nominal -> inside d_safe -> high risk -> collision)."""
    if len(pair_status) < 2:
        return []
    rises = (pair_status[1:] > pair_status[:-1]).any(axis=1)
    return (np.nonzero(rises)[0] + 1).tolist()


def key_steps(
    pair_status: np.ndarray,
    remaining_dv_mps: np.ndarray,
    runs: dict[int, list[Run]],
    sparse_run_limit: int = SPARSE_RUN_LIMIT,
) -> list[int]:
    """Sorted unique steps: conjunction entries (any pair's status rises), tank depletions, sparse run starts."""
    steps: set[int] = set()
    steps.update(conjunction_entries(pair_status))
    if len(pair_status) > 1:
        emptied = ((remaining_dv_mps[1:] <= DEPLETED_EPS_MPS) & (remaining_dv_mps[:-1] > DEPLETED_EPS_MPS)).any(axis=1)
        steps.update((np.nonzero(emptied)[0] + 1).tolist())
    for event_type in (EVENT_MANEUVER, EVENT_SHIELD):
        event_runs = runs.get(event_type, [])
        if len(event_runs) <= sparse_run_limit:
            steps.update(run.start for run in event_runs)
    return sorted(steps)
