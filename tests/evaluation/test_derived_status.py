"""Whole-episode status/event matrices, per-step density, event runs and key steps (EC-31, EC-10, EC-23).

The vectorised functions must agree with the per-frame `frame_at` (the already-tested source of display
semantics) at EVERY step of every real recorded fixture."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.visualization.derived.density import event_density
from orbit_guard.visualization.derived.key_steps import SPARSE_RUN_LIMIT, key_steps
from orbit_guard.visualization.derived.runs import Run, event_runs
from orbit_guard.visualization.derived.status import (
    EVENT_COLLISION,
    EVENT_MANEUVER,
    EVENT_NAMES,
    EVENT_SHIELD,
    PAIR_STATUS_NAMES,
    SAT_STATUS_NAMES,
    pair_status_matrix,
    satellite_event_matrix,
    satellite_status_matrix,
)
from orbit_guard.visualization.replay import frame_at
from tests.evaluation.replay_fixtures import (
    capability_trace,
    collision_trace,
    depletion_trace,
    shielded_trace,
    unshielded_trace,
)

_TRACES = {
    "collision": collision_trace,
    "shielded": shielded_trace,
    "unshielded": unshielded_trace,
    "capability": capability_trace,
    "depletion": depletion_trace,
}


@pytest.fixture(params=sorted(_TRACES))
def trace(request):
    return _TRACES[request.param]()


def test_pair_status_matches_frame_at_everywhere(trace):
    matrix = pair_status_matrix(trace)
    assert matrix.shape == (trace.num_steps + 1, trace.num_pairs) and matrix.dtype == np.uint8
    for t in range(trace.num_steps + 1):
        expected = [PAIR_STATUS_NAMES.index(p.status) for p in frame_at(trace, t).pairs]
        assert matrix[t].tolist() == expected, t


def test_satellite_status_matches_frame_at_everywhere(trace):
    matrix = satellite_status_matrix(trace)
    for t in range(trace.num_steps + 1):
        expected = [SAT_STATUS_NAMES.index(s.status) for s in frame_at(trace, t).satellites]
        assert matrix[t].tolist() == expected, t


def test_satellite_events_match_frame_at_everywhere(trace):
    matrix = satellite_event_matrix(trace)
    for t in range(trace.num_steps + 1):
        expected = [EVENT_NAMES.index(s.event) for s in frame_at(trace, t).satellites]
        assert matrix[t].tolist() == expected, t


def test_step_zero_has_no_events(trace):
    assert not satellite_event_matrix(trace)[0].any()


def test_collision_is_an_event_on_the_step_into_the_hit_and_status_persists():
    trace = collision_trace()
    hit = int(np.nonzero(trace.collided_pair.any(axis=1))[0][0])
    events = satellite_event_matrix(trace)
    assert events[hit + 1].tolist() == [EVENT_COLLISION, EVENT_COLLISION]
    status = satellite_status_matrix(trace)
    assert (status[hit + 1 :] == 3).all()  # collided stays collided for the rest of the replay


def test_shield_and_maneuver_events_are_distinct_types():
    shielded, plain = satellite_event_matrix(shielded_trace()), satellite_event_matrix(depletion_trace())
    assert (shielded == EVENT_SHIELD).any() and not (shielded == EVENT_MANEUVER).any()  # every burn was shield-changed
    assert (plain == EVENT_MANEUVER).any() and not (plain == EVENT_SHIELD).any()  # shield off: never a shield event


def test_density_counts_satellites_per_event_type_per_step(trace):
    events = satellite_event_matrix(trace)
    density = event_density(events)
    assert set(density) == {EVENT_MANEUVER, EVENT_SHIELD, EVENT_COLLISION}
    for event_type, series in density.items():
        assert series.shape == (trace.num_steps + 1,)
        assert series.tolist() == (events == event_type).sum(axis=1).tolist()


def _events(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


def test_runs_merge_events_within_three_steps_and_split_beyond():
    column = [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]  # steps 1-3, 6 (gap 3 -> merged), 11 (gap 5 -> new run)
    runs = event_runs(_events([[v] for v in column]), pair_index=np.zeros((0, 2), int), collided_pair=np.zeros((11, 0), bool))
    maneuver = runs[EVENT_MANEUVER]
    assert maneuver == [Run(EVENT_MANEUVER, (0,), 1, 6, 4), Run(EVENT_MANEUVER, (0,), 11, 11, 1)]


def test_runs_are_per_satellite_and_sorted_by_start():
    rows = [[0, 0], [0, 1], [1, 0], [1, 0], [0, 1]]  # sat0 events at 2,3 ; sat1 at 1,4
    runs = event_runs(_events(rows), pair_index=np.zeros((0, 2), int), collided_pair=np.zeros((4, 0), bool))
    starts = [r.start for r in runs[EVENT_MANEUVER]]
    assert starts == sorted(starts)
    assert {r.sats for r in runs[EVENT_MANEUVER]} == {(0,), (1,)}


def test_collision_runs_are_one_per_collided_pair_at_the_hit_step():
    trace = collision_trace()
    runs = event_runs(satellite_event_matrix(trace), trace.pair_index, trace.collided_pair)
    assert len(runs[EVENT_COLLISION]) == 1
    run = runs[EVENT_COLLISION][0]
    assert run.sats == (0, 1) and run.start == run.end == int(np.nonzero(trace.collided_pair.any(axis=1))[0][0]) + 1


def test_run_counts_add_up_to_the_events_in_the_matrix(trace):
    events = satellite_event_matrix(trace)
    runs = event_runs(events, trace.pair_index, trace.collided_pair)
    for event_type in (EVENT_MANEUVER, EVENT_SHIELD):
        assert sum(r.count for r in runs[event_type]) == int((events == event_type).sum())


def test_key_steps_include_conjunction_entries_and_collisions():
    trace = collision_trace()
    pair = pair_status_matrix(trace)
    runs = event_runs(satellite_event_matrix(trace), trace.pair_index, trace.collided_pair)
    steps = key_steps(pair, trace.remaining_dv_mps, runs)
    entries = [t for t in range(1, len(pair)) if (pair[t] > pair[t - 1]).any()]
    assert steps == sorted(set(steps)) and set(entries) <= set(steps)
    assert int(np.nonzero(trace.collided_pair.any(axis=1))[0][0]) + 1 in steps


def test_key_steps_include_budget_depletion():
    trace = depletion_trace()
    pair = pair_status_matrix(trace)
    runs = event_runs(satellite_event_matrix(trace), trace.pair_index, trace.collided_pair)
    steps = key_steps(pair, trace.remaining_dv_mps, runs)
    remaining = trace.remaining_dv_mps
    depleted = [t for t in range(1, len(remaining)) if ((remaining[t] <= 1e-9) & (remaining[t - 1] > 1e-9)).any()]
    assert depleted and set(depleted) <= set(steps)


def test_run_starts_are_key_steps_only_when_sparse():
    pair = np.zeros((10, 1), np.uint8)
    remaining = np.ones((10, 1))
    few = {EVENT_MANEUVER: [Run(EVENT_MANEUVER, (0,), 4, 4, 1)], EVENT_SHIELD: [], EVENT_COLLISION: []}
    assert key_steps(pair, remaining, few) == [4]
    many = {EVENT_MANEUVER: [Run(EVENT_MANEUVER, (0,), 1 + n % 9, 1 + n % 9, 1) for n in range(SPARSE_RUN_LIMIT + 1)],
            EVENT_SHIELD: [], EVENT_COLLISION: []}
    assert key_steps(pair, remaining, many) == []


def test_inputs_are_not_mutated(trace):
    before = (trace.risk.copy(), trace.collided_pair.copy(), trace.intervened.copy())
    pair_status_matrix(trace)
    satellite_status_matrix(trace)
    satellite_event_matrix(trace)
    assert all(np.array_equal(a, b) for a, b in zip(before, (trace.risk, trace.collided_pair, trace.intervened), strict=True))
