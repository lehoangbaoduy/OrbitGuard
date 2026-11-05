"""Collided satellites stop at their impact point in the replay (2026-11-04).

In the simulation the two bodies of a collision keep orbiting as debris; the replay shows them where they hit, so the
✕ markers stop instead of flying on. Display only: the recorded trace and every total are untouched. The impact point
is the midpoint of the pair at the moment of their closest approach inside the colliding step (positions linearly
interpolated between the two frames), so both satellites stop at one shared point.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.visualization.derived.impact import impact_display_positions
from orbit_guard.visualization.payload import build_payload, decode_array
from orbit_guard.visualization.source import ReplaySource
from tests.evaluation.replay_fixtures import fleet_collision_trace, shielded_trace

COLLIDED, SURVIVORS = (0, 2), (1, 3)  # fleet_collision_trace: satellites 0 and 2 collide, 1 and 3 survive


def _impact_step(trace) -> int:
    steps, _pairs = np.nonzero(trace.collided_pair)
    return int(steps.min())


def test_collided_satellites_stay_at_one_shared_point_from_the_frame_after_impact():
    trace = fleet_collision_trace()
    shown = impact_display_positions(trace)
    after = _impact_step(trace) + 1

    for i in COLLIDED:
        assert np.all(shown[after:, i] == shown[after, i])  # no longer moving
    np.testing.assert_array_equal(shown[after, COLLIDED[0]], shown[after, COLLIDED[1]])
    assert not np.allclose(trace.positions_m[after:, COLLIDED[0]], trace.positions_m[after, COLLIDED[0]])  # the debris did move


def test_before_impact_and_for_survivors_the_recorded_positions_are_shown():
    trace = fleet_collision_trace()
    shown = impact_display_positions(trace)
    after = _impact_step(trace) + 1

    np.testing.assert_array_equal(shown[:after], trace.positions_m[:after])
    np.testing.assert_array_equal(shown[:, list(SURVIVORS)], trace.positions_m[:, list(SURVIVORS)])


def test_the_impact_point_is_the_pairs_midpoint_at_their_closest_approach_within_the_step():
    trace = fleet_collision_trace()
    k = _impact_step(trace)
    i, j = COLLIDED
    point = impact_display_positions(trace)[k + 1, i]
    s = np.linspace(0.0, 1.0, 2001)[:, None]
    a = trace.positions_m[k, i] + s * (trace.positions_m[k + 1, i] - trace.positions_m[k, i])
    b = trace.positions_m[k, j] + s * (trace.positions_m[k + 1, j] - trace.positions_m[k, j])
    separation = np.linalg.norm(a - b, axis=1)
    best = int(separation.argmin())

    assert np.linalg.norm(point - (a[best] + b[best]) / 2.0) < 1e-3 * max(separation.max(), 1.0)
    assert separation[best] <= min(separation[0], separation[-1])


def test_the_recorded_trace_is_never_modified():
    trace = fleet_collision_trace()
    before = trace.positions_m.copy()

    impact_display_positions(trace)

    np.testing.assert_array_equal(trace.positions_m, before)


def test_an_episode_without_a_collision_is_shown_as_recorded():
    trace = shielded_trace()

    np.testing.assert_array_equal(impact_display_positions(trace), trace.positions_m)


def test_the_browser_payload_carries_the_stopped_positions():
    trace = fleet_collision_trace()

    positions = decode_array(build_payload(ReplaySource(trace))["arrays"]["positions_km"])

    np.testing.assert_allclose(positions, impact_display_positions(trace) / 1000.0, rtol=1e-6)


def test_the_legend_says_collided_satellites_are_shown_where_they_hit():
    pytest.importorskip("dash")
    from orbit_guard.visualization.layout import SCALE_CAPTION

    assert "impact point" in SCALE_CAPTION and "debris keeps orbiting" in SCALE_CAPTION
