"""`ReplaySource`: the only interface the UI talks to (handoff prompt, "ReplaySource layer"; real-data rule 1)."""

from __future__ import annotations

import dataclasses

import pytest

from orbit_guard.safety.maneuver_mapper import intent_catalog
from orbit_guard.visualization.derived.status import (
    EVENT_COLLISION,
    EVENT_MANEUVER,
    EVENT_SHIELD,
)
from orbit_guard.visualization.replay import frame_at
from orbit_guard.visualization.replay_meta import ReplayMetaError
from orbit_guard.visualization.source import ReplaySource
from tests.evaluation.replay_fixtures import (
    capability_trace,
    collision_trace,
    depletion_trace,
    shielded_trace,
)


def test_source_exposes_typed_meta_and_dimensions():
    source = ReplaySource(shielded_trace())
    assert source.meta.policy == "trained" and source.meta.shield is True
    assert (source.num_steps, source.num_agents, source.num_pairs) == (30, 2, 1)


def test_frame_at_delegates_to_the_replay_semantics():
    trace = shielded_trace()
    source = ReplaySource(trace)
    for t in (0, 5, trace.num_steps):
        assert source.frame_at(t) == frame_at(trace, t)


def test_legacy_trace_without_replay_meta_is_rejected_with_a_hint():
    legacy = dataclasses.replace(shielded_trace(), meta={"model": "demo", "condition": "normal", "seed": 1})
    with pytest.raises(ReplayMetaError, match="record_replay_traces"):
        ReplaySource(legacy)


def test_summary_is_the_recorded_totals():
    trace = shielded_trace()
    summary = ReplaySource(trace).summary
    assert summary.delta_v_mps == pytest.approx(trace.summary["total_delta_v_mps"])
    assert summary.maneuvers == trace.summary["maneuver_count"]
    assert summary.shield == trace.summary["shield_intervention_count"]
    assert summary.collisions == trace.summary["collision_count"]
    assert summary.mean_deviation == pytest.approx(trace.summary["mean_mission_deviation"])


def test_shield_total_is_none_when_the_shield_is_off():
    assert ReplaySource(depletion_trace()).summary.shield is None


def test_density_runs_and_key_steps_are_consistent_with_each_other():
    source = ReplaySource(shielded_trace())
    assert source.density[EVENT_SHIELD].sum() == sum(r.count for r in source.runs[EVENT_SHIELD])
    assert source.density[EVENT_MANEUVER].sum() == sum(r.count for r in source.runs[EVENT_MANEUVER])
    assert source.key_steps == sorted(set(source.key_steps)) and all(0 < s <= source.num_steps for s in source.key_steps)


def test_collision_is_reflected_in_density_and_summary():
    source = ReplaySource(collision_trace())
    assert source.density[EVENT_COLLISION].sum() == 2 and source.summary.collisions == 1


def test_fleet_fuel_at_a_step():
    source = ReplaySource(depletion_trace())
    start, end = source.fleet_fuel(0), source.fleet_fuel(source.num_steps)
    assert start.total_mps == pytest.approx(2.24) and end.total_mps == pytest.approx(0.0)
    assert (start.empty_count, end.empty_count) == (0, 3) and end.usable_count == 3
    assert start.remaining_mps.shape == (3,)


def test_closest_partner_returns_none_when_there_is_no_live_partner():
    source = ReplaySource(collision_trace())
    first = source.closest_partner(0, 0)
    assert first is not None and first.partner == 1 and first.d_min_m > 0
    assert source.closest_partner(source.num_steps, 0) is None


def test_capability_flags_and_start_step():
    source = ReplaySource(capability_trace())
    assert source.capability.offline.tolist() == [False, False, False, True]
    assert 0 <= source.start_step <= source.num_steps


def test_catalog_uses_the_traces_own_per_action_delta_v():
    trace = shielded_trace()
    source = ReplaySource(trace)
    assert source.catalog == intent_catalog(trace.meta["max_delta_v_per_action_mps"])


def test_derived_values_are_computed_once():
    source = ReplaySource(shielded_trace())
    assert source.events is source.events and source.runs is source.runs
