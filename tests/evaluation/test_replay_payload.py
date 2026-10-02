"""The browser payload (decision D2): everything the JS draws, computed in Python, JSON-safe and compact (EC-32,
EC-26, EC-25). Arrays travel as base64 typed buffers so a 361-step N=15 episode stays near a megabyte."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orbit_guard.visualization.payload import build_payload, build_strip, decode_array
from orbit_guard.visualization.replay import load_trace
from orbit_guard.visualization.source import ReplaySource
from orbit_guard.visualization.summary import SummaryTotals
from tests.evaluation.replay_fixtures import (
    capability_trace,
    collision_trace,
    depletion_trace,
    full_size_trace,
    shielded_trace,
)

REAL_TRACES = Path.home() / "orbitguard_runs" / "replay_traces"
PAYLOAD_BUDGET_BYTES = 1_500_000


def _payload(trace, **kwargs):
    return build_payload(ReplaySource(trace), **kwargs)


def test_payload_is_strict_json():
    for make in (shielded_trace, collision_trace, depletion_trace, capability_trace):
        json.dumps(_payload(make()), allow_nan=False)  # NaN/inf would raise


def test_arrays_round_trip_with_dtype_and_shape():
    trace = shielded_trace()
    payload = _payload(trace)
    positions = decode_array(payload["arrays"]["positions_km"])
    assert positions.dtype == np.float32 and positions.shape == trace.positions_m.shape
    np.testing.assert_allclose(positions, trace.positions_m / 1000.0, rtol=1e-6)
    assert "alive" not in payload["arrays"]  # `alive` is all False on a truncated final frame; retirement travels as sat_status
    for name in ("sat_status", "events", "pair_status"):
        assert decode_array(payload["arrays"][name]).dtype == np.uint8
    assert decode_array(payload["arrays"]["proposed"]).shape == trace.proposed.shape


def test_payload_carries_meta_dims_catalog_and_threshold():
    payload = _payload(shielded_trace())
    assert payload["meta"]["policy"] == "trained" and payload["meta"]["shield"] is True
    assert payload["dims"] == {"steps": 30, "agents": 2, "pairs": 1, "control_dt_s": 60.0}
    assert [c["label"] for c in payload["catalog"]][:2] == ["NO MANEUVER", "RAISE ORBIT"]
    assert payload["thresholds"]["high_risk"] == 0.5


def test_not_applicable_values_are_none_not_zero():
    payload = _payload(depletion_trace())  # shield off
    assert payload["summary"]["shield"] is None and payload["running"]["shield"] is None
    assert payload["running"]["mean_deviation"][0] is None  # no sample before the first step
    assert isinstance(payload["running"]["maneuvers"][-1], int)


def test_stale_partner_distances_after_impact_are_none():
    payload = _payload(collision_trace())
    partner_d = decode_array(payload["arrays"]["partner_d_min_m"])
    assert np.isnan(partner_d[-1]).all()  # NaN travels inside the binary buffer, never inside JSON


def test_runs_density_and_key_steps_are_plain_lists():
    payload = _payload(shielded_trace())
    assert payload["key_steps"] == sorted(payload["key_steps"])
    assert set(payload["density"]) == {"1", "2", "3"} and len(payload["density"]["2"]) == 31
    assert {"sats", "start", "end", "count"} <= set(payload["runs"]["2"][0])


def test_capability_flags_are_index_lists():
    payload = _payload(capability_trace())
    assert payload["capability"] == {"offline": [3], "limited": [0]}


def test_start_step_is_in_range():
    payload = _payload(shielded_trace())
    assert 0 <= payload["start_step"] <= 30


# ------------------------------------------------------------------ strip
def _totals(dv, man, shield, coll, dev):
    return SummaryTotals(delta_v_mps=dv, maneuvers=man, shield=shield, collisions=coll, mean_deviation=dev)


def test_strip_pairs_before_and_after_with_notes():
    strip = build_strip("untrained", {"untrained": _totals(30.0, 400, 3000, 0, 0.7), "trained": _totals(3.0, 40, 12, 0, 0.2)})
    cell = strip["delta_v_mps"]
    assert (cell["before"], cell["after"], cell["note"]) == (30.0, 3.0, "10× after")
    assert strip["collisions"]["note"] is None  # zero denominator: no ratio


def test_strip_after_tab_reports_percent_less():
    strip = build_strip("trained", {"untrained": _totals(30.0, 400, 3000, 0, 0.7), "trained": _totals(3.0, 40, 12, 0, 0.2)})
    assert strip["delta_v_mps"]["note"] == "90% less than before"


def test_strip_marks_a_missing_counterpart_as_none_without_a_note():
    strip = build_strip("trained", {"untrained": None, "trained": _totals(3.0, 40, 12, 0, 0.2)})
    assert strip["delta_v_mps"]["before"] is None and strip["delta_v_mps"]["note"] is None
    assert strip["delta_v_mps"]["after"] == 3.0


def test_strip_shield_cells_are_none_when_either_shield_is_off():
    strip = build_strip("trained", {"untrained": _totals(30.0, 400, None, 0, 0.7), "trained": _totals(3.0, 40, None, 0, 0.2)})
    assert strip["shield"] == {"before": None, "after": None, "note": None}


def test_payload_includes_the_strip_and_reference_note_only_when_given():
    plain = _payload(shielded_trace())
    assert plain["strip"] is None and plain["reference_note"] is None
    given = _payload(shielded_trace(), strip={"x": 1}, reference_note="ref")
    assert given["strip"] == {"x": 1} and given["reference_note"] == "ref"


@pytest.mark.skipif(not REAL_TRACES.is_dir(), reason="real N=15 traces not recorded on this machine")
def test_real_n15_payload_stays_within_the_size_budget():
    for path in sorted(REAL_TRACES.glob("*.npz"))[:3]:
        payload = build_payload(ReplaySource(load_trace(path)))
        assert len(json.dumps(payload)) < PAYLOAD_BUDGET_BYTES, path.name


def test_full_size_payload_stays_within_the_size_budget_without_any_local_files():
    trace = full_size_trace()
    assert (trace.num_agents, trace.num_steps) == (15, 360)

    size = len(json.dumps(_payload(trace), allow_nan=False))

    assert size < PAYLOAD_BUDGET_BYTES


def test_payload_ships_the_vocabulary_and_the_earth_radius_from_the_system():
    from orbit_guard.physics.constants import R_EARTH_EQUATORIAL_M
    from orbit_guard.visualization.vocabulary import vocabulary_payload

    payload = _payload(shielded_trace())

    assert payload["vocab"] == vocabulary_payload()
    assert payload["constants"]["earth_radius_km"] == R_EARTH_EQUATORIAL_M / 1000.0
