"""Hardening of the replay data layer after the PR1 review: truncation vs retirement, N=1, comparison wording,
one burn threshold, shield-off lanes, bad inputs, read-only caches and JSON safety."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from orbit_guard.rl.episode_metrics import _BURN_EPSILON_MPS
from orbit_guard.visualization.derived.comparison import comparison_note
from orbit_guard.visualization.derived.partners import closest_partner
from orbit_guard.visualization.derived.status import (
    EVENT_MANEUVER,
    satellite_event_matrix,
    satellite_status_matrix,
)
from orbit_guard.visualization.episodes import discover_episodes
from orbit_guard.visualization.payload import build_payload, partner_dtype
from orbit_guard.visualization.replay import (
    BURN_EPSILON_MPS,
    frame_at,
    load_trace,
    save_trace,
)
from orbit_guard.visualization.replay_meta import (
    ReplayMetaError,
    build_replay_meta,
    validate_replay_meta,
)
from orbit_guard.visualization.source import ReplaySource
from tests.evaluation.replay_fixtures import (
    capability_trace,
    fleet_collision_trace,
    shielded_trace,
    unshielded_trace,
)


def _meta_kwargs(**changes):
    base = {
        "condition": "normal", "seed": 1, "policy": "trained", "shield": True, "model": "m", "n_agents": 2,
        "delta_v_budget_mps": 2.0, "max_delta_v_per_action_mps": 0.1, "weights_sha256": "a" * 64,
    }
    return {**base, **changes}


# ----------------------------------------------------------------------- H1: end of episode is not retirement
def test_every_satellite_still_has_a_partner_on_the_final_frame_when_nothing_collided():
    trace = shielded_trace()
    assert not trace.alive[-1].any()  # the env empties `agents` on truncation; this must not read as "all retired"

    partners = ReplaySource(trace).partners

    assert (partners.partner[-1] != -1).all() and np.isfinite(partners.d_min_m[-1]).all()


def test_survivors_of_a_collision_keep_partners_and_the_collided_pair_loses_them():
    source = ReplaySource(fleet_collision_trace())
    collided_at = int(np.argmax(source.sat_status[:, 0] == 3))
    assert collided_at > 0 and set(np.nonzero(source.sat_status[-1] == 3)[0]) == {0, 2}

    last = source.partners.partner[-1]

    assert last[0] == -1 and last[2] == -1  # retired: stale after impact
    assert last[1] == 3 and last[3] == 1  # the survivors still track each other
    assert source.partners.partner[collided_at - 1, 0] != -1  # before impact the pair did have partners


# ----------------------------------------------------------------------- H2: no pairs (the env needs N >= 2, so this is a guard)
def _slice_to_one_satellite(trace):
    """A real trace cut down to satellite 0: the vectorised layer must not crash on zero pairs."""
    first = slice(0, 1)
    return dataclasses.replace(
        trace,
        positions_m=trace.positions_m[:, first], alive=trace.alive[:, first], remaining_dv_mps=trace.remaining_dv_mps[:, first],
        mission_deviation=trace.mission_deviation[:, first], maneuver_enabled=trace.maneuver_enabled[first],
        pair_index=np.zeros((0, 2), dtype=int), d_min_m=trace.d_min_m[:, :0], d_safe_m=trace.d_safe_m[:, :0], risk=trace.risk[:, :0],
        proposed=trace.proposed[:, first], approved=trace.approved[:, first], applied_dv_mps=trace.applied_dv_mps[:, first],
        intervened=trace.intervened[:, first], collided_pair=trace.collided_pair[:, :0],
    )


def test_status_and_partner_series_handle_a_fleet_without_pairs():
    trace = _slice_to_one_satellite(shielded_trace())

    assert satellite_status_matrix(trace).shape == (trace.num_steps + 1, 1) and not satellite_status_matrix(trace).any()
    assert satellite_event_matrix(trace).shape == (trace.num_steps + 1, 1)
    assert (closest_partner(trace).partner == -1).all()


def test_pair_index_keeps_its_shape_through_save_and_load(tmp_path):
    save_trace(_slice_to_one_satellite(shielded_trace()), tmp_path / "n1.npz")

    assert load_trace(tmp_path / "n1.npz").pair_index.shape == (0, 2)


def test_meta_needs_at_least_two_satellites_like_the_environment():
    with pytest.raises(ReplayMetaError, match="n_agents"):
        validate_replay_meta(build_replay_meta(**_meta_kwargs(n_agents=1)))


# ----------------------------------------------------------------------- H3: comparison wording
@pytest.mark.parametrize(
    ("policy", "value", "other", "expected"),
    [
        ("trained", 12.0, 3818.0, ">99% less than before"),
        ("trained", 0.0, 5.0, "100% less than before"),
        ("trained", 3.0, 100.0, "97% less than before"),
        ("untrained", 1.04, 1.0, "4% more than after"),
        ("untrained", 30.0, 3.0, "10× after"),
        ("untrained", 9.45, 3.0, "3.1× after"),
    ],
)
def test_comparison_notes_never_round_a_nonzero_value_to_nothing(policy, value, other, expected):
    assert comparison_note(policy, value, other) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_comparison_note_refuses_non_finite_inputs(bad):
    assert comparison_note("trained", bad, 5.0) is None
    assert comparison_note("trained", 5.0, bad) is None


# ----------------------------------------------------------------------- M4: one burn threshold
def _with_tiny_burn(trace):
    """A real trace plus one sub-threshold burn (6.9e-17 m/s) where nothing happened."""
    applied = trace.applied_dv_mps.copy()
    step, agent = np.argwhere((applied == 0) & ~trace.intervened)[0]
    applied[step, agent] = 6.9e-17
    return dataclasses.replace(trace, applied_dv_mps=applied), int(step), int(agent)


def test_burn_epsilon_matches_the_environment_constant():
    assert BURN_EPSILON_MPS == _BURN_EPSILON_MPS


def test_a_sub_threshold_burn_is_neither_an_event_nor_counted():
    trace, step, agent = _with_tiny_burn(unshielded_trace())
    baseline = ReplaySource(unshielded_trace())
    source = ReplaySource(trace)

    assert source.events[step + 1, agent] != EVENT_MANEUVER
    assert frame_at(trace, step + 1).satellites[agent].event == "none"
    assert (source.events == EVENT_MANEUVER).sum() == (baseline.events == EVENT_MANEUVER).sum()
    assert source.running.maneuvers[-1] == baseline.running.maneuvers[-1]


# ----------------------------------------------------------------------- M5: shield off is "not applicable"
def test_shield_lane_is_not_applicable_when_the_shield_is_off():
    off, on = build_payload(ReplaySource(unshielded_trace())), build_payload(ReplaySource(shielded_trace()))

    assert off["density"]["2"] is None and off["runs"]["2"] is None
    assert isinstance(on["density"]["2"], list) and isinstance(on["runs"]["2"], list)


# ----------------------------------------------------------------------- M6: usable fuel
def test_fleet_fuel_counts_only_usable_tanks_and_reports_the_stranded_remainder():
    trace = capability_trace()
    usable = trace.maneuver_enabled
    assert (~usable).any(), "fixture must contain an offline thruster"

    fuel = ReplaySource(trace).fuel

    assert fuel.total_mps[-1] == pytest.approx(trace.remaining_dv_mps[-1][usable].sum())
    assert fuel.total_budget_mps == pytest.approx(trace.remaining_dv_mps[0][usable].sum())
    assert fuel.stranded_mps == pytest.approx(trace.remaining_dv_mps[0][~usable].sum())


# ----------------------------------------------------------------------- M8 / M9: bad inputs fail loudly, naming the file
def _write_npz(path, **arrays):
    np.savez(path, **arrays)
    return path


@pytest.mark.parametrize(
    "make",
    [
        lambda p: _write_npz(p, header=np.array("{not json")),
        lambda p: _write_npz(p, header=np.array("[1, 2]")),
        lambda p: _write_npz(p, header=np.array(json.dumps({"meta": None}))),
        lambda p: p.write_bytes(b"this is not a zip archive"),
    ],
    ids=["bad-json", "header-list", "meta-null", "not-a-zip"],
)
def test_corrupt_traces_are_reported_by_file_name(tmp_path, make):
    make(tmp_path / "broken.npz")

    with pytest.raises(ReplayMetaError, match="broken.npz"):
        discover_episodes(tmp_path)


def test_a_directory_named_like_a_trace_is_reported_by_name(tmp_path):
    (tmp_path / "folder.npz").mkdir()

    with pytest.raises(ReplayMetaError, match="folder.npz"):
        discover_episodes(tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"weights_sha256": "a" * 64 + "\n"},
        {"condition": 5, "scenario_id": "5:1"},
        {"model": 7},
        {"delta_v_budget_mps": float("inf")},
        {"policy": "no_maneuver", "weights_sha256": 12},
    ],
)
def test_meta_validation_rejects_bad_types_and_values(changes):
    meta = {**build_replay_meta(**_meta_kwargs()), **changes}

    with pytest.raises(ReplayMetaError):
        validate_replay_meta(meta)


def test_validating_a_missing_meta_is_a_replay_meta_error():
    with pytest.raises(ReplayMetaError):
        validate_replay_meta(None)


def test_meta_that_disagrees_with_the_trace_is_rejected():
    trace = shielded_trace()
    lying = dataclasses.replace(trace, meta={**trace.meta, "n_agents": 99})

    with pytest.raises(ReplayMetaError, match="n_agents"):
        ReplaySource(lying)


# ----------------------------------------------------------------------- M10: conjunction entries for the scrubber ticks
def test_payload_carries_conjunction_entries_separately_from_key_steps():
    source = ReplaySource(fleet_collision_trace())
    payload = build_payload(source)

    assert payload["conjunction_entries"] == sorted(set(payload["conjunction_entries"]))
    assert set(payload["conjunction_entries"]) <= set(payload["key_steps"]) and payload["conjunction_entries"]


# ----------------------------------------------------------------------- M11: caches are read-only
def test_cached_derived_arrays_cannot_be_written_through():
    source = ReplaySource(fleet_collision_trace())

    for array in (source.pair_status, source.sat_status, source.events, source.partners.d_min_m, source.fuel.total_mps, source.density[1]):
        assert not array.flags.writeable


def test_negative_steps_are_rejected_not_wrapped():
    source = ReplaySource(shielded_trace())

    with pytest.raises(IndexError):
        source.fleet_fuel(-1)
    with pytest.raises(IndexError):
        source.closest_partner(-1, 0)
    with pytest.raises(IndexError):
        source.fleet_fuel(source.num_steps + 1)


# ----------------------------------------------------------------------- M12: JSON safety
def test_payload_is_strict_json_even_with_a_non_finite_recorded_summary():
    trace = shielded_trace()
    broken = dataclasses.replace(trace, summary={**trace.summary, "mean_mission_deviation": float("nan")})

    payload = build_payload(ReplaySource(broken))

    json.dumps(payload, allow_nan=False)
    assert payload["summary"]["mean_deviation"] is None


def test_partner_indices_widen_beyond_int8_range():
    assert partner_dtype(15) == "int8" and partner_dtype(127) == "int8"
    assert partner_dtype(200) == "int16"


# ----------------------------------------------------------------------- drift and import guards
def test_catalog_constants_match_the_maneuver_block_of_the_default_config():
    import yaml

    from orbit_guard.safety.maneuver_mapper import (
        DEFAULT_BASE_DELTA_V_MPS,
        DEFAULT_PHASE_FRACTION,
    )

    block = yaml.safe_load((Path(__file__).parents[2] / "configs" / "default.yaml").read_text())["maneuver"]

    assert block["base_delta_v_mps"] == DEFAULT_BASE_DELTA_V_MPS and block["phase_fraction"] == DEFAULT_PHASE_FRACTION


def test_nothing_in_the_replay_app_imports_the_recorder():
    """The recorder pulls in ray, torch and gymnasium; the replay path must stay free of them (FR16)."""
    root = Path(__file__).parents[2] / "orbit_guard" / "visualization"
    offenders = [path.name for path in root.rglob("*.py") if "replay_recording" in path.read_text()]

    assert offenders == []
