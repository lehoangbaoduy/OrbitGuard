"""Fleet fuel, closest partner, running metrics, capability flags, start step, comparison copy (EC-07, EC-25, EC-35).

Where a recorded `summary` exists the derived running series must END exactly on it: the strip's totals and its
running "T+n" values are two views of one number."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from orbit_guard.visualization.derived.capability import capability_flags
from orbit_guard.visualization.derived.comparison import comparison_note
from orbit_guard.visualization.derived.fuel import fleet_fuel
from orbit_guard.visualization.derived.metrics import running_metrics
from orbit_guard.visualization.derived.partners import closest_partner
from orbit_guard.visualization.derived.start_step import (
    default_start_step,
    peak_risk_step,
)
from orbit_guard.visualization.derived.status import pair_status_matrix
from tests.evaluation.replay_fixtures import (
    capability_trace,
    collision_trace,
    depletion_trace,
    shielded_trace,
    unshielded_trace,
)


# ---------------------------------------------------------------- fleet fuel
def test_fuel_budgets_are_each_satellites_own_starting_tank():
    fuel = fleet_fuel(depletion_trace())
    assert fuel.budget_mps.tolist() == pytest.approx([0.12, 2.0, 0.12])
    assert fuel.total_budget_mps == pytest.approx(2.24)
    assert fuel.total_mps[0] == pytest.approx(2.24)


def test_fuel_totals_never_increase():
    for make in (depletion_trace, shielded_trace, capability_trace):
        total = fleet_fuel(make()).total_mps
        assert (np.diff(total) <= 1e-12).all()


def test_empty_tanks_are_counted_and_the_all_empty_step_found():
    trace = depletion_trace()
    fuel = fleet_fuel(trace)
    assert fuel.usable_count == 3
    assert fuel.empty_count[0] == 0 and fuel.empty_count[-1] == 3
    first_full = int(np.nonzero(fuel.empty_count == 3)[0][0])
    assert fuel.all_empty_step == first_full
    assert (fuel.empty_count[first_full:] == 3).all()


def test_all_empty_step_is_none_when_tanks_never_run_dry():
    assert fleet_fuel(shielded_trace()).all_empty_step is None


def test_offline_thrusters_are_not_counted_as_usable_or_empty():
    trace = capability_trace()
    fuel = fleet_fuel(trace)
    assert fuel.usable_count == 3  # one of four is offline
    assert not fuel.empty[:, 3].any()


# ---------------------------------------------------------- closest partner
def _collided_by(trace, t):
    """Satellites in a pair that has collided on or before the step INTO frame t (independent of `alive`, which is
    all False on the final frame of any truncated episode)."""
    collided = set()
    for k, (i, j) in enumerate(trace.pair_index):
        if t >= 1 and trace.collided_pair[:t, k].any():
            collided |= {int(i), int(j)}
    return collided


def _oracle(trace, t, agent):
    """Slow loop version of the rule: closest pair involving `agent`, neither satellite retired by a collision."""
    status = pair_status_matrix(trace)
    retired = _collided_by(trace, t)
    best = None
    for k, (i, j) in enumerate(trace.pair_index):
        if agent not in (i, j) or status[t, k] == 3 or int(i) in retired or int(j) in retired:
            continue
        if best is None or trace.d_min_m[t, k] < best[0]:
            best = (float(trace.d_min_m[t, k]), float(trace.d_safe_m[t, k]), int(j if agent == i else i))
    return best


@pytest.mark.parametrize("make", [shielded_trace, capability_trace, depletion_trace, collision_trace])
def test_closest_partner_matches_a_slow_oracle_at_every_step(make):
    trace = make()
    partner = closest_partner(trace)
    for t in range(trace.num_steps + 1):
        for agent in range(trace.num_agents):
            expected = _oracle(trace, t, agent)
            if expected is None:
                assert np.isnan(partner.d_min_m[t, agent]) and partner.partner[t, agent] == -1
            else:
                assert partner.d_min_m[t, agent] == pytest.approx(expected[0])
                assert partner.d_safe_m[t, agent] == pytest.approx(expected[1])
                assert partner.partner[t, agent] == expected[2]


def test_collided_pair_has_no_partner_after_impact():
    trace = collision_trace()
    hit = int(np.nonzero(trace.collided_pair.any(axis=1))[0][0]) + 1
    partner = closest_partner(trace)
    assert (partner.partner[hit:] == -1).all() and np.isnan(partner.d_min_m[hit:]).all()
    assert (partner.partner[:hit] != -1).all()


# ----------------------------------------------------------- running metrics
@pytest.mark.parametrize("make", [shielded_trace, unshielded_trace, capability_trace, depletion_trace, collision_trace])
def test_running_metrics_end_exactly_on_the_recorded_summary(make):
    trace = make()
    shield = trace.meta["shield"]
    running = running_metrics(trace, shield_enabled=shield)
    summary = trace.summary
    assert running.delta_v_mps[-1] == pytest.approx(summary["total_delta_v_mps"])
    assert running.maneuvers[-1] == summary["maneuver_count"]
    assert running.collisions[-1] == summary["collision_count"]
    assert running.mean_deviation[-1] == pytest.approx(summary["mean_mission_deviation"], abs=1e-12)
    if shield:
        assert running.shield[-1] == summary["shield_intervention_count"]


def test_shield_series_is_none_not_zero_when_the_shield_is_off():
    assert running_metrics(depletion_trace(), shield_enabled=False).shield is None
    assert running_metrics(shielded_trace(), shield_enabled=True).shield is not None


def test_running_counts_start_at_zero_and_never_decrease():
    running = running_metrics(shielded_trace(), shield_enabled=True)
    for series in (running.delta_v_mps, running.maneuvers, running.shield, running.collisions):
        assert series[0] == 0 and (np.diff(series) >= 0).all()


def test_running_mean_deviation_has_no_value_before_the_first_step():
    running = running_metrics(shielded_trace(), shield_enabled=True)
    assert np.isnan(running.mean_deviation[0]) and not np.isnan(running.mean_deviation[1:]).any()


def test_collision_counts_each_pair_once_even_if_it_stays_collided():
    trace = collision_trace()
    doubled = trace.collided_pair.copy()
    doubled[np.nonzero(doubled.any(axis=1))[0][0] + 1 :, 0] = True  # the pair "collides" on every later step too
    running = running_metrics(dataclasses.replace(trace, collided_pair=doubled), shield_enabled=False)
    assert running.collisions[-1] == 1


# ---------------------------------------------------------- capability flags
def test_offline_and_limited_flags_come_from_enabled_and_starting_budget():
    trace = capability_trace()
    flags = capability_flags(trace, budget_mps=trace.meta["delta_v_budget_mps"])
    assert flags.offline.tolist() == [False, False, False, True]
    assert flags.limited.tolist() == [True, False, False, False]


def test_an_offline_satellite_is_never_also_limited():
    trace = capability_trace()
    reduced = dataclasses.replace(trace, remaining_dv_mps=trace.remaining_dv_mps * 0.1)  # everyone below budget
    flags = capability_flags(reduced, budget_mps=trace.meta["delta_v_budget_mps"])
    assert not (flags.offline & flags.limited).any() and flags.limited[:3].all()


def test_no_flags_on_an_undegraded_fleet():
    trace = shielded_trace()
    flags = capability_flags(trace, budget_mps=trace.meta["delta_v_budget_mps"])
    assert not flags.offline.any() and not flags.limited.any()


# ---------------------------------------------------------------- start step
def test_peak_risk_step_is_the_argmax_over_live_pairs():
    trace = shielded_trace()
    status = pair_status_matrix(trace)
    risk = np.where(status == 3, 0.0, trace.risk)
    assert peak_risk_step(trace, status) == int(risk.max(axis=1).argmax())


def test_collision_after_impact_does_not_count_toward_peak_risk():
    trace = collision_trace()
    status = pair_status_matrix(trace)
    hit = int(np.nonzero(trace.collided_pair.any(axis=1))[0][0]) + 1
    inflated = trace.risk.copy()
    inflated[hit:] = 9.0
    assert peak_risk_step(dataclasses.replace(trace, risk=inflated), status) < hit


def test_default_start_step_falls_back_to_first_key_step_then_zero():
    trace = shielded_trace()
    quiet = dataclasses.replace(trace, risk=np.zeros_like(trace.risk))
    status = pair_status_matrix(quiet)
    assert peak_risk_step(quiet, status) is None
    assert default_start_step(quiet, status, [7, 9]) == 7
    assert default_start_step(quiet, status, []) == 0
    assert default_start_step(trace, pair_status_matrix(trace), [7, 9]) == peak_risk_step(trace, pair_status_matrix(trace))


# ------------------------------------------------------------ comparison copy
def test_before_tab_note_states_how_many_times_after():
    assert comparison_note("untrained", 30.0, 3.0) == "10× after"
    assert comparison_note("untrained", 7.0, 2.0) == "3.5× after"


def test_after_tab_note_states_the_percent_less_than_before():
    assert comparison_note("trained", 3.0, 30.0) == "90% less than before"


def test_after_tab_note_is_honest_when_after_is_worse_or_equal():
    assert comparison_note("trained", 45.0, 30.0) == "50% more than before"
    assert comparison_note("trained", 30.0, 30.0) == "same as before"


@pytest.mark.parametrize("policy,value,other", [("trained", 5.0, 0.0), ("untrained", 5.0, 0.0), ("trained", None, 3.0),
                                                ("trained", 3.0, None), ("no_maneuver", 3.0, 5.0), ("rule_based", 1.0, 2.0)])
def test_no_note_when_the_denominator_is_zero_a_trace_is_missing_or_policy_is_a_baseline(policy, value, other):
    assert comparison_note(policy, value, other) is None
