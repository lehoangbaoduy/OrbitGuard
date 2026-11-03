"""Per-satellite budget provenance (replay fix b): each satellite's DRAWN initial Δv budget, before capability
degradation, and whether the capability_reduction degradation actually cut that satellite's budget.

The replay needs both to tell a satellite that was handed a small draw (normal) from one the shift degraded
(limited); the post-degradation budget alone cannot, once budgets are randomized per episode (CORE-3a).
"""

from __future__ import annotations

import pytest

from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.rl.budget_provenance import BudgetProvenance, budget_provenance
from orbit_guard.rl.environment import CapabilityDegradation, OrbitGuardEnv

_KW = {"episode_horizon_s": 180.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 60.0}
_N = 10


def test_without_degradation_the_drawn_budgets_are_the_initial_budgets_and_nothing_is_degraded():
    env = OrbitGuardEnv(num_agents=_N, **_KW)
    env.reset(seed=4)

    provenance = env.budget_provenance()
    assert isinstance(provenance, BudgetProvenance)
    assert list(provenance.drawn_mps) == list(env.initial_delta_v_budgets_mps().values())
    assert provenance.degraded == (False,) * _N


def test_degradation_marks_exactly_the_reduced_satellites_and_keeps_their_pre_degradation_draw():
    plain = OrbitGuardEnv(num_agents=_N, **_KW)
    degraded = OrbitGuardEnv(
        num_agents=_N, capability_degradation=CapabilityDegradation(disabled_count=2, reduced_count=3), **_KW
    )
    plain.reset(seed=6)
    degraded.reset(seed=6)

    provenance = degraded.budget_provenance()
    initial = list(degraded.initial_delta_v_budgets_mps().values())
    assert provenance.drawn_mps == plain.budget_provenance().drawn_mps, "the draw happens before degradation"
    reduced = [i for i in range(_N) if initial[i] < provenance.drawn_mps[i]]
    assert len(reduced) == 3
    assert [i for i, flag in enumerate(provenance.degraded) if flag] == reduced
    disabled = [i for i, c in enumerate(degraded._capabilities) if not c.maneuver_enabled]
    assert len(disabled) == 2 and not any(provenance.degraded[i] for i in disabled)


def test_a_scale_of_one_degrades_nobody():
    env = OrbitGuardEnv(
        num_agents=_N, capability_degradation=CapabilityDegradation(reduced_count=3, reduced_budget_scale=1.0), **_KW
    )
    env.reset(seed=6)
    assert env.budget_provenance().degraded == (False,) * _N


def test_provenance_is_per_episode():
    env = OrbitGuardEnv(num_agents=_N, **_KW)
    env.reset(seed=1)
    first = env.budget_provenance()
    env.reset(seed=2)
    assert env.budget_provenance().drawn_mps != first.drawn_mps


def test_provenance_values_are_plain_python_scalars():
    env = OrbitGuardEnv(num_agents=_N, capability_degradation=CapabilityDegradation(reduced_count=1), **_KW)
    env.reset(seed=3)
    provenance = env.budget_provenance()
    assert all(type(value) is float for value in provenance.drawn_mps)
    assert all(type(flag) is bool for flag in provenance.degraded)


def test_the_helper_compares_budgets_before_and_after_degradation():
    template = SatelliteCapability(0.1, 2.0, True, 1.0)
    before = [template.with_remaining_delta_v(v) for v in (1.6, 2.4, 0.0)]
    after = [before[0].with_remaining_delta_v(0.4), before[1], before[2]]

    provenance = budget_provenance(before, after)
    assert provenance.drawn_mps == pytest.approx((1.6, 2.4, 0.0))
    assert provenance.degraded == (True, False, False), "an empty draw scaled by any factor is not degraded"


def test_the_helper_rejects_mismatched_fleets():
    template = SatelliteCapability(0.1, 2.0, True, 1.0)
    with pytest.raises(ValueError, match="same satellites"):
        budget_provenance([template], [template, template])
