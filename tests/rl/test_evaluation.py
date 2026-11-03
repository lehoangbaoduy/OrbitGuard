"""Tests for orbit_guard.rl.evaluation: rollouts and the pre-registered gate criteria."""

from __future__ import annotations

import dataclasses
import math

import pytest

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.evaluation import (
    EvaluationReport,
    GateCriteria,
    evaluate_policy,
    gate_verdict,
    no_maneuver_policy,
)
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_BUDGET_MPS = 2.0


def _env_factory(config: dict) -> OrbitGuardEnv:
    return OrbitGuardEnv(
        num_agents=2, episode_horizon_s=180.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=60.0
    )


def _report(**overrides) -> EvaluationReport:
    base = {
        "episodes": 10,
        "mean_return_per_agent": 100.0,
        "collision_episodes": 1,
        "maneuver_rate": 0.1,
        "threat_maneuver_rate": 0.8,
        "calm_maneuver_rate": 0.05,
        "threat_step_fraction": 0.1,
        "mean_delta_v_used_mps": 0.2,
        "mean_episode_steps": 3.0,
        "mean_budget_fraction_used": 0.1,
        "unsafe_episodes": 0,
    }
    return EvaluationReport(**{**base, **overrides})


def test_no_maneuver_policy_never_maneuvers_and_spends_no_delta_v():
    report = evaluate_policy(no_maneuver_policy, _env_factory, num_episodes=3, seed=0)

    assert report.maneuver_rate == 0.0
    assert report.mean_delta_v_used_mps == 0.0
    assert report.mean_episode_steps == 3.0


def test_always_maneuvering_policy_reports_full_maneuver_rate_and_spends_delta_v():
    def always_raise(agent, observation):
        return int(ManeuverIntent.RAISE_ORBIT)

    report = evaluate_policy(always_raise, _env_factory, num_episodes=2, seed=0)

    assert report.maneuver_rate == 1.0
    assert report.mean_delta_v_used_mps > 0.0


def test_evaluation_is_reproducible_under_a_seed():
    first = evaluate_policy(no_maneuver_policy, _env_factory, num_episodes=2, seed=5)
    second = evaluate_policy(no_maneuver_policy, _env_factory, num_episodes=2, seed=5)

    assert first == second


def test_evaluation_rejects_a_non_positive_episode_count():
    with pytest.raises(ValueError):
        evaluate_policy(no_maneuver_policy, _env_factory, num_episodes=0)


def test_a_well_behaved_policy_passes_every_gate_check():
    verdict = gate_verdict(_report(mean_return_per_agent=110.0, collision_episodes=0), _report())

    assert verdict.passed
    assert all(verdict.checks.values())


def test_burning_the_whole_budget_fails_the_efficiency_check():
    verdict = gate_verdict(_report(mean_delta_v_used_mps=_BUDGET_MPS, mean_budget_fraction_used=1.0), _report())

    assert not verdict.checks["C2_efficiency"]
    assert not verdict.passed


def test_more_collisions_than_the_baseline_fails_the_safety_check():
    verdict = gate_verdict(_report(collision_episodes=3), _report(collision_episodes=1))

    assert not verdict.checks["C1_safety"]


def test_a_lower_return_than_the_baseline_fails_the_return_check():
    verdict = gate_verdict(_report(mean_return_per_agent=90.0), _report(mean_return_per_agent=100.0))

    assert not verdict.checks["C3_return"]


def test_maneuvering_regardless_of_threat_fails_the_conditioning_check():
    indiscriminate = _report(threat_maneuver_rate=0.84, calm_maneuver_rate=0.84)

    assert not gate_verdict(indiscriminate, _report()).checks["C4_conditioning"]


def test_a_frequently_maneuvering_calm_policy_fails_even_with_a_large_ratio():
    twitchy = _report(threat_maneuver_rate=1.0, calm_maneuver_rate=0.4)

    assert not gate_verdict(twitchy, _report()).checks["C4_conditioning"]


def test_gate_criteria_are_overridable_but_default_to_the_preregistered_values():
    defaults = GateCriteria()

    assert dataclasses.astuple(defaults) == (0.25, 2.0, 0.3)


def test_shield_metrics_are_zero_without_a_shield_and_positive_with_one_on_a_conjunction():
    from orbit_guard.rl.environment import OrbitGuardEnv
    from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv

    kwargs = {"episode_horizon_s": 180.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0}

    plain = evaluate_policy(no_maneuver_policy, lambda _c: OrbitGuardEnv(num_agents=2, **kwargs), 1, seed=10)
    shielded = evaluate_policy(no_maneuver_policy, lambda _c: ShieldedOrbitGuardEnv(num_agents=2, **kwargs), 1, seed=10)

    assert plain.shield_intervention_rate == 0.0
    assert plain.shield_infeasible_step_rate == 0.0
    assert shielded.shield_intervention_rate > 0.0
    assert 0.0 <= shielded.shield_infeasible_step_rate <= 1.0


def _fixed_budget_factory(*, remaining_delta_v_mps: float = _BUDGET_MPS, degradation=None):
    from orbit_guard.physics.state import SatelliteCapability
    from orbit_guard.rl.environment import CapabilityDegradation

    template = SatelliteCapability(
        max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=remaining_delta_v_mps,
        maneuver_enabled=True, mission_deviation_limit=1.0,
    )
    reduce_one = CapabilityDegradation(reduced_count=1, reduced_budget_scale=0.25) if degradation else None

    def factory(config: dict) -> OrbitGuardEnv:
        return OrbitGuardEnv(
            num_agents=2, episode_horizon_s=180.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=60.0,
            capability_template=template, capability_degradation=reduce_one,
        )

    return factory


def _always_raise(agent, observation):
    return int(ManeuverIntent.RAISE_ORBIT)


def test_budget_fraction_is_each_satellites_use_over_its_own_initial_budget_averaged():
    report = evaluate_policy(_always_raise, _fixed_budget_factory(degradation=True), num_episodes=1, seed=0)

    # Three 0.1 m/s burns each: 0.3 of the full 2.0 budget, and 0.3 of the reduced 0.5 one.
    assert report.mean_delta_v_used_mps == pytest.approx(0.3)
    assert report.mean_budget_fraction_used == pytest.approx((0.3 / 2.0 + 0.3 / 0.5) / 2)


def test_budget_fraction_is_zero_for_a_policy_that_never_burns():
    report = evaluate_policy(no_maneuver_policy, _fixed_budget_factory(degradation=True), num_episodes=2, seed=0)

    assert report.mean_budget_fraction_used == 0.0


def test_satellites_with_no_initial_budget_are_excluded_from_the_fraction():
    report = evaluate_policy(_always_raise, _fixed_budget_factory(remaining_delta_v_mps=0.0), num_episodes=1, seed=0)

    assert report.mean_delta_v_used_mps == 0.0
    assert math.isnan(report.mean_budget_fraction_used), "no satellite has a budget to take a fraction of"


def test_unsafe_episodes_count_episodes_with_an_unsafe_conjunction_but_not_collisions():
    from orbit_guard.sensing.noise import NoiseModel

    def factory(config: dict) -> OrbitGuardEnv:
        return OrbitGuardEnv(
            num_agents=2, episode_horizon_s=2400.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=1800.0,
            noise_model=NoiseModel(0.0, 0.0), hard_body_radius_m=100.0,
        )

    # tests/rl/test_episode_metrics.py: seed 5 is a no-op near miss inside the margin, seed 3 a no-op collision.
    report = evaluate_policy(no_maneuver_policy, factory, num_episodes=2, seeds=[5, 3])

    assert report.unsafe_episodes == 1
    assert report.collision_episodes == 1


def test_efficiency_check_uses_the_fraction_of_each_satellites_own_budget():
    small_mean_but_most_of_a_reduced_budget = _report(mean_delta_v_used_mps=0.3, mean_budget_fraction_used=0.6)

    assert not gate_verdict(small_mean_but_most_of_a_reduced_budget, _report()).checks["C2_efficiency"]


@pytest.mark.parametrize("used_mps", [0.0, 0.3, 0.5, 0.5000001, 1.2, _BUDGET_MPS])
def test_efficiency_check_matches_the_old_absolute_test_when_every_budget_is_equal(used_mps):
    criteria = GateCriteria()
    report = _report(mean_delta_v_used_mps=used_mps, mean_budget_fraction_used=used_mps / _BUDGET_MPS)

    old_check = used_mps <= criteria.max_delta_v_fraction * _BUDGET_MPS
    assert gate_verdict(report, _report(), criteria).checks["C2_efficiency"] == old_check
