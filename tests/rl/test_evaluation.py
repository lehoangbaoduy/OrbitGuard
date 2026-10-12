"""Tests for orbit_guard.rl.evaluation: rollouts and the pre-registered gate criteria."""

from __future__ import annotations

import dataclasses

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
    verdict = gate_verdict(_report(mean_return_per_agent=110.0, collision_episodes=0), _report(), _BUDGET_MPS)

    assert verdict.passed
    assert all(verdict.checks.values())


def test_burning_the_whole_budget_fails_the_efficiency_check():
    verdict = gate_verdict(_report(mean_delta_v_used_mps=_BUDGET_MPS), _report(), _BUDGET_MPS)

    assert not verdict.checks["C2_efficiency"]
    assert not verdict.passed


def test_more_collisions_than_the_baseline_fails_the_safety_check():
    verdict = gate_verdict(_report(collision_episodes=3), _report(collision_episodes=1), _BUDGET_MPS)

    assert not verdict.checks["C1_safety"]


def test_a_lower_return_than_the_baseline_fails_the_return_check():
    verdict = gate_verdict(_report(mean_return_per_agent=90.0), _report(mean_return_per_agent=100.0), _BUDGET_MPS)

    assert not verdict.checks["C3_return"]


def test_maneuvering_regardless_of_threat_fails_the_conditioning_check():
    indiscriminate = _report(threat_maneuver_rate=0.84, calm_maneuver_rate=0.84)

    assert not gate_verdict(indiscriminate, _report(), _BUDGET_MPS).checks["C4_conditioning"]


def test_a_frequently_maneuvering_calm_policy_fails_even_with_a_large_ratio():
    twitchy = _report(threat_maneuver_rate=1.0, calm_maneuver_rate=0.4)

    assert not gate_verdict(twitchy, _report(), _BUDGET_MPS).checks["C4_conditioning"]


def test_gate_criteria_are_overridable_but_default_to_the_preregistered_values():
    defaults = GateCriteria()

    assert dataclasses.astuple(defaults) == (0.25, 2.0, 0.3)
