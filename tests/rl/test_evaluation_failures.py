"""evaluate_policy survives a failing episode: logged, counted in failed_episodes, excluded from every mean."""

from __future__ import annotations

import dataclasses
import math

import pytest

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.evaluation import EvaluationReport, evaluate_policy, gate_verdict
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_BAD_SEED = 1
_FAIL_ON_STEP = 2  # mid-episode, after some steps were already counted


class _FlakyEnv(OrbitGuardEnv):
    """Raises on its second step of the episode seeded `_BAD_SEED`; behaves normally otherwise."""

    def reset(self, seed=None, options=None):
        self._flaky_seed, self._flaky_steps = seed, 0
        return super().reset(seed=seed, options=options)

    def step(self, actions):
        self._flaky_steps += 1
        if self._flaky_seed == _BAD_SEED and self._flaky_steps == _FAIL_ON_STEP:
            raise FloatingPointError("propagator diverged")
        return super().step(actions)


def _flaky_factory(config: dict) -> OrbitGuardEnv:
    return _FlakyEnv(num_agents=2, episode_horizon_s=180.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=60.0)


def _clean_factory(config: dict) -> OrbitGuardEnv:
    return OrbitGuardEnv(num_agents=2, episode_horizon_s=180.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=60.0)


def _observation_policy(agent, observation) -> int:
    """Stateless (so two runs see the same choices); one satellite always burns, so the rates are non-trivial."""
    if agent.endswith("_0"):
        return int(ManeuverIntent.RAISE_ORBIT)
    return int(ManeuverIntent.NO_MANEUVER)


def test_a_failing_episode_is_counted_and_excluded_from_every_mean():
    flaky = evaluate_policy(_observation_policy, _flaky_factory, 3, seeds=[0, _BAD_SEED, 2])
    clean = evaluate_policy(_observation_policy, _clean_factory, 2, seeds=[0, 2])

    assert flaky.episodes == 3, "episodes counts every attempted seed, so a resumed grid still matches it"
    assert flaky.failed_episodes == 1
    assert flaky.ok_episodes == 2
    assert dataclasses.replace(flaky, episodes=2, failed_episodes=0) == clean


def test_a_failing_episode_is_logged_to_stderr_with_its_seed_and_the_error(capsys):
    evaluate_policy(_observation_policy, _flaky_factory, 3, seeds=[0, _BAD_SEED, 2])

    err = capsys.readouterr().err
    assert f"seed {_BAD_SEED}" in err
    assert "FloatingPointError" in err and "propagator diverged" in err


def test_a_clean_evaluation_reports_zero_failed_episodes():
    report = evaluate_policy(_observation_policy, _clean_factory, 2, seeds=[0, 2])

    assert report.failed_episodes == 0
    assert report.ok_episodes == report.episodes == 2


def test_an_evaluation_whose_every_episode_failed_reports_nan_means_instead_of_raising():
    report = evaluate_policy(_observation_policy, _flaky_factory, 1, seeds=[_BAD_SEED])

    assert report.episodes == report.failed_episodes == 1
    assert report.ok_episodes == 0
    assert report.collision_episodes == 0 and report.unsafe_episodes == 0
    assert math.isnan(report.mean_return_per_agent)
    assert math.isnan(report.mean_budget_fraction_used)


def test_keyboard_interrupt_is_not_swallowed_as_a_failed_episode():
    class _Interrupted(OrbitGuardEnv):
        def step(self, actions):
            raise KeyboardInterrupt

    def factory(config):
        return _Interrupted(num_agents=2, episode_horizon_s=180.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=60.0)

    with pytest.raises(KeyboardInterrupt):
        evaluate_policy(_observation_policy, factory, 1, seeds=[0])


def test_failed_episodes_defaults_to_zero_and_leaves_the_gate_verdict_unchanged():
    base = {
        "episodes": 10, "mean_return_per_agent": 100.0, "collision_episodes": 1, "maneuver_rate": 0.1,
        "threat_maneuver_rate": 0.8, "calm_maneuver_rate": 0.05, "threat_step_fraction": 0.1,
        "mean_delta_v_used_mps": 0.2, "mean_episode_steps": 3.0, "mean_budget_fraction_used": 0.1, "unsafe_episodes": 0,
    }
    trained = EvaluationReport(**{**base, "mean_return_per_agent": 110.0, "collision_episodes": 0})
    baseline = EvaluationReport(**base)

    assert trained.failed_episodes == 0
    assert gate_verdict(trained, baseline).passed


def _gate_report(**overrides) -> EvaluationReport:
    base = {
        "episodes": 10, "mean_return_per_agent": 110.0, "collision_episodes": 0, "maneuver_rate": 0.1,
        "threat_maneuver_rate": 0.8, "calm_maneuver_rate": 0.05, "threat_step_fraction": 0.1,
        "mean_delta_v_used_mps": 0.2, "mean_episode_steps": 3.0, "mean_budget_fraction_used": 0.1, "unsafe_episodes": 0,
    }
    return EvaluationReport(**{**base, **overrides})


def test_c1_compares_collision_rates_over_finished_episodes_when_episodes_failed():
    baseline = _gate_report(collision_episodes=2, mean_return_per_agent=100.0)  # 2 of 10
    crashing = _gate_report(collision_episodes=2, failed_episodes=6)  # 2 of the 4 that finished

    assert not gate_verdict(crashing, baseline).checks["C1_safety"], (
        "crashing on seeds must not make a policy look at least as safe as the baseline"
    )


def test_c1_fails_when_the_trained_policy_finished_no_episode():
    baseline = _gate_report(collision_episodes=2, mean_return_per_agent=100.0)
    crashed = _gate_report(collision_episodes=0, failed_episodes=10)

    assert not gate_verdict(crashed, baseline).checks["C1_safety"]


def test_c1_still_compares_counts_when_nothing_failed():
    baseline = _gate_report(collision_episodes=2, mean_return_per_agent=100.0)

    assert gate_verdict(_gate_report(collision_episodes=2), baseline).checks["C1_safety"]
    assert not gate_verdict(_gate_report(collision_episodes=3), baseline).checks["C1_safety"]
