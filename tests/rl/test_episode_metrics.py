"""Tests for per-episode outcome metrics and failure labels (PRD §34, §36, §40).

Scenarios (N=2, zero sensing noise, 100 m hard-body radius -> deterministic geometry):
  seed 3  : collides under all-NO_MANEUVER at control step 10
  seed 5  : closest approach ~287 m -- inside any margin, but no collision
  seed 14 (N=4) is not used here.
"""

from __future__ import annotations

import pytest

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel

_KW = {
    "episode_horizon_s": 2400.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0,
    "noise_model": NoiseModel(0.0, 0.0), "hard_body_radius_m": 100.0,
}
_COLLIDING_SEED = 3
_NEAR_MISS_SEED = 5


def _run_no_op(env_class, seed, **overrides):
    env = env_class(num_agents=2, **{**_KW, **overrides})
    env.reset(seed=seed)
    while env.agents:
        env.step({agent: 0 for agent in env.agents})
    return env.episode_summary()


def test_unshielded_collision_is_counted_and_labelled():
    summary = _run_no_op(OrbitGuardEnv, _COLLIDING_SEED)

    assert summary.collision_count == 1
    assert summary.total_delta_v_mps == 0.0
    assert summary.maneuver_count == 0
    assert "physical_collision" in summary.failure_labels
    assert summary.shield_intervention_count == 0


def test_near_miss_inside_the_margin_is_an_unsafe_conjunction_not_a_collision():
    summary = _run_no_op(OrbitGuardEnv, _NEAR_MISS_SEED)

    assert summary.collision_count == 0
    assert summary.unsafe_conjunction_count == 1
    assert "unsafe_conjunction_remained" in summary.failure_labels
    assert "physical_collision" not in summary.failure_labels


def test_shield_interventions_burns_and_rejections_are_recorded():
    summary = _run_no_op(ShieldedOrbitGuardEnv, _COLLIDING_SEED)

    assert summary.collision_count == 0
    assert summary.shield_intervention_count > 0
    assert summary.shield_rejected_proposal_steps > 0
    assert summary.total_delta_v_mps > 0.0
    assert summary.maneuver_count > 0
    assert "unsafe_proposal_rejected_by_shield" in summary.failure_labels


def test_benign_episode_has_no_failure_labels():
    env = OrbitGuardEnv(num_agents=2, **_KW)
    for seed in range(60):
        env.reset(seed=seed)
        if env._pending_assessments[0].d_min_m > 5.0e5:
            break
    else:
        pytest.skip("no benign seed found")
    while env.agents:
        env.step({agent: 0 for agent in env.agents})

    summary = env.episode_summary()

    assert summary.failure_labels == ()
    assert summary.collision_count == summary.unsafe_conjunction_count == 0


def test_summary_resets_with_the_episode():
    env = OrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=_COLLIDING_SEED)
    while env.agents:
        env.step({agent: 0 for agent in env.agents})
    assert env.episode_summary().collision_count == 1

    env.reset(seed=_NEAR_MISS_SEED)

    assert env.episode_summary().collision_count == 0
    assert env.episode_summary().steps == 0


def test_summary_reports_the_fields_the_logging_spec_requires():
    summary = _run_no_op(OrbitGuardEnv, _NEAR_MISS_SEED)

    for field in (
        "collision_count", "unsafe_conjunction_count", "total_delta_v_mps", "maneuver_count",
        "mean_mission_deviation", "shield_intervention_count", "mean_residual_risk", "steps",
    ):
        assert hasattr(summary, field), field
    assert summary.steps == 40
    assert summary.mean_residual_risk >= 0.0


def test_maneuver_disabled_satellites_involved_in_a_conjunction_are_labelled():
    from orbit_guard.physics.state import SatelliteCapability

    disabled = SatelliteCapability(0.1, 2.0, False, 1.0)

    summary = _run_no_op(OrbitGuardEnv, _NEAR_MISS_SEED, capability_template=disabled)

    assert "satellite_unable_to_maneuver" in summary.failure_labels


def test_exhausted_budget_is_labelled():
    from orbit_guard.physics.state import SatelliteCapability

    spent = SatelliteCapability(0.1, 0.0, True, 1.0)

    summary = _run_no_op(OrbitGuardEnv, _NEAR_MISS_SEED, capability_template=spent)

    assert "maneuver_budget_exhausted" in summary.failure_labels


def test_unsafe_conjunctions_are_scored_against_the_adaptive_margin_even_for_a_fixed_margin_run():
    from orbit_guard.safety.adaptive_margin import make_fixed_margin

    # A 1 m fixed margin would call the 287 m near miss "safe"; the common reference margin does not.
    summary = _run_no_op(OrbitGuardEnv, _NEAR_MISS_SEED, margin_fn=make_fixed_margin(1.0))

    assert summary.unsafe_conjunction_count == 1


def test_mean_mission_deviation_averages_the_post_step_deviations_the_next_observation_reports():
    from orbit_guard.physics.mission_deviation import (
        compute_mission_deviation,
        compute_weighted_mission_deviation,
    )
    from orbit_guard.safety.maneuver_mapper import ManeuverIntent

    env = OrbitGuardEnv(num_agents=2, episode_horizon_s=300.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=600.0)
    env.reset(seed=11)
    expected = []
    while env.agents:
        acting = list(env.agents)
        env.step({agent: int(ManeuverIntent.RAISE_ORBIT) for agent in acting})
        for agent in acting:
            idx = env._agent_index[agent]
            components = compute_mission_deviation(env._pending_beliefs[idx].state_vector, env._shadow_states[idx])
            expected.append(compute_weighted_mission_deviation(components, **env._mission_weights))

    assert env.episode_summary().mean_mission_deviation == pytest.approx(sum(expected) / len(expected), rel=1e-12)
