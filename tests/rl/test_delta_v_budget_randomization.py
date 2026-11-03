"""Per-episode Δv budget randomization (PRD §17, CORE-3a).

At reset each satellite's initial budget is drawn Uniform(capability.remaining_delta_v_range_mps) from a
DEDICATED stream seeded by the episode seed, so the draw never touches the scenario or sensing streams and its
count never depends on actions, the model or the shield. Capability degradation (PRD §32 #4) scales the drawn budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.rl.environment import CapabilityDegradation, OrbitGuardEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_KW = {"episode_horizon_s": 180.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 60.0}
_RANGE = (1.5, 2.5)
_N = 10


def _budgets(env: OrbitGuardEnv) -> list[float]:
    return [capability.remaining_delta_v_mps for capability in env._capabilities]


def _run_episode(env: OrbitGuardEnv, seed: int, intent: ManeuverIntent) -> None:
    env.reset(seed=seed)
    while env.agents:
        env.step({agent: int(intent) for agent in env.agents})


def test_a_bare_environment_draws_each_satellites_budget_from_the_configured_range():
    env = OrbitGuardEnv(num_agents=_N, **_KW)
    env.reset(seed=4)

    budgets = _budgets(env)
    assert all(_RANGE[0] <= budget <= _RANGE[1] for budget in budgets)
    assert len(set(budgets)) == _N, "independent draws per satellite"
    assert list(env.initial_delta_v_budgets_mps().values()) == budgets
    assert env.delta_v_used_mps() == {agent: 0.0 for agent in env.possible_agents}


def test_budgets_depend_only_on_the_episode_seed_not_on_the_policy_that_ran_before():
    burner, idler = OrbitGuardEnv(num_agents=_N, **_KW), OrbitGuardEnv(num_agents=_N, **_KW)
    _run_episode(burner, 3, ManeuverIntent.RAISE_ORBIT)
    _run_episode(idler, 3, ManeuverIntent.NO_MANEUVER)

    burner.reset(seed=8)
    idler.reset(seed=8)
    fresh = OrbitGuardEnv(num_agents=_N, **_KW)
    fresh.reset(seed=8)

    assert _budgets(burner) == _budgets(idler) == _budgets(fresh)


def test_shielded_and_unshielded_models_face_the_same_budgets_under_a_seed():
    plain, shielded = OrbitGuardEnv(num_agents=_N, **_KW), ShieldedOrbitGuardEnv(num_agents=_N, **_KW)
    plain.reset(seed=5)
    shielded.reset(seed=5)

    assert _budgets(plain) == _budgets(shielded)


def test_different_seeds_draw_different_budgets():
    env = OrbitGuardEnv(num_agents=_N, **_KW)
    env.reset(seed=1)
    first = _budgets(env)
    env.reset(seed=2)

    assert _budgets(env) != first


def test_the_budget_draw_leaves_the_scenario_and_sensing_noise_streams_untouched():
    randomized = OrbitGuardEnv(num_agents=_N, **_KW)
    fixed = OrbitGuardEnv(num_agents=_N, remaining_delta_v_range_mps=None, **_KW)
    randomized.reset(seed=6)
    fixed.reset(seed=6)
    assert _budgets(fixed) == [2.0] * _N and _budgets(randomized) != _budgets(fixed)

    for _ in range(3):
        np.testing.assert_array_equal(np.array(randomized._ground_truth), np.array(fixed._ground_truth))
        np.testing.assert_array_equal(np.array(randomized._shadow_states), np.array(fixed._shadow_states))
        np.testing.assert_array_equal(
            np.array([b.state_vector for b in randomized._pending_beliefs]),
            np.array([b.state_vector for b in fixed._pending_beliefs]),
        )
        no_op = {agent: int(ManeuverIntent.NO_MANEUVER) for agent in randomized.agents}
        randomized.step(no_op)
        fixed.step(dict(no_op))


def test_capability_degradation_scales_the_drawn_budget():
    single = {"benign": 0.0, "single_conjunction": 1.0, "multi_conjunction": 0.0}
    undegraded = OrbitGuardEnv(num_agents=_N, distribution_mix=single, **_KW)
    degraded = OrbitGuardEnv(
        num_agents=_N, distribution_mix=single,
        capability_degradation=CapabilityDegradation(disabled_count=1, reduced_count=2, reduced_budget_scale=0.25), **_KW,
    )
    undegraded.reset(seed=2)
    degraded.reset(seed=2)

    drawn, after = _budgets(undegraded), _budgets(degraded)
    reduced = [i for i in range(_N) if after[i] != drawn[i]]
    assert len(reduced) == 2
    assert all(after[i] == pytest.approx(0.25 * drawn[i]) for i in reduced)
    assert list(degraded.initial_delta_v_budgets_mps().values()) == after


def test_an_explicit_capability_template_without_a_range_keeps_the_fixed_budget():
    template = SatelliteCapability(
        max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=1.2, maneuver_enabled=True, mission_deviation_limit=1.0
    )
    env = OrbitGuardEnv(num_agents=_N, capability_template=template, **_KW)
    env.reset(seed=4)

    assert _budgets(env) == [1.2] * _N


def test_an_explicit_range_applies_even_with_an_explicit_template():
    template = SatelliteCapability(
        max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=1.2, maneuver_enabled=True, mission_deviation_limit=1.0
    )
    env = OrbitGuardEnv(num_agents=_N, capability_template=template, remaining_delta_v_range_mps=(0.4, 0.6), **_KW)
    env.reset(seed=4)

    assert all(0.4 <= budget <= 0.6 for budget in _budgets(env))


@pytest.mark.parametrize("bad_range", [(2.5, 1.5), (-1.0, 1.0), (1.0, float("nan")), (1.0,)])
def test_an_invalid_budget_range_is_rejected_at_construction(bad_range):
    with pytest.raises(ValueError, match="remaining_delta_v_range_mps"):
        OrbitGuardEnv(num_agents=2, remaining_delta_v_range_mps=bad_range, **_KW)


def test_the_production_factory_randomizes_budgets_for_every_model():
    from orbit_guard.evaluation.baselines import build_env_factory

    for model in ("no_maneuver", "adaptive_mappo"):
        env = build_env_factory(model, _N)({})
        env.reset(seed=4)
        assert all(_RANGE[0] <= budget <= _RANGE[1] for budget in _budgets(env))
        assert len(set(_budgets(env))) == _N
