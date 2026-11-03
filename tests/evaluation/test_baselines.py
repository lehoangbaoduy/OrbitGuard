"""Tests for orbit_guard.evaluation.baselines (PRD §31, FR14: baselines from config alone)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from orbit_guard.evaluation.baselines import (
    MODEL_NAMES,
    build_env_factory,
    build_policy,
    rule_based_policy,
)
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_FAST = {"episode_horizon_s": 120.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 60.0}


def _observation(*, risk=0.9, along_track=0.4, enabled=1.0, remaining=1.0, valid=1.0):
    neighbors = np.zeros((4, 10))
    neighbors[0, 1] = along_track
    neighbors[0, 6] = risk
    neighbors[0, 9] = valid
    own = np.zeros(10)
    own[6] = remaining
    own[9] = enabled
    return {"own": own, "neighbors": neighbors, "global_summary": np.zeros(4)}


def test_there_are_exactly_five_models():
    assert MODEL_NAMES == (
        "no_maneuver", "rule_based", "fixed_margin_mappo", "adaptive_mappo", "adaptive_mappo_no_shield"
    )


def test_rule_based_does_nothing_without_a_threat():
    assert rule_based_policy("sat_0", _observation(risk=0.2)) == ManeuverIntent.NO_MANEUVER
    assert rule_based_policy("sat_0", _observation(valid=0.0)) == ManeuverIntent.NO_MANEUVER


def test_rule_based_moves_away_along_track_from_the_threat():
    ahead = rule_based_policy("sat_0", _observation(along_track=+0.4))
    behind = rule_based_policy("sat_0", _observation(along_track=-0.4))

    assert ahead == ManeuverIntent.RAISE_ORBIT  # neighbour ahead -> prograde -> drift backward
    assert behind == ManeuverIntent.LOWER_ORBIT


def test_rule_based_falls_back_to_a_radial_burn_only_when_the_threat_is_co_located():
    # every relative-position component is inside the deadband (within 5 m on all three axes)
    assert rule_based_policy("sat_0", _observation(along_track=0.0)) == ManeuverIntent.RADIAL_ADJUSTMENT
    assert rule_based_policy("sat_0", _observation(along_track=5.0e-4)) == ManeuverIntent.RADIAL_ADJUSTMENT


def test_rule_based_cannot_act_when_disabled_or_out_of_budget():
    assert rule_based_policy("sat_0", _observation(enabled=0.0)) == ManeuverIntent.NO_MANEUVER
    assert rule_based_policy("sat_0", _observation(remaining=0.0)) == ManeuverIntent.NO_MANEUVER


def test_rule_based_is_deterministic_and_stateless():
    observation = _observation()

    assert {rule_based_policy("sat_0", observation) for _ in range(5)} == {rule_based_policy("sat_1", observation)}


def test_no_maneuver_policy_always_returns_no_maneuver():
    policy = build_policy("no_maneuver")

    assert policy("sat_0", _observation()) == ManeuverIntent.NO_MANEUVER


@pytest.mark.parametrize(
    ("model", "expected_class"),
    [
        ("no_maneuver", OrbitGuardEnv),
        ("rule_based", OrbitGuardEnv),
        ("adaptive_mappo_no_shield", OrbitGuardEnv),
        ("fixed_margin_mappo", ShieldedOrbitGuardEnv),
        ("adaptive_mappo", ShieldedOrbitGuardEnv),
    ],
)
def test_the_model_name_alone_selects_the_environment_class(model, expected_class):
    factory = build_env_factory(model, num_agents=2, **_FAST)

    assert type(factory({})) is expected_class


def test_fixed_margin_models_use_a_constant_margin_and_adaptive_ones_do_not():
    fixed = build_env_factory("fixed_margin_mappo", num_agents=2, **_FAST)({})
    adaptive = build_env_factory("adaptive_mappo", num_agents=2, **_FAST)({})

    assert fixed._margin_fn(100.0, 0.0, 0.0) == fixed._margin_fn(900.0, 1.0, 1.0) == 1142.0
    assert adaptive._margin_fn(100.0, 0.0, 0.0) != adaptive._margin_fn(900.0, 1.0, 1.0)


def test_learned_models_need_actor_weights_and_heuristics_do_not(tmp_path):
    with pytest.raises(ValueError, match="actor weights"):
        build_policy("adaptive_mappo")

    assert callable(build_policy("rule_based"))


def test_switching_the_model_in_the_config_is_the_only_change_between_baselines(tmp_path):
    from orbit_guard.evaluation.baselines import model_from_config

    base = {"experiment": {"model": "no_maneuver"}}
    other = {"experiment": {"model": "adaptive_mappo"}}
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text(yaml.safe_dump(base))
    second.write_text(yaml.safe_dump(other))

    assert model_from_config(first) == "no_maneuver"
    assert model_from_config(second) == "adaptive_mappo"
    with pytest.raises(ValueError):
        model_from_config(_write(tmp_path, {"experiment": {"model": "oracle"}}))


def _write(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(content))
    return path
