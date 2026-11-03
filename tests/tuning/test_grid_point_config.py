"""A grid point's overrides reach the environment through the config path (CORE-1)."""

from __future__ import annotations

import hashlib
import json

import yaml

from orbit_guard.config import DEFAULT_CONFIG_PATH, load_config
from orbit_guard.rl.evaluation import EvaluationReport
from orbit_guard.tuning.grid_search import (
    GridPoint,
    GridResult,
    config_for_point,
    factory_for_point,
    freeze_best,
)


def test_config_for_point_applies_reward_and_margin_overrides_on_top_of_the_default_config():
    point = GridPoint("wide", reward_weights={"w4": 1.5}, margin={"alpha": 2.0, "gamma": 2.0})

    config = config_for_point(point)

    assert config["reward"]["w4_coordination"] == 1.5
    assert config["reward"]["w3_delta_v"] == 5.0
    assert config["adaptive_margin"]["alpha"] == 2.0
    assert config["adaptive_margin"]["gamma"] == 2.0
    assert load_config()["adaptive_margin"]["alpha"] == 1.0


def test_an_unmodified_point_reproduces_the_default_config():
    assert config_for_point(GridPoint("default")) == load_config()


def test_grid_point_overrides_reach_the_environment_built_for_that_point():
    point = GridPoint("tuned", reward_weights={"w3": 2.5}, margin={"alpha": 2.0})

    env = factory_for_point(point, 3, shielded=False)({})

    assert env._reward_weights["w3"] == 2.5
    assert env._margin_fn.alpha == 2.0


def _report(collisions: int) -> EvaluationReport:
    return EvaluationReport(
        episodes=10, mean_return_per_agent=0.0, collision_episodes=collisions, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=0.1, mean_budget_fraction_used=0.05, unsafe_episodes=0, mean_episode_steps=360.0,
    )


def test_freeze_best_hash_is_the_shared_overlay_digest(tmp_path):
    results = [GridResult(GridPoint("strong", margin={"alpha": 2.0}), _report(0), 0.01, "q27")]
    target = tmp_path / "tuned.yaml"

    digest = freeze_best(results, target)

    frozen = yaml.safe_load(target.read_text())
    body = {key: value for key, value in frozen.items() if key != "config_hash"}
    expected = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes() + json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert digest == expected
