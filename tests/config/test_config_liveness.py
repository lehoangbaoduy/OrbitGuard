"""Changing a configs/default.yaml value changes behaviour through the production factories (CORE-1).

Every test writes a tmp copy of the default config with one edited value and builds the environment
through `evaluation.baselines.build_env_factory`, the same path the golden traces and the experiments use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import expit

from orbit_guard.config import DEFAULT_CONFIG_PATH
from orbit_guard.evaluation.baselines import build_env_factory
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.observation import POSITION_SCALE_M
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.maneuver_mapper import ManeuverIntent
from tests.rl.test_coordination_clip import risk_raising_proposal

_NO_SHIELD = "adaptive_mappo_no_shield"
_SHIELDED = "adaptive_mappo"
_SEED = 11


def _tmp_config(tmp_path: Path, edits: dict, name: str = "config.yaml") -> Path:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    for dotted, value in edits.items():
        *parents, leaf = dotted.split(".")
        node = config
        for key in parents:
            node = node[key]
        node[leaf] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=True))
    return path


def _env(model: str, num_agents: int, config_path: Path | None = None, **kwargs):
    factory = build_env_factory(model, num_agents, **({"config_path": config_path} if config_path else {}), **kwargs)
    return factory({})


def _pair_values(env, index: int) -> list[float]:
    return [pair[index] for pair in env.snapshot()["pairs"]]


def test_k_neighbors_sets_the_neighbour_slots_of_the_observation(tmp_path):
    env = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"observation.k_neighbors": 2}))

    obs, _ = env.reset(seed=_SEED)

    assert obs["sat_0"]["neighbors"].shape == (2, 10)


def test_lookahead_changes_the_predicted_closest_approach(tmp_path):
    short = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"simulation.lookahead_s": 60.0}))
    default = _env(_NO_SHIELD, 3)
    short.reset(seed=_SEED)
    default.reset(seed=_SEED)

    assert _pair_values(short, 2) != _pair_values(default, 2)


def test_hard_body_radius_decides_which_pairs_collide(tmp_path):
    wide = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"collision.hard_body_radius_m": 1.0e9}))
    default = _env(_NO_SHIELD, 3)
    noop = {"sat_0": 0, "sat_1": 0, "sat_2": 0}
    wide.reset(seed=_SEED)
    default.reset(seed=_SEED)

    _, _, wide_terminations, _, _ = wide.step(noop)
    _, _, default_terminations, _, _ = default.step(noop)

    assert all(wide_terminations.values())
    assert not any(default_terminations.values())


def test_risk_z0_and_k_set_the_risk_proxy_of_every_pair(tmp_path):
    shifted = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"risk.z0": 10.0, "risk.k": 0.8, "risk.epsilon_m": 4.0}))
    shifted.reset(seed=_SEED)

    assessments = shifted._pending_assessments
    assert assessments
    for assessment in assessments:
        z_score = assessment.d_min_m / (assessment.sigma_rel_m + 4.0)
        assert assessment.risk_proxy == pytest.approx(float(expit(0.8 * (10.0 - z_score))), rel=1e-12)


def test_density_ref_count_changes_the_local_density_in_the_margin(tmp_path):
    dense = {"adaptive_margin.density_radius_km": 20_000.0}
    low_ref = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {**dense, "adaptive_margin.density_ref_count": 5.0}))
    high_ref = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {**dense, "adaptive_margin.density_ref_count": 50.0}, "high.yaml"))
    low_ref.reset(seed=_SEED)
    high_ref.reset(seed=_SEED)

    assert _pair_values(low_ref, 3) != _pair_values(high_ref, 3)


def test_phase_fraction_scales_the_phase_burn_magnitude(tmp_path):
    quarter = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"maneuver.phase_fraction": 0.25}))
    default = _env(_NO_SHIELD, 3)
    quarter.reset(seed=_SEED)
    default.reset(seed=_SEED)
    burn = {"sat_0": int(ManeuverIntent.PHASE_FORWARD), "sat_1": int(ManeuverIntent.PHASE_FORWARD), "sat_2": 0}

    quarter.step(burn)
    default.step(burn)

    assert default.last_step["applied_dv_mps"][0] == pytest.approx(2.0 * quarter.last_step["applied_dv_mps"][0])


def test_shield_search_settings_reach_the_shield(tmp_path):
    env = _env(_SHIELDED, 3, _tmp_config(tmp_path, {"shield.beam_width": 7, "shield.exact_enumeration_max_agents": 2}))

    assert isinstance(env, ShieldedOrbitGuardEnv)
    assert env._shield.beam_width == 7
    assert env._shield.exact_max_agents == 2


def test_shield_uses_the_environment_risk_and_phase_settings(tmp_path):
    env = _env(_SHIELDED, 3, _tmp_config(tmp_path, {"risk.z0": 4.5, "maneuver.phase_fraction": 0.3}))

    assert env._shield.risk_z0 == 4.5
    assert env._shield.phase_fraction == 0.3


def test_episode_horizon_sets_the_number_of_control_steps(tmp_path):
    env = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"simulation.episode_horizon_s": 120.0}))
    env.reset(seed=_SEED)
    steps = 0
    while env.agents:
        _, _, _, truncations, _ = env.step({agent: 0 for agent in env.agents})
        steps += 1
        assert steps <= 10
    assert steps == 2
    assert all(truncations.values())


def test_sensing_sigma_of_zero_gives_noise_free_beliefs(tmp_path):
    env = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"sensing.position_sigma_m": 0.0, "sensing.velocity_sigma_mps": 0.0}))

    obs, _ = env.reset(seed=_SEED)

    np.testing.assert_allclose(obs["sat_0"]["own"][0:3], env._ground_truth[0][:3] / POSITION_SCALE_M, rtol=0, atol=1e-6)


def test_capability_template_sets_budget_per_action_and_mission_limit(tmp_path):
    edits = {
        "capability.remaining_delta_v_mps": 0.5,
        "capability.mission_deviation_limit": 3.0,
        "maneuver.base_delta_v_mps": 0.2,
    }
    env = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, edits))

    env.reset(seed=_SEED)

    assert env.delta_v_budget_mps == 0.5
    assert env.max_delta_v_per_action_mps == 0.2
    assert all(capability.mission_deviation_limit == 3.0 for capability in env._capabilities)


def test_distribution_mix_decides_whether_conjunctions_are_injected(tmp_path):
    benign_only = {"experiment.training_distribution_mix": {"benign": 1.0, "single_conjunction": 0.0, "multi_conjunction": 0.0}}
    env = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, benign_only))
    default = _env(_NO_SHIELD, 3)
    seeds = range(20, 40)

    benign_counts = [len(env.reset(seed=seed) and env._last_scenario.injected_conjunctions) for seed in seeds]
    default_counts = [len(default.reset(seed=seed) and default._last_scenario.injected_conjunctions) for seed in seeds]

    assert benign_counts == [0] * len(seeds)
    assert sum(default_counts) > 0


def test_mission_weights_decide_the_observed_mission_deviation(tmp_path):
    zero = {f"reward.mission_weights.{key}": 0.0 for key in ("q_a", "q_e", "q_i", "q_M")}
    unweighted = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, zero))
    default = _env(_NO_SHIELD, 3)

    obs_zero, _ = unweighted.reset(seed=_SEED)
    obs_default, _ = default.reset(seed=_SEED)

    assert obs_zero["sat_0"]["own"][7] == 0.0
    assert obs_default["sat_0"]["own"][7] > 0.0


def test_reward_weights_reach_the_environment_through_the_factory(tmp_path):
    env = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {"reward.w3_delta_v": 7.5}))

    assert env._reward_weights["w3"] == 7.5


def test_coordination_clip_caps_the_coordination_term_in_the_reward(tmp_path):
    # A 300 s horizon puts the injected conjunction inside the 1800 s lookahead at reset.
    w4_only = {
        "simulation.episode_horizon_s": 300.0, "reward.w1_safety": 0.0, "reward.w2_mission": 0.0,
        "reward.w3_delta_v": 0.0, "reward.collision_penalty": 0.0, "reward.w4_coordination": 1.0,
    }
    clipped = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {**w4_only, "reward.coordination_clip": 0.01}))
    loose = _env(_NO_SHIELD, 3, _tmp_config(tmp_path, {**w4_only, "reward.coordination_clip": 1.0}, "loose.yaml"))
    clipped.reset(seed=_SEED)
    loose.reset(seed=_SEED)
    proposal, raw_term = risk_raising_proposal(loose)
    assert raw_term > 0.01  # precondition: the clip binds

    _, clipped_rewards, _, _, _ = clipped.step(proposal)
    _, loose_rewards, _, _, _ = loose.step(proposal)

    assert all(reward == -0.01 for reward in clipped_rewards.values())
    assert all(reward == pytest.approx(-raw_term, rel=1e-12) for reward in loose_rewards.values())


def test_bare_constructor_resolves_reward_weights_from_the_config():
    env = OrbitGuardEnv(num_agents=2)

    assert env._reward_weights["w3"] == 5.0
    assert env._reward_weights["w4"] == 0.5


def test_overlay_reaches_the_environment_through_the_factory(tmp_path):
    overlay = tmp_path / "tuned.yaml"
    overlay.write_text(yaml.safe_dump({"reward": {"w4": 1.5}, "adaptive_margin": {"alpha": 2.0}}))

    factory = build_env_factory(_NO_SHIELD, 3, overlay_path=overlay)
    env = factory({})

    assert env._reward_weights["w4"] == 1.5
    assert env._margin_fn.alpha == 2.0


def test_explicit_env_kwargs_win_over_the_config(tmp_path):
    env = build_env_factory(_NO_SHIELD, 3, k_neighbors=2)({})

    obs, _ = env.reset(seed=_SEED)

    assert obs["sat_0"]["neighbors"].shape == (2, 10)


def test_shielded_factory_builds_a_shielded_env():
    assert isinstance(_env(_SHIELDED, 3), ShieldedOrbitGuardEnv)
