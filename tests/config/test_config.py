"""Tests for orbit_guard.config (CORE-1): one loader for configs/default.yaml plus a tuned overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from orbit_guard.config import (
    DEFAULT_CONFIG_PATH,
    apply_overlay,
    config_digest,
    env_kwargs_from_config,
    load_config,
    margin_from_config,
)
from orbit_guard.safety.adaptive_margin import FixedMargin

_REPO = Path(__file__).resolve().parents[2]


def _write_overlay(tmp_path: Path, body: dict, name: str = "tuned.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body, sort_keys=True))
    return path


def _with_hash(body: dict, base_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    digest = hashlib.sha256(base_path.read_bytes() + json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {**body, "config_hash": digest}


def test_default_config_path_points_at_the_repo_yaml():
    assert DEFAULT_CONFIG_PATH == _REPO / "configs" / "default.yaml"


def test_load_config_without_overlay_equals_the_default_yaml():
    assert load_config() == yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def test_load_config_returns_a_fresh_dict_on_every_call():
    first = load_config()
    first["reward"]["w3_delta_v"] = 999.0

    second = load_config()

    assert second["reward"]["w3_delta_v"] == 5.0


def test_overlay_merges_allowed_blocks_per_key(tmp_path):
    overlay = _write_overlay(
        tmp_path,
        {
            "reward": {"w4": 1.5},
            "adaptive_margin": {"alpha": 2.0},
            "rl": {"learning_rate": 1.0e-3},
        },
    )

    config = load_config(overlay_path=overlay)

    assert config["reward"]["w4_coordination"] == 1.5
    assert config["reward"]["w1_safety"] == 1.0
    assert config["adaptive_margin"]["alpha"] == 2.0
    assert config["adaptive_margin"]["beta"] == 0.5
    assert config["rl"]["learning_rate"] == 1.0e-3
    assert config["rl"]["gamma"] == 0.95


def test_overlay_metadata_keys_are_accepted_and_do_not_merge(tmp_path):
    overlay = _write_overlay(
        tmp_path,
        {
            "selected": "margin_wide",
            "selected_on": "validation",
            "validation": {"episodes": 20, "score": 0.1},
            "reward": {},
        },
    )

    config = load_config(overlay_path=overlay)

    assert config == load_config()
    assert "selected" not in config and "validation" not in config


def test_overlay_rejects_an_unknown_top_level_block(tmp_path):
    overlay = _write_overlay(tmp_path, {"simulation": {"physics_dt_s": 5}})

    with pytest.raises(ValueError, match="simulation"):
        load_config(overlay_path=overlay)


def test_overlay_rejects_an_unknown_key_inside_an_allowed_block(tmp_path):
    for body in (
        {"reward": {"w9": 1.0}},
        {"adaptive_margin": {"delta": 1.0}},
        {"rl": {"not_a_setting": 1}},
    ):
        overlay = _write_overlay(tmp_path, body, name="bad.yaml")
        with pytest.raises(ValueError, match="unknown"):
            load_config(overlay_path=overlay)


def test_overlay_rejects_a_config_hash_that_does_not_match_the_base_config(tmp_path):
    overlay = _write_overlay(tmp_path, _with_hash({"adaptive_margin": {"alpha": 2.0}}) | {"config_hash": "0" * 64})

    with pytest.raises(ValueError, match="config_hash"):
        load_config(overlay_path=overlay)


def test_overlay_rejects_a_hash_made_against_a_different_base_config(tmp_path):
    other_base = tmp_path / "other.yaml"
    other_base.write_text(DEFAULT_CONFIG_PATH.read_text().replace("alpha: 1.0", "alpha: 3.0"))
    overlay = _write_overlay(tmp_path, _with_hash({"adaptive_margin": {"alpha": 2.0}}, other_base))

    with pytest.raises(ValueError, match="config_hash"):
        load_config(overlay_path=overlay)


def test_overlay_with_a_correct_config_hash_is_accepted(tmp_path):
    overlay = _write_overlay(tmp_path, _with_hash({"adaptive_margin": {"alpha": 2.0}, "selected": "x"}))

    assert load_config(overlay_path=overlay)["adaptive_margin"]["alpha"] == 2.0


def test_overlay_without_a_config_hash_is_accepted(tmp_path):
    overlay = _write_overlay(tmp_path, {"adaptive_margin": {"alpha": 2.0}})

    assert load_config(overlay_path=overlay)["adaptive_margin"]["alpha"] == 2.0


def test_config_digest_is_the_base_bytes_without_overlay():
    assert config_digest(DEFAULT_CONFIG_PATH) == hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()


def test_config_digest_changes_when_an_overlay_is_applied(tmp_path):
    overlay = _write_overlay(tmp_path, {"adaptive_margin": {"alpha": 2.0}})
    other = _write_overlay(tmp_path, {"adaptive_margin": {"alpha": 3.0}}, name="other.yaml")

    plain = config_digest(DEFAULT_CONFIG_PATH)
    tuned = config_digest(DEFAULT_CONFIG_PATH, overlay)

    assert tuned != plain
    assert tuned != config_digest(DEFAULT_CONFIG_PATH, other)
    assert tuned == config_digest(DEFAULT_CONFIG_PATH, overlay)


def test_apply_overlay_is_a_pure_merge_that_leaves_the_base_untouched():
    base = load_config()

    merged = apply_overlay(base, {"reward": {"w3": 10.0}})

    assert merged["reward"]["w3_delta_v"] == 10.0
    assert base["reward"]["w3_delta_v"] == 5.0


def test_env_kwargs_carry_every_value_the_simulation_blocks_declare():
    kwargs = env_kwargs_from_config(load_config(), shielded=True)

    assert kwargs["episode_horizon_s"] == 21_600.0
    assert kwargs["physics_dt_s"] == 10.0
    assert kwargs["control_dt_s"] == 60.0
    assert kwargs["lookahead_s"] == 1800.0
    assert kwargs["k_neighbors"] == 4
    assert kwargs["hard_body_radius_m"] == 5.0
    assert kwargs["phase_fraction"] == 0.5
    assert kwargs["risk_epsilon_m"] == 1.0
    assert kwargs["risk_z0"] == 3.0
    assert kwargs["risk_k"] == 1.5
    assert kwargs["density_radius_m"] == 50_000.0
    assert kwargs["density_ref_count"] == 5.0
    assert kwargs["mission_weights"] == {"q_a": 0.25, "q_e": 0.25, "q_i": 0.25, "q_m": 0.25}
    assert kwargs["reward_weights"] == {"w1": 1.0, "w2": 0.3, "w3": 5.0, "w4": 0.5, "collision_penalty": -100}
    assert kwargs["distribution_mix"] == {"benign": 0.30, "single_conjunction": 0.50, "multi_conjunction": 0.20}
    assert kwargs["capability_template"].max_delta_v_per_action_mps == 0.1
    assert kwargs["capability_template"].remaining_delta_v_mps == 2.0
    assert kwargs["capability_template"].mission_deviation_limit == 1.0
    assert kwargs["noise_model"].position_sigma_m == 100.0
    assert kwargs["noise_model"].velocity_sigma_mps == 0.1
    assert kwargs["shield_exact_max_agents"] == 4
    assert kwargs["shield_beam_width"] == 50


def test_env_kwargs_omit_shield_arguments_for_the_unshielded_env():
    kwargs = env_kwargs_from_config(load_config(), shielded=False)

    assert "shield_exact_max_agents" not in kwargs
    assert "shield_beam_width" not in kwargs


def test_env_kwargs_resolve_the_tle_file_against_the_repo_root():
    kwargs = env_kwargs_from_config(load_config(), shielded=False)

    assert Path(kwargs["tle_path"]) == _REPO / "data" / "TLE_snapshot_53deg_shell.csv"


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("sensing", "noise_model", "uniform", "noise_model"),
        ("sensing", "covariance_mode", "sampled", "covariance_mode"),
        ("dynamics", "model", "point_mass", "dynamics.model"),
        ("dynamics", "integrator", "euler", "integrator"),
        ("risk", "model", "linear", "risk.model"),
        ("shield", "repair_mode", "greedy", "repair_mode"),
        ("simulation", "physics_dt_s", 0, "physics_dt_s"),
        ("simulation", "control_dt_s", -60, "control_dt_s"),
        ("simulation", "control_dt_s", 65, "multiple"),
        ("simulation", "episode_horizon_s", 0, "episode_horizon_s"),
        ("collision", "hard_body_radius_m", 0, "hard_body_radius_m"),
        ("risk", "epsilon_m", 0, "epsilon_m"),
        ("observation", "k_neighbors", 0, "k_neighbors"),
        ("shield", "beam_width", 0, "beam_width"),
        ("shield", "exact_enumeration_max_agents", 0, "exact_enumeration_max_agents"),
    ],
)
def test_env_kwargs_reject_unsupported_or_invalid_values(section, key, value, message):
    config = load_config()
    config[section][key] = value

    with pytest.raises(ValueError, match=message):
        env_kwargs_from_config(config, shielded=True)


def test_margin_from_config_returns_the_adaptive_margin_or_the_fixed_constant():
    config = load_config()

    fixed = margin_from_config(config, "fixed")
    adaptive = margin_from_config(config, "adaptive")

    assert fixed == FixedMargin(1142.0)
    assert adaptive(0.0, 0.0, 0.0) == pytest.approx(1000.0)


def test_margin_from_config_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="model_margin"):
        margin_from_config(load_config(), "dynamic")


def test_an_overlay_written_by_freeze_best_loads_with_its_hash_verified(tmp_path):
    from orbit_guard.rl.evaluation import EvaluationReport
    from orbit_guard.tuning.grid_search import GridPoint, GridResult, freeze_best

    report = EvaluationReport(
        episodes=10, mean_return_per_agent=0.0, collision_episodes=0, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=0.1, mean_episode_steps=360.0,
    )
    target = tmp_path / "tuned.yaml"
    freeze_best([GridResult(GridPoint("wide", reward_weights={"w4": 1.5}, margin={"alpha": 2.0}), report, 0.01)], target)

    config = load_config(overlay_path=target)

    assert config["reward"]["w4_coordination"] == 1.5
    assert config["adaptive_margin"]["alpha"] == 2.0
    assert "selected" not in config
