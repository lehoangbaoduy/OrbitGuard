"""Every leaf of configs/default.yaml is consumed by a loader or listed as documentation-only (CORE-1).

A future block that nothing reads fails here, so the duplicated-constant drift this refactor removed
cannot silently return. The CONSUMED map names the loader that reads each leaf; the liveness tests in
test_config_liveness.py prove that changing those values changes behaviour.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

CONSUMED = {
    # orbit_guard.config.env_kwargs_from_config -> OrbitGuardEnv / ShieldedOrbitGuardEnv arguments
    "simulation.episode_horizon_s": "env_kwargs_from_config",
    "simulation.physics_dt_s": "env_kwargs_from_config",
    "simulation.control_dt_s": "env_kwargs_from_config",
    "simulation.lookahead_s": "env_kwargs_from_config",
    "dynamics.model": "env_kwargs_from_config (validated)",
    "dynamics.integrator": "env_kwargs_from_config (validated)",
    "sensing.noise_model": "env_kwargs_from_config (validated)",
    "sensing.position_sigma_m": "env_kwargs_from_config",
    "sensing.velocity_sigma_mps": "env_kwargs_from_config",
    "sensing.covariance_mode": "env_kwargs_from_config (validated)",
    "collision.hard_body_radius_m": "env_kwargs_from_config",
    "risk.model": "env_kwargs_from_config (validated)",
    "risk.epsilon_m": "env_kwargs_from_config",
    "risk.z0": "env_kwargs_from_config",
    "risk.k": "env_kwargs_from_config",
    "adaptive_margin.d_base_m": "margin_from_config",
    "adaptive_margin.d_min_m": "margin_from_config",
    "adaptive_margin.d_max_m": "margin_from_config",
    "adaptive_margin.alpha": "margin_from_config",
    "adaptive_margin.beta": "margin_from_config",
    "adaptive_margin.gamma": "margin_from_config",
    "adaptive_margin.density_radius_km": "env_kwargs_from_config",
    "adaptive_margin.density_ref_count": "env_kwargs_from_config",
    "rl.learning_rate": "rl.training.load_training_settings",
    "rl.gamma": "rl.training.load_training_settings",
    "rl.gae_lambda": "rl.training.load_training_settings",
    "rl.clip_param": "rl.training.load_training_settings",
    "rl.entropy_coeff": "rl.training.load_training_settings",
    "rl.train_batch_size": "rl.training.load_training_settings",
    "rl.sgd_minibatch_size": "rl.training.load_training_settings",
    "rl.num_sgd_iter": "rl.training.load_training_settings",
    "rl.actor_hidden_layers": "rl.training.load_training_settings",
    "rl.critic_hidden_layers": "rl.training.load_training_settings",
    "rl.critic_input": "rl.training.load_training_settings",
    "rl.vf_clip_param": "rl.training.load_training_settings",
    "rl.reward_scale": "rl.training.load_training_settings",
    "rl.reward_offset": "rl.training.load_training_settings",
    "rl.num_env_runners": "rl.training.load_training_settings",
    "rl.num_training_seeds": "evaluation.registry_checks.expected_training_seeds (registry/freeze completeness)",
    "observation.k_neighbors": "env_kwargs_from_config",
    "maneuver.base_delta_v_mps": "env_kwargs_from_config",
    "maneuver.phase_fraction": "env_kwargs_from_config",
    "capability.remaining_delta_v_mps": "env_kwargs_from_config",
    "capability.remaining_delta_v_range_mps": "env_kwargs_from_config",
    "capability.mission_deviation_limit": "env_kwargs_from_config",
    "reward.w1_safety": "env_kwargs_from_config",
    "reward.w2_mission": "env_kwargs_from_config",
    "reward.w3_delta_v": "env_kwargs_from_config",
    "reward.w4_coordination": "env_kwargs_from_config",
    "reward.collision_penalty": "env_kwargs_from_config",
    "reward.coordination_clip": "env_kwargs_from_config",
    "reward.mission_weights.q_a": "env_kwargs_from_config",
    "reward.mission_weights.q_e": "env_kwargs_from_config",
    "reward.mission_weights.q_i": "env_kwargs_from_config",
    "reward.mission_weights.q_M": "env_kwargs_from_config",
    "shield.repair_mode": "env_kwargs_from_config (validated)",
    "shield.exact_enumeration_max_agents": "env_kwargs_from_config",
    "shield.beam_width": "env_kwargs_from_config",
    "models.no_maneuver.policy": "evaluation.baselines._model_spec",
    "models.no_maneuver.shield": "evaluation.baselines._model_spec",
    "models.no_maneuver.margin": "evaluation.baselines._model_spec",
    "models.rule_based.policy": "evaluation.baselines._model_spec",
    "models.rule_based.shield": "evaluation.baselines._model_spec",
    "models.rule_based.margin": "evaluation.baselines._model_spec",
    "models.fixed_margin_mappo.policy": "evaluation.baselines._model_spec",
    "models.fixed_margin_mappo.shield": "evaluation.baselines._model_spec",
    "models.fixed_margin_mappo.margin": "evaluation.baselines._model_spec",
    "models.adaptive_mappo.policy": "evaluation.baselines._model_spec",
    "models.adaptive_mappo.shield": "evaluation.baselines._model_spec",
    "models.adaptive_mappo.margin": "evaluation.baselines._model_spec",
    "models.adaptive_mappo_no_shield.policy": "evaluation.baselines._model_spec",
    "models.adaptive_mappo_no_shield.shield": "evaluation.baselines._model_spec",
    "models.adaptive_mappo_no_shield.margin": "evaluation.baselines._model_spec",
    "fixed_margin.d_safe_m": "margin_from_config (fixed)",
    "shifts.normal": "evaluation.shifts.condition_env_kwargs",
    "shifts.noise_increase.noise_scale": "evaluation.shifts.condition_env_kwargs",
    "shifts.density_increase.phase_spread_rad": "evaluation.shifts.condition_env_kwargs",
    "shifts.simultaneous_conjunctions.distribution_mix.triple_conjunction": "evaluation.shifts.condition_env_kwargs",
    "shifts.capability_reduction.disabled_count": "evaluation.shifts.condition_env_kwargs",
    "shifts.capability_reduction.reduced_count": "evaluation.shifts.condition_env_kwargs",
    "shifts.capability_reduction.reduced_budget_scale": "evaluation.shifts.condition_env_kwargs",
    "experiment.model": "evaluation.baselines.model_from_config",
    "experiment.training_distribution_mix.benign": "env_kwargs_from_config",
    "experiment.training_distribution_mix.single_conjunction": "env_kwargs_from_config",
    "experiment.training_distribution_mix.multi_conjunction": "env_kwargs_from_config",
    "data.tle_file": "env_kwargs_from_config (resolved against the repo root)",
}

# Keys nothing reads at run time. Each carries the reason it is documentation, not configuration.
DOCUMENTATION_ONLY = {
    "simulation.N": "default constellation size; every experiment passes num_agents / --agents explicitly",
    "simulation.N_range": "PRD design range of constellation sizes; not enforced (tests use N=2 and N=3)",
    "rl.algorithm": "names the algorithm the training code implements (MAPPO/CTDE)",
    "rl.parameter_sharing": "the training code always uses one shared actor (PRD section 28)",
    "experiment.validation_seeds": "validation pool is configs/validation_seeds.json (seed_pools)",
    "experiment.test_seeds_per_condition": "final-test pool size is set by the test-protocol module",
    "experiment.evaluation_seeds_per_model": "evaluation seed count is set by the experiment CLI",
    "data.tle_source": "provenance of the TLE snapshot, documentation only",
    "data.tle_format": "the snapshot loader reads the pre-parsed element CSV in data.tle_file",
    "data.tle_epoch_range": "provenance of the TLE snapshot, documentation only",
    "data.tle_satellite_count": "provenance of the TLE snapshot, documentation only",
    "data.tle_inclination_range_deg": "provenance of the TLE snapshot, documentation only",
    "data.tle_altitude_range_km": "provenance of the TLE snapshot, documentation only",
    "data.snapshot_frozen": "provenance flag, documentation only",
}


def _leaf_paths(node, prefix: str = "") -> list[str]:
    if isinstance(node, dict) and node:
        paths: list[str] = []
        for key, value in node.items():
            paths.extend(_leaf_paths(value, f"{prefix}{key}."))
        return paths
    return [prefix[:-1]]


def _config_leaves() -> list[str]:
    return _leaf_paths(yaml.safe_load(_DEFAULT.read_text()))


def test_every_leaf_of_default_yaml_is_consumed_or_documentation_only():
    unread = [leaf for leaf in _config_leaves() if leaf not in CONSUMED and leaf not in DOCUMENTATION_ONLY]

    assert unread == [], f"config keys nothing reads (wire them or justify them as documentation): {unread}"


def test_every_consumed_or_documented_key_exists_in_default_yaml():
    leaves = set(_config_leaves())
    stale = sorted((set(CONSUMED) | set(DOCUMENTATION_ONLY)) - leaves)

    assert stale == []


def test_consumed_and_documentation_lists_do_not_overlap():
    assert set(CONSUMED).isdisjoint(DOCUMENTATION_ONLY)


def test_every_documentation_only_key_has_a_justification():
    assert all(reason.strip() for reason in DOCUMENTATION_ONLY.values())
