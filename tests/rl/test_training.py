"""Tests for orbit_guard.rl.training: config-driven MAPPO/CTDE on RLlib 2.58.

Exercises the mechanical pipeline against the stub env (no learnable signal,
so no reward/convergence assertions -- see the Acceptance D docstring). What
IS asserted is the architecture: one shared module, a centralized critic that
really receives the global state, and an actor that never does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium.spaces.utils import flatten
from ray.rllib.core.columns import Columns

from orbit_guard.rl.ctde_module import POLICY_ID
from orbit_guard.rl.evaluation import make_greedy_policy
from orbit_guard.rl.training import (
    TrainingSettings,
    build_algo,
    build_mappo_config,
    load_reward_weights,
    load_training_settings,
    restore,
    train,
)
from tests.rl.stub_env import AGENT_OBS_FLAT_DIM, StubOrbitGuardEnv

_NUM_AGENTS = 2
_GLOBAL_DIM = _NUM_AGENTS * AGENT_OBS_FLAT_DIM + _NUM_AGENTS + 1  # slots | alive mask | time remaining
_SETTINGS = TrainingSettings(train_batch_size=64, minibatch_size=32, num_epochs=1, seed=0)


def _env_factory(config: dict) -> StubOrbitGuardEnv:
    return StubOrbitGuardEnv(num_agents=_NUM_AGENTS)


@pytest.fixture(scope="module")
def algo():
    built = build_algo(_env_factory, _SETTINGS, env_name="test_training_stub")
    yield built
    built.stop()


def test_settings_load_the_prd_hyperparameters_from_the_default_config():
    settings = load_training_settings()

    assert settings.learning_rate == pytest.approx(5.0e-4)
    assert settings.train_batch_size == 4000
    assert settings.minibatch_size == 256
    assert settings.num_epochs == 10
    assert settings.actor_hidden_layers == (128, 128)
    assert settings.critic_hidden_layers == (256, 256)


def test_every_numeric_setting_loads_as_a_number():
    # PyYAML parses `1.0e6` (no exponent sign) as a STRING; that once reached torch.clamp and crashed a run.
    settings = load_training_settings()

    for name in ("learning_rate", "gamma", "gae_lambda", "clip_param", "entropy_coeff", "vf_clip_param", "reward_scale", "reward_offset"):
        assert isinstance(getattr(settings, name), (int, float)), name
    for name in ("train_batch_size", "minibatch_size", "num_epochs", "num_env_runners"):
        assert isinstance(getattr(settings, name), int), name


def test_settings_overrides_take_precedence_over_the_config_file():
    assert load_training_settings(train_batch_size=64, seed=7).train_batch_size == 64


def test_one_shared_module_serves_every_agent(algo):
    assert set(algo.config.policies) == {POLICY_ID}
    assert algo.config.policy_mapping_fn("sat_0", None) == POLICY_ID
    assert algo.config.policy_mapping_fn("sat_1", None) == POLICY_ID


def test_module_has_an_actor_over_local_obs_and_a_critic_over_global_state(algo):
    module = algo.get_module(POLICY_ID)

    assert module.actor[0].in_features == AGENT_OBS_FLAT_DIM
    assert module.critic[0].in_features == _GLOBAL_DIM


def test_critic_receives_the_global_state_and_actor_does_not(algo):
    module = algo.get_module(POLICY_ID)
    local = torch.randn(3, AGENT_OBS_FLAT_DIM)
    global_a, global_b = torch.randn(3, _GLOBAL_DIM), torch.randn(3, _GLOBAL_DIM)

    def batch(global_state):
        return {Columns.OBS: torch.cat([local, global_state], dim=-1)}

    with torch.no_grad():
        values_differ = not torch.allclose(module.compute_values(batch(global_a)), module.compute_values(batch(global_b)))
        logits_equal = torch.equal(
            module.forward_inference(batch(global_a))[Columns.ACTION_DIST_INPUTS],
            module.forward_inference(batch(global_b))[Columns.ACTION_DIST_INPUTS],
        )

    assert values_differ
    assert logits_equal


def test_training_runs_and_reports_progress(algo):
    report = train(algo, num_iterations=2)

    assert report.iterations == 2
    assert len(report.episode_return_means) == 2
    assert report.env_steps_sampled >= 2 * _SETTINGS.train_batch_size


def test_train_rejects_a_non_positive_iteration_count(algo):
    with pytest.raises(ValueError):
        train(algo, num_iterations=0)


def test_checkpoint_round_trip_restores_identical_actor_weights(algo, tmp_path):
    report = train(algo, num_iterations=1, checkpoint_dir=tmp_path / "checkpoint")

    restored = restore(_env_factory, _SETTINGS, report.checkpoint_path)
    try:
        original_weights = algo.get_module(POLICY_ID).actor.state_dict()
        restored_weights = restored.get_module(POLICY_ID).actor.state_dict()
        assert all(torch.equal(original_weights[key], restored_weights[key]) for key in original_weights)
    finally:
        restored.stop()


def test_greedy_execution_ignores_the_global_state(algo):
    module = algo.get_module(POLICY_ID)
    policy = make_greedy_policy(algo, _env_factory)
    raw_env = _env_factory({})
    observations, _ = raw_env.reset(seed=0)
    agent = raw_env.possible_agents[0]

    decentralized_action = policy(agent, observations[agent])
    local = flatten(raw_env.observation_space(agent), observations[agent]).astype(np.float32)
    with_noise = torch.as_tensor(np.concatenate([local, np.full(_GLOBAL_DIM, 99.0, np.float32)]))[None]
    with torch.no_grad():
        logits = module.forward_inference({Columns.OBS: with_noise})[Columns.ACTION_DIST_INPUTS]

    assert decentralized_action == int(torch.argmax(logits).item())


def test_settings_expose_the_critic_fix_and_reward_scale_defaults():
    settings = TrainingSettings()

    assert settings.vf_clip_param >= 1.0e3
    assert settings.reward_scale == 1.0


def test_the_critic_fix_reaches_the_ppo_config(algo):
    assert algo.config.vf_clip_param == _SETTINGS.vf_clip_param


def test_reward_weights_load_the_tuned_defaults_from_the_config():
    assert load_reward_weights() == {"w1": 1.0, "w2": 0.3, "w3": 5.0, "w4": 0.5, "collision_penalty": -100}


def test_reward_weight_overrides_win_over_the_config_file():
    assert load_reward_weights(w3=1.0)["w3"] == 1.0


def test_unknown_reward_weight_names_are_rejected():
    with pytest.raises(ValueError):
        load_reward_weights(w9=1.0)


def test_env_factory_builds_a_shielded_env_only_when_asked():
    from orbit_guard.rl.environment import OrbitGuardEnv
    from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
    from orbit_guard.rl.training import make_env_factory

    weights = load_reward_weights()
    plain = make_env_factory(2, weights, shielded=False)({})
    shielded = make_env_factory(2, weights, shielded=True)({})

    assert type(plain) is OrbitGuardEnv
    assert isinstance(shielded, ShieldedOrbitGuardEnv)


def test_margin_fn_loads_the_configured_coefficients_and_accepts_overrides():
    from orbit_guard.rl.training import load_margin_fn
    from orbit_guard.safety.adaptive_margin import compute_d_safe, compute_sigma_norm

    default = load_margin_fn()
    tuned = load_margin_fn(alpha=2.0, gamma=0.5)

    assert default(141.0, 0.4, 0.3) == pytest.approx(compute_d_safe(compute_sigma_norm(141.0), 0.4, 0.3))
    assert tuned(141.0, 0.4, 0.3) == pytest.approx(
        compute_d_safe(compute_sigma_norm(141.0), 0.4, 0.3, alpha=2.0, gamma=0.5)
    )
    with pytest.raises(ValueError):
        load_margin_fn(delta=1.0)


def test_saved_checkpoint_carries_a_ray_free_actor_that_agrees_with_the_rllib_greedy_policy(algo, tmp_path):
    from orbit_guard.rl.actor_policy import NumpyActorPolicy, load_actor_weights
    from orbit_guard.rl.training import ACTOR_WEIGHTS_FILENAME

    report = train(algo, num_iterations=1, checkpoint_dir=tmp_path / "checkpoint")
    exported = NumpyActorPolicy(load_actor_weights(f"{report.checkpoint_path}/{ACTOR_WEIGHTS_FILENAME}"))
    rllib_policy = make_greedy_policy(algo, _env_factory)
    raw_env = _env_factory({})
    observations, _ = raw_env.reset(seed=3)

    for _ in range(5):
        actions = {agent: rllib_policy(agent, observations[agent]) for agent in raw_env.agents}
        for agent in raw_env.agents:
            assert exported(agent, observations[agent]) == actions[agent]
        observations, _, _, _, _ = raw_env.step(actions)


def test_the_critic_input_defaults_to_the_global_state_from_the_config():
    assert load_training_settings().critic_input == "global"
    assert TrainingSettings().critic_input == "global"


def test_the_critic_input_override_takes_precedence_over_the_config():
    assert load_training_settings(critic_input="agent_global").critic_input == "agent_global"


def test_training_settings_reject_an_unknown_critic_input():
    with pytest.raises(ValueError, match="critic_input"):
        TrainingSettings(critic_input="local")


@pytest.mark.parametrize("critic_input", ["global", "agent_global"])
def test_the_critic_input_reaches_the_rl_module_model_config(critic_input):
    settings = TrainingSettings(critic_input=critic_input)

    config = build_mappo_config("unused_env_name", AGENT_OBS_FLAT_DIM, settings)

    assert config.rl_module_spec.rl_module_specs[POLICY_ID].model_config["critic_input"] == critic_input


def test_env_runners_wait_long_enough_for_a_slow_shielded_n15_rollout_fragment():
    # A shielded N=15 step takes ~0.07 s, so a 1000-step fragment (4000 batch / 4 runners) needs ~70 s. RLlib's
    # default 60 s sample timeout returned no samples and every iteration trained on nothing (R4, 2026-11-03).
    config = build_mappo_config("test_training_stub", AGENT_OBS_FLAT_DIM, _SETTINGS)

    assert config.sample_timeout_s >= 600


class _NoSampleAlgo:
    """An algorithm whose iteration produced no samples, so no learner update (what a sample timeout looks like)."""

    def train(self) -> dict:
        return {"env_runners": {}, "learners": {}}


def test_an_iteration_without_a_learner_update_fails_loudly_instead_of_training_on_nothing():
    with pytest.raises(RuntimeError, match="no learner update"):
        train(_NoSampleAlgo(), 1)
