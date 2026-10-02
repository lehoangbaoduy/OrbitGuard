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
    load_reward_weights,
    load_training_settings,
    restore,
    train,
)
from tests.rl.stub_env import AGENT_OBS_FLAT_DIM, StubOrbitGuardEnv

_NUM_AGENTS = 2
_GLOBAL_DIM = _NUM_AGENTS * AGENT_OBS_FLAT_DIM
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
