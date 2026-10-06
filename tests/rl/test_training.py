"""Unit and integration contract tests for orbit_guard.rl.rllib_env and orbit_guard.rl.training (PRD §28).

Verifies:
- Flattened and CentralizedCritic environment wrappers conform to PettingZoo/RLlib specs.
- CentralizedCriticTorchRLModule architectural guarantee: actor ONLY receives local obs,
  while critic receives global state.
- Actor gradients never depend on global state inputs (hard CTDE separation).
- Parameter sharing across all agents under "shared_policy".
- End-to-end MAPPO training iteration with centralized critic on StubOrbitGuardEnv.
- Checkpoint save and restore round trip.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.tune.registry import register_env

from orbit_guard.rl.rllib_env import (
    CentralizedCriticParallelPettingZooEnv,
    FlattenedParallelPettingZooEnv,
    create_rllib_env,
)
from orbit_guard.rl.training import (
    CentralizedCriticTorchRLModule,
    build_mappo_config,
    create_algorithm,
    restore_checkpoint,
    save_checkpoint,
    train_iteration,
)
from tests.rl.stub_env import (
    AGENT_OBS_FLAT_DIM,
    NUM_ACTIONS,
    StubOrbitGuardEnv,
)

_ENV_NAME = "test_stub_orbit_guard_ctde"


# --- Environment wrapper tests ------------------------------------------------


def test_flattened_parallel_pettingzoo_env_flattens_observations():
    # Arrange
    raw_env = StubOrbitGuardEnv(num_agents=2)
    env = FlattenedParallelPettingZooEnv(raw_env)

    # Act
    observations, infos = env.reset(seed=42)

    # Assert
    for agent in raw_env.possible_agents:
        assert isinstance(env.observation_space[agent], gym.spaces.Box)
        assert env.observation_space[agent].shape == (AGENT_OBS_FLAT_DIM,)
        assert observations[agent].shape == (AGENT_OBS_FLAT_DIM,)
        assert isinstance(infos[agent], dict)

    action_dict = {agent: 0 for agent in raw_env.possible_agents}
    next_obs, rewards, _terminations, _truncations, _infos = env.step(action_dict)
    for agent in raw_env.possible_agents:
        assert next_obs[agent].shape == (AGENT_OBS_FLAT_DIM,)
        assert isinstance(rewards[agent], float)


def test_centralized_critic_parallel_pettingzoo_env_packages_obs_and_state():
    # Arrange
    raw_env = StubOrbitGuardEnv(num_agents=2)
    expected_state_dim = 2 * AGENT_OBS_FLAT_DIM
    env = CentralizedCriticParallelPettingZooEnv(raw_env)

    # Act
    observations, _infos = env.reset(seed=42)

    # Assert
    for agent in raw_env.possible_agents:
        agent_space = env.observation_space[agent]
        assert isinstance(agent_space, gym.spaces.Dict)
        assert "obs" in agent_space.spaces
        assert "state" in agent_space.spaces
        assert agent_space["obs"].shape == (AGENT_OBS_FLAT_DIM,)
        assert agent_space["state"].shape == (expected_state_dim,)

        obs_dict = observations[agent]
        assert obs_dict["obs"].shape == (AGENT_OBS_FLAT_DIM,)
        assert obs_dict["state"].shape == (expected_state_dim,)

    action_dict = {agent: 1 for agent in raw_env.possible_agents}
    next_obs, _rewards, _, _, _ = env.step(action_dict)
    for agent in raw_env.possible_agents:
        assert next_obs[agent]["obs"].shape == (AGENT_OBS_FLAT_DIM,)
        assert next_obs[agent]["state"].shape == (expected_state_dim,)


def test_centralized_critic_wrapper_rejects_env_without_state_space():
    # Arrange: mock env missing state_space
    class MissingStateEnv(StubOrbitGuardEnv):
        def __init__(self):
            super().__init__(num_agents=2)
            delattr(self, "state_space")

    # Act & Assert
    with pytest.raises(ValueError, match="state_space"):
        CentralizedCriticParallelPettingZooEnv(MissingStateEnv())


def test_create_rllib_env_factory_respects_centralized_critic_flag():
    # Arrange
    config_centralized = {
        "env_class": StubOrbitGuardEnv,
        "env_kwargs": {"num_agents": 2},
        "centralized_critic": True,
    }
    config_flattened = {
        "env_class": StubOrbitGuardEnv,
        "env_kwargs": {"num_agents": 2},
        "centralized_critic": False,
    }

    # Act
    env_c = create_rllib_env(config_centralized)
    env_f = create_rllib_env(config_flattened)

    # Assert
    assert isinstance(env_c, CentralizedCriticParallelPettingZooEnv)
    assert isinstance(env_f, FlattenedParallelPettingZooEnv)


def test_create_rllib_env_requires_valid_config():
    # Act & Assert
    with pytest.raises(ValueError, match="requires 'env'"):
        create_rllib_env({})


# --- CentralizedCriticTorchRLModule CTDE guarantee tests ---------------------


def test_centralized_critic_module_actor_forward_never_uses_global_state():
    # Arrange: module with Dict observation space
    obs_space = gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(-1.0, 1.0, shape=(AGENT_OBS_FLAT_DIM,), dtype=np.float32),
            "state": gym.spaces.Box(-1.0, 1.0, shape=(2 * AGENT_OBS_FLAT_DIM,), dtype=np.float32),
        }
    )
    act_space = gym.spaces.Discrete(NUM_ACTIONS)
    module = CentralizedCriticTorchRLModule(
        observation_space=obs_space,
        action_space=act_space,
        model_config={"actor_hidden_layers": [64], "critic_hidden_layers": [64]},
    )

    local_obs = torch.randn(2, AGENT_OBS_FLAT_DIM, requires_grad=True)
    global_state = torch.randn(2, 2 * AGENT_OBS_FLAT_DIM, requires_grad=True)
    batch = {"obs": {"obs": local_obs, "state": global_state}}

    # Act: forward pass through decentralized actor
    out = module._forward(batch)
    action_logits = out["action_dist_inputs"]
    loss = action_logits.sum()
    loss.backward()

    # Assert: local_obs has gradients, but global_state has ZERO gradient
    assert local_obs.grad is not None
    assert torch.any(local_obs.grad != 0.0)
    assert global_state.grad is None, (
        "Architectural violation: global state leaked into actor forward pass!"
    )


def test_centralized_critic_module_compute_values_receives_and_uses_global_state():
    # Arrange
    obs_space = gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(-1.0, 1.0, shape=(AGENT_OBS_FLAT_DIM,), dtype=np.float32),
            "state": gym.spaces.Box(-1.0, 1.0, shape=(2 * AGENT_OBS_FLAT_DIM,), dtype=np.float32),
        }
    )
    act_space = gym.spaces.Discrete(NUM_ACTIONS)
    module = CentralizedCriticTorchRLModule(
        observation_space=obs_space,
        action_space=act_space,
        model_config={"actor_hidden_layers": [64], "critic_hidden_layers": [64]},
    )

    local_obs = torch.randn(2, AGENT_OBS_FLAT_DIM, requires_grad=True)
    global_state = torch.randn(2, 2 * AGENT_OBS_FLAT_DIM, requires_grad=True)
    batch = {"obs": {"obs": local_obs, "state": global_state}}

    # Act: compute values through centralized critic
    values = module.compute_values(batch)
    loss = values.sum()
    loss.backward()

    # Assert: global_state receives gradients through the critic
    assert values.shape == (2,)
    assert global_state.grad is not None
    assert torch.any(global_state.grad != 0.0)
    assert local_obs.grad is None, (
        "Critic should be conditioned on global state, not local observation directly"
    )


# --- Training configuration and loop integration tests -----------------------


def test_build_mappo_config_parameter_sharing_across_all_agents():
    # Arrange & Act
    config = build_mappo_config(policy_id="shared_policy")

    # Assert
    assert set(config.policies) == {"shared_policy"}
    assert config.policy_mapping_fn("sat_0", None) == "shared_policy"
    assert config.policy_mapping_fn("sat_1", None) == "shared_policy"
    assert config.policy_mapping_fn("sat_14", None) == "shared_policy"


def test_one_training_iteration_runs_with_centralized_critic_on_stub_env():
    # Arrange: register centralized critic env
    env_name = f"{_ENV_NAME}_train"

    def _make_env(cfg):
        return CentralizedCriticParallelPettingZooEnv(StubOrbitGuardEnv(num_agents=2))

    register_env(env_name, _make_env)
    config = build_mappo_config(
        env_name=env_name,
        train_batch_size=64,
        minibatch_size=32,
        num_epochs=1,
        actor_hidden_layers=(64, 64),
        critic_hidden_layers=(64, 64),
    )

    # Act
    algo = create_algorithm(config)
    try:
        result = train_iteration(algo)
        # Assert: algorithm ran successfully and reported learner metrics
        assert "env_runners" in result or "num_env_steps_sampled_lifetime" in result
        assert "learners" in result
        shared_metrics = result["learners"]["shared_policy"]
        assert "vf_loss" in shared_metrics
        assert "policy_loss" in shared_metrics
    finally:
        algo.stop()


def test_checkpoint_save_and_restore_with_centralized_critic_round_trips(tmp_path):
    # Arrange
    env_name = f"{_ENV_NAME}_ckpt"

    def _make_env(cfg):
        return CentralizedCriticParallelPettingZooEnv(StubOrbitGuardEnv(num_agents=2))

    register_env(env_name, _make_env)
    config = build_mappo_config(
        env_name=env_name,
        train_batch_size=64,
        minibatch_size=32,
        num_epochs=1,
        actor_hidden_layers=(64, 64),
        critic_hidden_layers=(64, 64),
    )

    algo = create_algorithm(config)
    try:
        train_iteration(algo)
        checkpoint_path = save_checkpoint(algo, tmp_path / "checkpoint")
        assert checkpoint_path is not None
    finally:
        algo.stop()

    # Act: restore into a freshly built algorithm
    restored_algo = create_algorithm(config)
    try:
        restore_checkpoint(restored_algo, checkpoint_path)
    finally:
        restored_algo.stop()
