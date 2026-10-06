"""MAPPO/CTDE training pipeline and RLlib integration (PRD §28).

Implements Multi-Agent PPO with Centralized Training and Decentralized Execution (CTDE):
- Parameter sharing across all satellites via a single shared-weights policy.
- Decentralized actor policy network conditioned strictly on local agent observations.
- Centralized critic value network conditioned on global constellation state during training.
- Checkpoint persistence and training cycle helpers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.tune.registry import register_env
from torch import nn

_DEFAULT_LR = 3.0e-4
_DEFAULT_GAMMA = 0.99
_DEFAULT_GAE_LAMBDA = 0.95
_DEFAULT_CLIP_PARAM = 0.2
_DEFAULT_ENTROPY_COEFF = 0.01
_DEFAULT_TRAIN_BATCH_SIZE = 4000
_DEFAULT_MINIBATCH_SIZE = 256
_DEFAULT_NUM_EPOCHS = 10
_DEFAULT_ACTOR_HIDDENS = (128, 128)
_DEFAULT_CRITIC_HIDDENS = (256, 256)
_DEFAULT_POLICY_ID = "shared_policy"


class CentralizedCriticTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """MAPPO/CTDE PyTorch RLModule with a decentralized actor and a centralized critic.

    CTDE architectural guarantee:
        - Actor network takes ONLY the local observation (`batch[Columns.OBS]["obs"]`).
          At NO point does any global-state or opponent-specific feature reach the actor.
        - Critic network takes the global constellation state (`batch[Columns.OBS]["state"]`)
          during value prediction in training and GAE calculation.
    """

    def setup(self) -> None:
        if isinstance(self.observation_space, gym.spaces.Dict):
            if "obs" not in self.observation_space.spaces or "state" not in self.observation_space.spaces:
                raise ValueError(
                    f"Observation Dict must contain 'obs' and 'state' keys, "
                    f"got {list(self.observation_space.spaces.keys())}"
                )
            obs_dim = int(self.observation_space["obs"].shape[0])
            state_dim = int(self.observation_space["state"].shape[0])
        elif isinstance(self.observation_space, gym.spaces.Box):
            obs_dim = int(self.observation_space.shape[0])
            state_dim = obs_dim
        else:
            raise TypeError(f"Unsupported observation space: {self.observation_space}")

        if isinstance(self.action_space, gym.spaces.Discrete):
            act_dim = int(self.action_space.n)
        else:
            raise TypeError(f"Unsupported action space: {self.action_space}")

        actor_hiddens = self.model_config.get("actor_hidden_layers", list(_DEFAULT_ACTOR_HIDDENS))
        critic_hiddens = self.model_config.get("critic_hidden_layers", list(_DEFAULT_CRITIC_HIDDENS))

        # Build decentralized actor MLP
        actor_layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in actor_hiddens:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(nn.Tanh())
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Build centralized critic MLP
        critic_layers: list[nn.Module] = []
        in_dim = state_dim
        for h in critic_hiddens:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(nn.Tanh())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def _extract_actor_obs(self, batch: dict[str, Any]) -> torch.Tensor:
        raw_obs = batch[Columns.OBS]
        if isinstance(raw_obs, dict):
            return raw_obs["obs"]
        return raw_obs

    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Forward pass for inference and exploration (decentralized actor only)."""
        obs = self._extract_actor_obs(batch)
        logits = self.actor(obs)
        return {Columns.ACTION_DIST_INPUTS: logits}

    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Forward pass during training."""
        return self._forward(batch, **kwargs)

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Any | None = None,
    ) -> torch.Tensor:
        """Centralized value function forward pass (critic receiving global state)."""
        raw_obs = batch[Columns.OBS]
        if isinstance(raw_obs, dict) and "state" in raw_obs:
            state = raw_obs["state"]
        else:
            state = raw_obs
        values = self.critic(state)
        return values.squeeze(-1)


def build_mappo_config(
    env_name: str | None = None,
    env_creator: Callable[[dict[str, Any]], Any] | None = None,
    *,
    policy_id: str = _DEFAULT_POLICY_ID,
    learning_rate: float = _DEFAULT_LR,
    gamma: float = _DEFAULT_GAMMA,
    gae_lambda: float = _DEFAULT_GAE_LAMBDA,
    clip_param: float = _DEFAULT_CLIP_PARAM,
    entropy_coeff: float = _DEFAULT_ENTROPY_COEFF,
    train_batch_size: int = _DEFAULT_TRAIN_BATCH_SIZE,
    minibatch_size: int = _DEFAULT_MINIBATCH_SIZE,
    num_epochs: int = _DEFAULT_NUM_EPOCHS,
    actor_hidden_layers: Sequence[int] = _DEFAULT_ACTOR_HIDDENS,
    critic_hidden_layers: Sequence[int] = _DEFAULT_CRITIC_HIDDENS,
    num_env_runners: int = 0,
) -> PPOConfig:
    """Build a PPOConfig configured for MAPPO/CTDE with parameter sharing.

    Configures:
    - Shared actor policy mapped across all agents.
    - CentralizedCriticTorchRLModule integrating decentralized actor and centralized critic.
    - Exact hyperparameters from default configuration (learning rate, GAE, clipping).
    - Local synchronous environment runners for CI stability.

    Parameters:
        env_name: Name of registered environment.
        env_creator: Optional env constructor callable to register under env_name.
        policy_id: Policy ID string (default: "shared_policy").
        learning_rate: PPO learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_param: PPO surrogate clipping threshold.
        entropy_coeff: Entropy bonus coefficient.
        train_batch_size: Total timesteps per training iteration.
        minibatch_size: SGD minibatch size.
        num_epochs: Number of SGD passes per training iteration.
        actor_hidden_layers: Actor MLP hidden layer sizes.
        critic_hidden_layers: Critic MLP hidden layer sizes.
        num_env_runners: Number of parallel environment runners (0 for in-driver).

    Returns:
        Configured PPOConfig ready for .build_algo().
    """
    if env_name and env_creator:
        register_env(env_name, env_creator)

    config = (
        PPOConfig()
        .multi_agent(
            policies={policy_id},
            policy_mapping_fn=lambda agent_id, episode, **kw: policy_id,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=CentralizedCriticTorchRLModule,
                model_config={
                    "actor_hidden_layers": list(actor_hidden_layers),
                    "critic_hidden_layers": list(critic_hidden_layers),
                },
            )
        )
        .training(
            lr=learning_rate,
            gamma=gamma,
            lambda_=gae_lambda,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
        )
        .env_runners(num_env_runners=num_env_runners)
    )

    if env_name:
        config = config.environment(env_name)

    return config


def create_algorithm(config: PPOConfig) -> Algorithm:
    """Instantiate an RLlib Algorithm from PPOConfig using modern build_algo()."""
    return config.build_algo()


def train_iteration(algo: Algorithm) -> dict[str, Any]:
    """Step the algorithm through one training iteration and return results dict."""
    return algo.train()


def save_checkpoint(algo: Algorithm, checkpoint_path: str | Path) -> str:
    """Persist an algorithm checkpoint to an explicit destination path."""
    return algo.save_to_path(str(checkpoint_path))


def restore_checkpoint(algo: Algorithm, checkpoint_path: str | Path) -> None:
    """Restore algorithm weights and state from an explicit checkpoint path."""
    algo.restore_from_path(str(checkpoint_path))
