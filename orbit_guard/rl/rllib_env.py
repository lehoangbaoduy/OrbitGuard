"""RLlib multi-agent environment wrappers and factory helpers (PRD §28).

Provides:
- FlattenedParallelPettingZooEnv: Flattens Dict observations to a single Box
  for compatibility with RLlib's default encoders.
- CentralizedCriticParallelPettingZooEnv: Packages each agent's flattened local
  observation together with the global environment state (CTDE / MAPPO).
- create_rllib_env: Factory helper to instantiate and wrap any PettingZoo ParallelEnv.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space
from pettingzoo import ParallelEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


class FlattenedParallelPettingZooEnv(ParallelPettingZooEnv):
    """ParallelPettingZooEnv with each agent's Dict observation flattened
    to a single Box, so RLlib's new-API-stack default encoder can build."""

    def __init__(self, env: ParallelEnv) -> None:
        super().__init__(env)
        self._inner_observation_spaces = dict(self.observation_space.spaces)
        self.observation_space = gym.spaces.Dict(
            {
                agent_id: flatten_space(space)
                for agent_id, space in self._inner_observation_spaces.items()
            }
        )

    def _flatten_all(self, observations: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            agent_id: flatten(self._inner_observation_spaces[agent_id], obs)
            for agent_id, obs in observations.items()
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observations, infos = super().reset(seed=seed, options=options)
        return self._flatten_all(observations), infos

    def step(
        self,
        action_dict: dict[str, Any],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        observations, rewards, terminations, truncations, infos = super().step(action_dict)
        return self._flatten_all(observations), rewards, terminations, truncations, infos


class CentralizedCriticParallelPettingZooEnv(ParallelPettingZooEnv):
    """ParallelPettingZooEnv packaging each agent's local observation alongside
    the global constellation state for Centralized Training with Decentralized Execution (CTDE).

    Each agent's observation space is structured as:
        gym.spaces.Dict({
            "obs": flatten_space(local_observation_space),  # Local, decentralized
            "state": global_state_space,                   # Global, centralized critic only
        })
    """

    def __init__(self, env: ParallelEnv) -> None:
        super().__init__(env)
        self._inner_observation_spaces = dict(self.observation_space.spaces)

        # Retrieve global state space from environment
        state_space = getattr(self.par_env, "state_space", None)
        if state_space is None and hasattr(self.par_env, "unwrapped"):
            state_space = getattr(self.par_env.unwrapped, "state_space", None)

        if state_space is None:
            raise ValueError(
                "Underlying environment must expose `state_space` for centralized critic CTDE training."
            )
        self._state_space = state_space

        self.observation_space = gym.spaces.Dict(
            {
                agent_id: gym.spaces.Dict(
                    {
                        "obs": flatten_space(space),
                        "state": self._state_space,
                    }
                )
                for agent_id, space in self._inner_observation_spaces.items()
            }
        )

    def _get_global_state(self) -> np.ndarray:
        if hasattr(self.par_env, "state"):
            return self.par_env.state()
        if hasattr(self.par_env, "unwrapped") and hasattr(self.par_env.unwrapped, "state"):
            return self.par_env.unwrapped.state()
        raise ValueError(
            "Underlying environment must implement `state()` method returning global state."
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
        observations, infos = super().reset(seed=seed, options=options)
        global_state = self._get_global_state()
        wrapped_obs = {
            agent_id: {
                "obs": flatten(self._inner_observation_spaces[agent_id], obs),
                "state": global_state,
            }
            for agent_id, obs in observations.items()
        }
        return wrapped_obs, infos

    def step(
        self,
        action_dict: dict[str, Any],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        observations, rewards, terminations, truncations, infos = super().step(action_dict)
        if observations:
            global_state = self._get_global_state()
            wrapped_obs = {
                agent_id: {
                    "obs": flatten(self._inner_observation_spaces[agent_id], obs),
                    "state": global_state,
                }
                for agent_id, obs in observations.items()
            }
        else:
            wrapped_obs = {}
        return wrapped_obs, rewards, terminations, truncations, infos


def create_rllib_env(config: dict[str, Any] | None = None) -> ParallelPettingZooEnv:
    """Env-creator factory for RLlib environment registration.

    Accepts any PettingZoo ParallelEnv factory, class, or instance, and wraps it
    into an RLlib-compatible ParallelPettingZooEnv.

    Parameters:
        config: Configuration dictionary. Supported keys:
            - "env": An existing ParallelEnv instance.
            - "env_class": A ParallelEnv class to instantiate.
            - "env_creator": A factory callable returning a ParallelEnv.
            - "env_kwargs": Optional kwargs dict passed to env_class/env_creator.
            - "centralized_critic": bool (default: True). If True, wraps with
              CentralizedCriticParallelPettingZooEnv. If False, wraps with
              FlattenedParallelPettingZooEnv.

    Returns:
        Wrapped ParallelPettingZooEnv instance.
    """
    config = config or {}
    env_inst = config.get("env")
    if env_inst is None:
        factory: Callable[..., ParallelEnv] | type[ParallelEnv] | None = config.get(
            "env_creator"
        ) or config.get("env_class")
        if factory is None:
            raise ValueError(
                "create_rllib_env requires 'env', 'env_creator', or 'env_class' in config dictionary."
            )
        kwargs = config.get("env_kwargs", {})
        env_inst = factory(**kwargs)

    centralized = config.get("centralized_critic", True)
    if centralized:
        return CentralizedCriticParallelPettingZooEnv(env_inst)
    return FlattenedParallelPettingZooEnv(env_inst)
