"""RLlib env-wrapping for MAPPO/CTDE (PRD §28), verified against Ray 2.58.

Two Ray 2.58 (new RLModule/Catalog API stack) facts shape this module, both
confirmed empirically rather than assumed from older tutorials:

  1. RLlib's default encoder cannot build for a `Dict` observation space with
     several `Box` sub-spaces (`ValueError: No default encoder config for obs
     space=Dict(...)`). Each agent's Dict observation is therefore flattened
     into one Box via `gymnasium.spaces.utils.flatten`.
  2. There is no first-class "centralized critic input" channel. The accepted
     workaround is to carry the global state in the observation vector. Each
     agent's observation becomes ``[local | global_state]`` and
     `ctde_module.CentralizedCriticTorchRLModule` makes sure the actor only
     ever reads the ``local`` slice (PRD §28: no critic-only feature may leak
     into decentralized execution).

The wrapper accepts ANY PettingZoo `ParallelEnv` exposing `state()` -- the
stub during development, `OrbitGuardEnv` in production -- via an env factory,
so nothing here imports the environment itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space
from pettingzoo import ParallelEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

EnvFactory = Callable[[dict], ParallelEnv]


class CTDEParallelPettingZooEnv(ParallelPettingZooEnv):
    """`ParallelPettingZooEnv` emitting flat ``[local | global_state]`` observations.

    With ``include_global_state=False`` it emits only the flattened local
    observation (the plain, IPPO-style / execution-time view). `reward_offset`
    and `reward_scale` reshape rewards for the learner only, as
    ``(reward + reward_offset) * reward_scale``. Centering (a negative offset
    cancelling the constant ~+1 a calm step earns) shrinks value magnitudes so
    the critic's error stays small next to the action-dependent signal; it
    never changes evaluation rewards.
    """

    def __init__(
        self,
        env: ParallelEnv,
        include_global_state: bool = True,
        reward_scale: float = 1.0,
        reward_offset: float = 0.0,
    ):
        super().__init__(env)
        if reward_scale <= 0:
            raise ValueError(f"reward_scale must be > 0, got {reward_scale}")
        self._reward_scale = reward_scale
        self._reward_offset = reward_offset
        if include_global_state and not callable(getattr(env, "state", None)):
            raise ValueError("include_global_state=True requires the env to expose a state() method")

        self._include_global_state = include_global_state
        self._inner_observation_spaces = dict(self.observation_space.spaces)
        local_dims = {flatten_space(space).shape[0] for space in self._inner_observation_spaces.values()}
        if len(local_dims) != 1:
            raise ValueError(f"all agents must share one local observation size, got {sorted(local_dims)}")
        self.local_obs_dim = local_dims.pop()
        self.global_state_dim = int(env.state_space.shape[0]) if include_global_state else 0

        flat_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.local_obs_dim + self.global_state_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(dict.fromkeys(self._inner_observation_spaces, flat_observation_space))

    def _assemble(self, observations: dict[str, Any]) -> dict[str, np.ndarray]:
        global_state = self.par_env.state() if self._include_global_state else None
        assembled = {}
        for agent_id, observation in observations.items():
            local = flatten(self._inner_observation_spaces[agent_id], observation).astype(np.float32)
            assembled[agent_id] = local if global_state is None else np.concatenate([local, global_state])
        return assembled

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observations, infos = super().reset(seed=seed, options=options)
        return self._assemble(observations), infos

    def step(self, action_dict: dict):
        observations, rewards, terminations, truncations, infos = super().step(action_dict)
        scaled_rewards = {agent: (reward + self._reward_offset) * self._reward_scale for agent, reward in rewards.items()}
        return self._assemble(observations), scaled_rewards, terminations, truncations, infos


def make_env_creator(
    env_factory: EnvFactory,
    include_global_state: bool = True,
    reward_scale: float = 1.0,
    reward_offset: float = 0.0,
) -> Callable[[dict], MultiAgentEnv]:
    """Return the `env_creator(config)` RLlib's registry expects."""

    def env_creator(config: dict) -> MultiAgentEnv:
        return CTDEParallelPettingZooEnv(env_factory(config), include_global_state, reward_scale, reward_offset)

    return env_creator


def register_orbit_guard_env(
    name: str,
    env_factory: EnvFactory,
    include_global_state: bool = True,
    reward_scale: float = 1.0,
    reward_offset: float = 0.0,
) -> str:
    """Register `env_factory` under `name` with RLlib's env registry; returns `name`."""
    register_env(name, make_env_creator(env_factory, include_global_state, reward_scale, reward_offset))
    return name
