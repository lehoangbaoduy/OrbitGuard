"""MAPPO/CTDE training pipeline on RLlib 2.58's new API stack (PRD §28).

Verified API pattern (recorded because PRD §28 says to re-verify, not assume):
  * `ray.rllib.algorithms.ppo.PPOConfig` on the new RLModule/Learner stack,
    built with `build_algo()` (not the deprecated `build()`).
  * ONE shared-weights module id (`shared_policy`) via `.multi_agent(policies=...,
    policy_mapping_fn=...)` -- parameter sharing across satellites.
  * A custom `RLModuleSpec` -> `ctde_module.CentralizedCriticTorchRLModule`: a
    shared actor over each agent's local observation plus a CENTRALIZED critic
    over the global state, carried as the tail of the flat observation by
    `rllib_env.CTDEParallelPettingZooEnv`. This is true CTDE, not IPPO.
  * Dict observations are flattened to a Box first (see `rllib_env`).

Config-driven: hyperparameters come from `configs/default.yaml`'s `rl:` block
(PRD §28/§39), so every learned baseline trains from a config swap alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from orbit_guard.rl.ctde_module import (
    DEFAULT_ACTOR_HIDDEN_LAYERS,
    DEFAULT_CRITIC_HIDDEN_LAYERS,
    CentralizedCriticTorchRLModule,
)
from orbit_guard.rl.rllib_env import (
    CTDEParallelPettingZooEnv,
    EnvFactory,
    register_orbit_guard_env,
)

POLICY_ID = "shared_policy"
DEFAULT_ENV_NAME = "orbit_guard_ctde"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class TrainingSettings:
    """PPO/MAPPO hyperparameters (PRD §28 starting point)."""

    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    train_batch_size: int = 4000
    minibatch_size: int = 256
    num_epochs: int = 10
    actor_hidden_layers: tuple[int, ...] = DEFAULT_ACTOR_HIDDEN_LAYERS
    critic_hidden_layers: tuple[int, ...] = DEFAULT_CRITIC_HIDDEN_LAYERS
    num_env_runners: int = 0
    seed: int | None = None


def load_training_settings(config_path: str | Path = _DEFAULT_CONFIG_PATH, **overrides: Any) -> TrainingSettings:
    """Read the `rl:` block of a config YAML; `overrides` win over the file."""
    with Path(config_path).open() as handle:
        rl_block = yaml.safe_load(handle)["rl"]
    settings = {
        "learning_rate": rl_block["learning_rate"],
        "gamma": rl_block["gamma"],
        "gae_lambda": rl_block["gae_lambda"],
        "clip_param": rl_block["clip_param"],
        "entropy_coeff": rl_block["entropy_coeff"],
        "train_batch_size": rl_block["train_batch_size"],
        "minibatch_size": rl_block["sgd_minibatch_size"],
        "num_epochs": rl_block["num_sgd_iter"],
        "actor_hidden_layers": tuple(rl_block["actor_hidden_layers"]),
        "critic_hidden_layers": tuple(rl_block["critic_hidden_layers"]),
    }
    settings.update(overrides)
    return TrainingSettings(**settings)


def build_mappo_config(env_name: str, local_obs_dim: int, settings: TrainingSettings) -> PPOConfig:
    """PPOConfig with one shared actor and a centralized critic (true CTDE)."""
    module_spec = RLModuleSpec(
        module_class=CentralizedCriticTorchRLModule,
        model_config={
            "local_obs_dim": local_obs_dim,
            "actor_hidden_layers": settings.actor_hidden_layers,
            "critic_hidden_layers": settings.critic_hidden_layers,
        },
    )
    config = (
        PPOConfig()
        .environment(env_name)
        .multi_agent(
            policies={POLICY_ID},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: POLICY_ID,
        )
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs={POLICY_ID: module_spec}))
        .env_runners(num_env_runners=settings.num_env_runners)
        .training(
            lr=settings.learning_rate,
            gamma=settings.gamma,
            lambda_=settings.gae_lambda,
            clip_param=settings.clip_param,
            entropy_coeff=settings.entropy_coeff,
            train_batch_size=settings.train_batch_size,
            minibatch_size=settings.minibatch_size,
            num_epochs=settings.num_epochs,
        )
    )
    if settings.seed is not None:
        config = config.debugging(seed=settings.seed)
    return config


def build_algo(env_factory: EnvFactory, settings: TrainingSettings, env_name: str = DEFAULT_ENV_NAME) -> Algorithm:
    """Register `env_factory` and build the MAPPO/CTDE algorithm for it."""
    register_orbit_guard_env(env_name, env_factory)
    probe_env = CTDEParallelPettingZooEnv(env_factory({}))
    local_obs_dim = probe_env.local_obs_dim
    probe_env.close()
    return build_mappo_config(env_name, local_obs_dim, settings).build_algo()


@dataclass(frozen=True)
class TrainingReport:
    iterations: int
    episode_return_means: tuple[float, ...]
    env_steps_sampled: int
    checkpoint_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def train(algo: Algorithm, num_iterations: int, checkpoint_dir: str | Path | None = None) -> TrainingReport:
    """Run `num_iterations` PPO iterations; optionally save a checkpoint to `checkpoint_dir`."""
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")

    returns: list[float] = []
    result: dict[str, Any] = {}
    for _ in range(num_iterations):
        result = algo.train()
        returns.append(float(result.get("env_runners", {}).get("episode_return_mean", float("nan"))))

    checkpoint_path = algo.save_to_path(str(checkpoint_dir)) if checkpoint_dir is not None else None
    return TrainingReport(
        iterations=num_iterations,
        episode_return_means=tuple(returns),
        env_steps_sampled=int(result.get("num_env_steps_sampled_lifetime", 0)),
        checkpoint_path=checkpoint_path,
    )


def restore(env_factory: EnvFactory, settings: TrainingSettings, checkpoint_path: str | Path) -> Algorithm:
    """Rebuild the algorithm from the same config and restore weights (no code changes between train/eval)."""
    algo = build_algo(env_factory, settings)
    algo.restore_from_path(str(checkpoint_path))
    return algo


def greedy_action(algo: Algorithm, local_observation: np.ndarray, global_state_dim: int) -> int:
    """Decentralized execution: argmax of the actor from the LOCAL observation only.

    The module still expects the full ``[local | global]`` vector shape, so the
    global slice is zero-padded; the actor never reads it (see ctde_module).
    """
    module = algo.get_module(POLICY_ID)
    padded = np.concatenate([local_observation.astype(np.float32), np.zeros(global_state_dim, dtype=np.float32)])
    with torch.no_grad():
        outputs = module.forward_inference({Columns.OBS: torch.as_tensor(padded)[None]})
    return int(torch.argmax(outputs[Columns.ACTION_DIST_INPUTS], dim=-1).item())


@dataclass(frozen=True)
class EvaluationReport:
    episodes: int
    mean_return_per_agent: float
    collisions: int
    mean_episode_steps: float


def evaluate(algo: Algorithm, env_factory: EnvFactory, num_episodes: int, seed: int = 0) -> EvaluationReport:
    """Greedy rollout on the raw env: one local observation per agent, no global state."""
    from gymnasium.spaces.utils import flatten

    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")

    env = env_factory({})
    global_state_dim = int(env.state_space.shape[0])
    returns: list[float] = []
    steps: list[int] = []
    collisions = 0

    for episode in range(num_episodes):
        observations, _ = env.reset(seed=seed + episode)
        episode_return, episode_steps = 0.0, 0
        while env.agents:
            actions = {
                agent: greedy_action(algo, flatten(env.observation_space(agent), observations[agent]), global_state_dim)
                for agent in env.agents
            }
            observations, rewards, terminations, _, _ = env.step(actions)
            episode_return += sum(rewards.values()) / max(len(rewards), 1)
            episode_steps += 1
            collisions += sum(bool(terminated) for terminated in terminations.values())
        returns.append(episode_return)
        steps.append(episode_steps)

    return EvaluationReport(
        episodes=num_episodes,
        mean_return_per_agent=float(np.mean(returns)),
        collisions=collisions,
        mean_episode_steps=float(np.mean(steps)),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: train MAPPO/CTDE on `OrbitGuardEnv`, then evaluate the trained actor.

    This is the human-read convergence run (PRD §44 P3-M5): inspect the printed
    per-iteration returns and the evaluation report yourself; it is not CI-gated.
    """
    import argparse

    from orbit_guard.rl.environment import OrbitGuardEnv

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args(argv)

    settings = load_training_settings(args.config, seed=args.seed)

    def env_factory(config: dict) -> OrbitGuardEnv:
        return OrbitGuardEnv(num_agents=args.agents)

    algo = build_algo(env_factory, settings)
    try:
        report = train(algo, args.iterations, args.checkpoint_dir)
        for iteration, mean_return in enumerate(report.episode_return_means, start=1):
            print(f"iteration {iteration:3d}  episode_return_mean {mean_return:10.3f}")
        print(evaluate(algo, env_factory, args.eval_episodes, seed=args.seed))
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
