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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from orbit_guard.rl.ctde_module import (
    DEFAULT_ACTOR_HIDDEN_LAYERS,
    DEFAULT_CRITIC_HIDDEN_LAYERS,
    POLICY_ID,
    CentralizedCriticTorchRLModule,
)
from orbit_guard.rl.rllib_env import (
    CTDEParallelPettingZooEnv,
    EnvFactory,
    register_orbit_guard_env,
)

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
    vf_clip_param: float = 1.0e6
    reward_scale: float = 1.0
    reward_offset: float = 0.0
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
        **{key: rl_block[key] for key in ("vf_clip_param", "reward_scale", "reward_offset", "num_env_runners") if key in rl_block},
    }
    settings.update(overrides)
    return TrainingSettings(**settings)


def load_reward_weights(config_path: str | Path = _DEFAULT_CONFIG_PATH, **overrides: float) -> dict[str, float]:
    """Reward weights from the `reward:` block (PRD §25/§39), as `compute_reward` keyword args."""
    with Path(config_path).open() as handle:
        reward_block = yaml.safe_load(handle)["reward"]
    weights = {
        "w1": reward_block["w1_safety"],
        "w2": reward_block["w2_mission"],
        "w3": reward_block["w3_delta_v"],
        "w4": reward_block["w4_coordination"],
        "collision_penalty": reward_block["collision_penalty"],
    }
    unknown = set(overrides) - set(weights)
    if unknown:
        raise ValueError(f"unknown reward weights {sorted(unknown)}; expected {sorted(weights)}")
    weights.update(overrides)
    return weights


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
            vf_clip_param=settings.vf_clip_param,
        )
    )
    if settings.seed is not None:
        config = config.debugging(seed=settings.seed)
    return config


def build_algo(env_factory: EnvFactory, settings: TrainingSettings, env_name: str = DEFAULT_ENV_NAME) -> Algorithm:
    """Register `env_factory` and build the MAPPO/CTDE algorithm for it."""
    register_orbit_guard_env(
        env_name, env_factory, reward_scale=settings.reward_scale, reward_offset=settings.reward_offset
    )
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
    entropies: tuple[float, ...] = ()
    vf_explained_variances: tuple[float, ...] = ()


def train(algo: Algorithm, num_iterations: int, checkpoint_dir: str | Path | None = None) -> TrainingReport:
    """Run `num_iterations` PPO iterations; optionally save a checkpoint to `checkpoint_dir`."""
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")

    returns: list[float] = []
    entropies: list[float] = []
    explained_variances: list[float] = []
    result: dict[str, Any] = {}
    for _ in range(num_iterations):
        result = algo.train()
        returns.append(float(result.get("env_runners", {}).get("episode_return_mean", float("nan"))))
        learner_metrics = result.get("learners", {}).get(POLICY_ID, {})
        entropies.append(float(learner_metrics.get("entropy", float("nan"))))
        explained_variances.append(float(learner_metrics.get("vf_explained_var", float("nan"))))

    checkpoint_path = algo.save_to_path(str(checkpoint_dir)) if checkpoint_dir is not None else None
    return TrainingReport(
        iterations=num_iterations,
        episode_return_means=tuple(returns),
        env_steps_sampled=int(result.get("num_env_steps_sampled_lifetime", 0)),
        checkpoint_path=checkpoint_path,
        entropies=tuple(entropies),
        vf_explained_variances=tuple(explained_variances),
    )


def restore(env_factory: EnvFactory, settings: TrainingSettings, checkpoint_path: str | Path) -> Algorithm:
    """Rebuild the algorithm from the same config and restore weights (no code changes between train/eval)."""
    algo = build_algo(env_factory, settings)
    algo.restore_from_path(str(checkpoint_path))
    return algo


def main(argv: list[str] | None = None) -> None:
    """CLI: train MAPPO/CTDE on `OrbitGuardEnv`, then judge it against the no-maneuver baseline.

    This is the human-read convergence run (PRD §44 P3-M5): read the printed
    per-iteration returns and the gate verdict yourself; it is not CI-gated.
    """
    import argparse
    import dataclasses

    from orbit_guard.rl.environment import OrbitGuardEnv
    from orbit_guard.rl.evaluation import (
        evaluate,
        evaluate_policy,
        gate_verdict,
        no_maneuver_policy,
    )

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--eval-seed", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--num-env-runners", type=int, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--reward-offset", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    for weight in ("w1", "w2", "w3", "w4", "collision-penalty"):
        parser.add_argument(f"--{weight}", type=float, default=None)
    args = parser.parse_args(argv)

    overrides = {
        "seed": args.seed,
        "num_env_runners": args.num_env_runners,
        "reward_scale": args.reward_scale,
        "reward_offset": args.reward_offset,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "entropy_coeff": args.entropy_coeff,
        "gamma": args.gamma,
    }
    settings = load_training_settings(args.config, **{k: v for k, v in overrides.items() if v is not None})
    weight_overrides = {
        key: getattr(args, key)
        for key in ("w1", "w2", "w3", "w4", "collision_penalty")
        if getattr(args, key) is not None
    }
    reward_weights = load_reward_weights(args.config, **weight_overrides)
    print(settings, reward_weights, flush=True)

    def env_factory(config: dict) -> OrbitGuardEnv:
        return OrbitGuardEnv(num_agents=args.agents, reward_weights=reward_weights)

    algo = build_algo(env_factory, settings)
    try:
        report = train(algo, args.iterations, args.checkpoint_dir)
        for iteration, (mean_return, entropy, explained_variance) in enumerate(
            zip(report.episode_return_means, report.entropies, report.vf_explained_variances, strict=True), start=1
        ):
            print(
                f"iteration {iteration:3d}  return {mean_return:10.3f}  "
                f"entropy {entropy:6.3f}  vf_explained_var {explained_variance:7.3f}"
            )
        baseline = evaluate_policy(no_maneuver_policy, env_factory, args.eval_episodes, args.eval_seed)
        trained = evaluate(algo, env_factory, args.eval_episodes, args.eval_seed)
        budget_mps = env_factory({}).delta_v_budget_mps
        print("baseline:", dataclasses.asdict(baseline))
        print("trained: ", dataclasses.asdict(trained))
        print("verdict: ", gate_verdict(trained, baseline, budget_mps))
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
