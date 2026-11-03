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
    `rl.critic_input` (or `--critic-input`) selects the critic's input: the
    global state alone (`global`, default) or `[own local | global]`
    (`agent_global`), the two arms of the pre-registered critic A/B (R4).
  * Dict observations are flattened to a Box first (see `rllib_env`).

Config-driven: hyperparameters come from `configs/default.yaml`'s `rl:` block
(PRD §28/§39), so every learned baseline trains from a config swap alone.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from orbit_guard.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    config_digest,
    load_config,
    margin_from_config,
)
from orbit_guard.rl.actor_policy import export_from_algo
from orbit_guard.rl.ctde_module import (
    CRITIC_INPUTS,
    DEFAULT_ACTOR_HIDDEN_LAYERS,
    DEFAULT_CRITIC_HIDDEN_LAYERS,
    DEFAULT_CRITIC_INPUT,
    POLICY_ID,
    CentralizedCriticTorchRLModule,
    validate_critic_input,
)
from orbit_guard.rl.env_factory import (
    load_margin_fn,
    load_reward_weights,
    make_env_factory,  # noqa: F401  (re-exported: callers import it from here)
    make_env_factory_from_config,
)
from orbit_guard.rl.rllib_env import (
    CTDEParallelPettingZooEnv,
    EnvFactory,
    register_orbit_guard_env,
)
from orbit_guard.safety.adaptive_margin import AdaptiveMargin, FixedMargin

DEFAULT_ENV_NAME = "orbit_guard_ctde"
ACTOR_WEIGHTS_FILENAME = "actor_weights.npz"
RUN_CONFIG_FILENAME = "run_config.json"
_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH


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
    critic_input: str = DEFAULT_CRITIC_INPUT

    def __post_init__(self) -> None:
        validate_critic_input(self.critic_input)


def load_training_settings(
    config_path: str | Path = _DEFAULT_CONFIG_PATH, overlay_path: str | Path | None = None, **overrides: Any
) -> TrainingSettings:
    """Read the `rl:` block of the config (and of a tuned overlay, if given); `overrides` win over both."""
    rl_block = load_config(config_path, overlay_path)["rl"]
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
        "critic_input": rl_block["critic_input"],
        **{key: rl_block[key] for key in ("vf_clip_param", "reward_scale", "reward_offset", "num_env_runners") if key in rl_block},
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
            "critic_input": settings.critic_input,
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


def build_run_config(
    *,
    config_path: str | Path,
    overlay_path: str | Path | None,
    reward_weights: dict,
    margin_fn: Any,
    shield: bool,
    settings: TrainingSettings,
    num_agents: int,
    git_commit: str | None,
) -> dict:
    """Provenance record of one training run: which config, overlay, reward, margin and settings produced it."""
    return {
        "config_path": str(config_path),
        "overlay_path": None if overlay_path is None else str(overlay_path),
        "config_digest": config_digest(config_path, overlay_path),
        "reward_weights": dict(reward_weights),
        "margin": _margin_record(margin_fn),
        "shield": bool(shield),
        "training": dataclasses.asdict(settings),
        "seed": settings.seed,
        "num_agents": num_agents,
        "git_commit": git_commit,
    }


def _margin_record(margin_fn: Any) -> dict:
    if isinstance(margin_fn, FixedMargin):
        return {"kind": "fixed", "d_safe_m": margin_fn.d_safe_m}
    if isinstance(margin_fn, AdaptiveMargin):
        return {"kind": "adaptive", **dataclasses.asdict(margin_fn)}
    raise ValueError(f"cannot record margin of type {type(margin_fn).__name__}")


def write_run_config(checkpoint_dir: str | Path, run_config: dict) -> Path:
    """Write the provenance sidecar next to the checkpoint's actor weights."""
    path = Path(checkpoint_dir) / RUN_CONFIG_FILENAME
    path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")
    return path


def _git_commit() -> str | None:
    """HEAD of the checkout the run came from; None when git or the repository is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


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


def train(
    algo: Algorithm,
    num_iterations: int,
    checkpoint_dir: str | Path | None = None,
    on_iteration: Callable[[int, float, float, float], None] | None = None,
) -> TrainingReport:
    """Run `num_iterations` PPO iterations; optionally save a checkpoint to `checkpoint_dir`.

    `on_iteration(iteration, mean_return, entropy, vf_explained_variance)` fires after each
    iteration so long runs can be watched live (the CLI prints through it).
    """
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
        if on_iteration is not None:
            on_iteration(len(returns), returns[-1], entropies[-1], explained_variances[-1])

    checkpoint_path = algo.save_to_path(str(checkpoint_dir)) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        # Ray-free copy of the trained actor, so evaluation can fan out over plain worker processes.
        export_from_algo(algo, Path(checkpoint_path) / ACTOR_WEIGHTS_FILENAME)
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
    parser.add_argument(
        "--overlay", default=None,
        help="tuned overlay (tuning/grid_search tuned.yaml) merged over --config; explicit flags still win",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--num-env-runners", type=int, default=None)
    parser.add_argument(
        "--margin", choices=("adaptive", "fixed"), default="adaptive",
        help="adaptive margin (PRD §22) or the Fixed-Margin baseline's constant from `fixed_margin:` in the config",
    )
    parser.add_argument(
        "--shield", action=argparse.BooleanOptionalAction, default=True,
        help="train/evaluate with the joint-action safety shield active (default); --no-shield is a separate run",
    )
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--reward-offset", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--critic-input", choices=CRITIC_INPUTS, default=None,
        help="critic input: global state only, or [own local | global] (prereg R4); default: rl.critic_input",
    )
    for weight in ("w1", "w2", "w3", "w4", "collision-penalty"):
        parser.add_argument(f"--{weight}", type=float, default=None)
    for coefficient in ("alpha", "beta", "gamma-margin"):
        parser.add_argument(f"--margin-{coefficient}", type=float, default=None)
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
        "critic_input": args.critic_input,
    }
    settings = load_training_settings(
        args.config, args.overlay, **{k: v for k, v in overrides.items() if v is not None}
    )
    weight_overrides = {
        key: getattr(args, key)
        for key in ("w1", "w2", "w3", "w4", "collision_penalty")
        if getattr(args, key) is not None
    }
    reward_weights = load_reward_weights(args.config, args.overlay, **weight_overrides)
    print(settings, reward_weights, f"shield={args.shield} margin={args.margin}", flush=True)

    margin_overrides = {
        name: getattr(args, f"margin_{name}")
        for name in ("alpha", "beta")
        if getattr(args, f"margin_{name}") is not None
    }
    if args.margin_gamma_margin is not None:
        margin_overrides["gamma"] = args.margin_gamma_margin
    config = load_config(args.config, args.overlay)
    if args.margin == "fixed":
        margin_fn = margin_from_config(config, "fixed")
    else:
        margin_fn = load_margin_fn(args.config, args.overlay, **margin_overrides)
    env_factory = make_env_factory_from_config(
        config, args.agents, shielded=args.shield, margin_fn=margin_fn, reward_weights=reward_weights,
    )

    algo = build_algo(env_factory, settings)
    try:

        def print_iteration(iteration: int, mean_return: float, entropy: float, explained_variance: float) -> None:
            print(
                f"iteration {iteration:3d}  return {mean_return:10.3f}  "
                f"entropy {entropy:6.3f}  vf_explained_var {explained_variance:7.3f}",
                flush=True,
            )

        report = train(algo, args.iterations, args.checkpoint_dir, on_iteration=print_iteration)
        if report.checkpoint_path is not None:
            run_config = build_run_config(
                config_path=args.config, overlay_path=args.overlay, reward_weights=reward_weights,
                margin_fn=margin_fn, shield=args.shield, settings=settings, num_agents=args.agents,
                git_commit=_git_commit(),
            )
            print("run config:", write_run_config(report.checkpoint_path, run_config), flush=True)
        baseline = evaluate_policy(no_maneuver_policy, env_factory, args.eval_episodes, args.eval_seed)
        trained = evaluate(algo, env_factory, args.eval_episodes, args.eval_seed)
        print("baseline:", dataclasses.asdict(baseline), flush=True)
        print("trained: ", dataclasses.asdict(trained), flush=True)
        print("verdict: ", gate_verdict(trained, baseline), flush=True)
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
