"""Policy evaluation and the Phase 3 gate criteria (PRD §44 "2-agent gate").

`evaluate_policy` rolls any decentralized policy -- a callable from one agent's
OWN raw observation to an action -- through the raw environment (no global
state exists on this path). `gate_verdict` then compares a learned policy with
the no-maneuver baseline on the same seeds against criteria that were fixed
BEFORE tuning, so "clean" can't drift to whatever the latest run happened to
produce:

  C1 safety       -- no more collision episodes than the baseline
  C2 efficiency   -- mean delta-v used <= `max_delta_v_fraction` of the budget
  C3 return       -- mean return >= the baseline's
  C4 conditioning -- maneuvers at least `min_threat_calm_ratio` times as often
                     on threatening steps as on calm ones, and on at most
                     `max_calm_maneuver_rate` of calm steps (a policy that
                     burns fuel regardless of threat is not responding to it)

A "threatening" step is one where the agent's own observation reports a
neighbor with risk_proxy above `THREATENING_RISK_THRESHOLD`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium.spaces.utils import flatten
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns

from orbit_guard.rl.ctde_module import POLICY_ID
from orbit_guard.rl.observation import THREATENING_RISK_THRESHOLD
from orbit_guard.rl.rllib_env import EnvFactory
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

Policy = Callable[[str, dict[str, np.ndarray]], int]
_MAX_RISK_INDEX = 1


@dataclass(frozen=True)
class EvaluationReport:
    episodes: int
    mean_return_per_agent: float
    collision_episodes: int
    maneuver_rate: float
    threat_maneuver_rate: float
    calm_maneuver_rate: float
    threat_step_fraction: float
    mean_delta_v_used_mps: float
    mean_episode_steps: float


def no_maneuver_policy(agent: str, observation: dict[str, np.ndarray]) -> int:
    return int(ManeuverIntent.NO_MANEUVER)


def make_greedy_policy(algo: Algorithm, env_factory: EnvFactory) -> Policy:
    """Decentralized execution: argmax of the actor from the LOCAL observation only.

    The module still expects the full ``[local | global]`` vector shape, so the
    global slice is zero-padded; the actor never reads it (see ctde_module).
    """
    module = algo.get_module(POLICY_ID)
    probe_env = env_factory({})
    observation_space = probe_env.observation_space(probe_env.possible_agents[0])
    global_state_dim = int(probe_env.state_space.shape[0])

    def policy(agent: str, observation: dict[str, np.ndarray]) -> int:
        local = flatten(observation_space, observation).astype(np.float32)
        padded = np.concatenate([local, np.zeros(global_state_dim, dtype=np.float32)])
        with torch.no_grad():
            outputs = module.forward_inference({Columns.OBS: torch.as_tensor(padded)[None]})
        return int(torch.argmax(outputs[Columns.ACTION_DIST_INPUTS], dim=-1).item())

    return policy


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_policy(policy: Policy, env_factory: EnvFactory, num_episodes: int, seed: int = 0) -> EvaluationReport:
    """Roll `policy` through `num_episodes` episodes on seeds `seed .. seed+num_episodes-1`."""
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")

    env = env_factory({})
    returns, steps, delta_v_used = [], [], []
    collision_episodes = 0
    threat_steps = threat_maneuvers = calm_steps = calm_maneuvers = 0

    for episode in range(num_episodes):
        observations, _ = env.reset(seed=seed + episode)
        episode_return, episode_steps, collided = 0.0, 0, False
        while env.agents:
            actions = {agent: policy(agent, observations[agent]) for agent in env.agents}
            for agent, action in actions.items():
                maneuvered = action != ManeuverIntent.NO_MANEUVER
                if observations[agent]["global_summary"][_MAX_RISK_INDEX] > THREATENING_RISK_THRESHOLD:
                    threat_steps, threat_maneuvers = threat_steps + 1, threat_maneuvers + maneuvered
                else:
                    calm_steps, calm_maneuvers = calm_steps + 1, calm_maneuvers + maneuvered
            observations, rewards, terminations, _, _ = env.step(actions)
            episode_return += sum(rewards.values()) / max(len(rewards), 1)
            episode_steps += 1
            collided |= any(terminations.values())
        returns.append(episode_return)
        steps.append(episode_steps)
        collision_episodes += collided
        used = getattr(env, "delta_v_used_mps", None)
        delta_v_used.append(float(np.mean(list(used().values()))) if used else math.nan)

    total_steps = threat_steps + calm_steps
    return EvaluationReport(
        episodes=num_episodes,
        mean_return_per_agent=float(np.mean(returns)),
        collision_episodes=collision_episodes,
        maneuver_rate=_rate(threat_maneuvers + calm_maneuvers, total_steps),
        threat_maneuver_rate=_rate(threat_maneuvers, threat_steps),
        calm_maneuver_rate=_rate(calm_maneuvers, calm_steps),
        threat_step_fraction=_rate(threat_steps, total_steps),
        mean_delta_v_used_mps=float(np.mean(delta_v_used)),
        mean_episode_steps=float(np.mean(steps)),
    )


def evaluate(algo: Algorithm, env_factory: EnvFactory, num_episodes: int, seed: int = 0) -> EvaluationReport:
    """Greedy rollout of the trained actor, decentralized (local observation only)."""
    return evaluate_policy(make_greedy_policy(algo, env_factory), env_factory, num_episodes, seed)


@dataclass(frozen=True)
class GateCriteria:
    max_delta_v_fraction: float = 0.25
    min_threat_calm_ratio: float = 2.0
    max_calm_maneuver_rate: float = 0.3


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    checks: dict[str, bool]


def gate_verdict(
    trained: EvaluationReport,
    baseline: EvaluationReport,
    delta_v_budget_mps: float,
    criteria: GateCriteria = GateCriteria(),  # noqa: B008 - frozen dataclass default
) -> GateVerdict:
    """Apply C1-C4 (see module docstring) to a trained policy vs. the no-maneuver baseline."""
    checks = {
        "C1_safety": trained.collision_episodes <= baseline.collision_episodes,
        "C2_efficiency": trained.mean_delta_v_used_mps <= criteria.max_delta_v_fraction * delta_v_budget_mps,
        "C3_return": trained.mean_return_per_agent >= baseline.mean_return_per_agent,
        "C4_conditioning": (
            trained.threat_maneuver_rate >= criteria.min_threat_calm_ratio * trained.calm_maneuver_rate
            and trained.calm_maneuver_rate <= criteria.max_calm_maneuver_rate
        ),
    }
    return GateVerdict(passed=all(checks.values()), checks=checks)
