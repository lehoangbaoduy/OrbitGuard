"""Policy evaluation and the Phase 3 gate criteria (PRD §44 "2-agent gate").

`evaluate_policy` rolls any decentralized policy -- a callable from one agent's
OWN raw observation to an action -- through the raw environment (no global
state exists on this path). `gate_verdict` then compares a learned policy with
the no-maneuver baseline on the same seeds against criteria that were fixed
BEFORE tuning, so "clean" can't drift to whatever the latest run happened to
produce:

  C1 safety       -- no more collision episodes than the baseline
  C2 efficiency   -- mean fraction of each satellite's OWN initial budget used
                     <= `max_delta_v_fraction` (with equal budgets this is the old
                     "mean delta-v used <= fraction x budget" test; CORE-3)
  C3 return       -- mean return >= the baseline's
  C4 conditioning -- maneuvers at least `min_threat_calm_ratio` times as often
                     on threatening steps as on calm ones, and on at most
                     `max_calm_maneuver_rate` of calm steps (a policy that
                     burns fuel regardless of threat is not responding to it)

A "threatening" step is one where the agent's own observation reports a
neighbor with risk_proxy above `THREATENING_RISK_THRESHOLD`.

Budgets differ per satellite (PRD §17 per-episode randomization, and the §32
capability-reduction shift), so fuel is reported as `mean_budget_fraction_used`:
per episode, the mean over satellites with a non-zero initial budget of
used / initial, then averaged over episodes (NaN when no satellite had a budget).
`unsafe_episodes` counts episodes whose `episode_summary().unsafe_conjunction_count`
is positive (PRD §36 #2), the extra term of the pre-registered R3 grid score.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
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
    mean_budget_fraction_used: float  # per-satellite used / own initial budget, averaged (NaN if no budgets)
    unsafe_episodes: int  # episodes with at least one unsafe (non-colliding, inside-margin) conjunction
    shield_intervention_rate: float = 0.0  # intervened agent-steps / agent-steps (0 without a shield)
    shield_infeasible_step_rate: float = 0.0  # steps whose approved joint action still had a violation


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


def _episode_budget_fraction(env) -> float | None:
    """Mean over satellites with a positive initial budget of used / initial; None if none has one (or no accounting)."""
    used = getattr(env, "delta_v_used_mps", None)
    initial = getattr(env, "initial_delta_v_budgets_mps", None)
    if used is None or initial is None:
        return None
    spent, budgets = used(), initial()
    fractions = [spent[agent] / budget for agent, budget in budgets.items() if budget > 0.0]
    return float(np.mean(fractions)) if fractions else None


def evaluate_policy(
    policy: Policy,
    env_factory: EnvFactory,
    num_episodes: int,
    seed: int = 0,
    *,
    seeds: Sequence[int] | None = None,
) -> EvaluationReport:
    """Roll `policy` through `num_episodes` episodes on seeds `seed .. seed+num_episodes-1`, or on `seeds` if given."""
    if seeds is not None:
        num_episodes = len(seeds)
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    episode_seeds = tuple(seeds) if seeds is not None else tuple(seed + episode for episode in range(num_episodes))

    env = env_factory({})
    returns, steps, delta_v_used, budget_fractions = [], [], [], []
    collision_episodes = unsafe_episodes = 0
    threat_steps = threat_maneuvers = calm_steps = calm_maneuvers = 0
    shield_agent_steps = shield_interventions = shield_steps = shield_infeasible_steps = 0

    for episode in range(num_episodes):
        observations, _ = env.reset(seed=episode_seeds[episode])
        episode_return, episode_steps, collided = 0.0, 0, False
        while env.agents:
            actions = {agent: policy(agent, observations[agent]) for agent in env.agents}
            for agent, action in actions.items():
                maneuvered = action != ManeuverIntent.NO_MANEUVER
                if observations[agent]["global_summary"][_MAX_RISK_INDEX] > THREATENING_RISK_THRESHOLD:
                    threat_steps, threat_maneuvers = threat_steps + 1, threat_maneuvers + maneuvered
                else:
                    calm_steps, calm_maneuvers = calm_steps + 1, calm_maneuvers + maneuvered
            observations, rewards, terminations, _, infos = env.step(actions)
            decisions = [info["shield"] for info in infos.values() if "shield" in info]
            if decisions:
                shield_steps += 1
                shield_infeasible_steps += not decisions[0]["feasible"]
                shield_agent_steps += len(decisions)
                shield_interventions += sum(bool(d["intervened"]) for d in decisions)
            episode_return += sum(rewards.values()) / max(len(rewards), 1)
            episode_steps += 1
            collided |= any(terminations.values())
        returns.append(episode_return)
        steps.append(episode_steps)
        collision_episodes += collided
        used = getattr(env, "delta_v_used_mps", None)
        delta_v_used.append(float(np.mean(list(used().values()))) if used else math.nan)
        fraction = _episode_budget_fraction(env)
        if fraction is not None:
            budget_fractions.append(fraction)
        summary = getattr(env, "episode_summary", None)
        unsafe_episodes += bool(summary and summary().unsafe_conjunction_count > 0)

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
        mean_budget_fraction_used=float(np.mean(budget_fractions)) if budget_fractions else math.nan,
        unsafe_episodes=unsafe_episodes,
        shield_intervention_rate=_rate(shield_interventions, shield_agent_steps),
        shield_infeasible_step_rate=_rate(shield_infeasible_steps, shield_steps),
    )


def evaluate(
    algo: Algorithm, env_factory: EnvFactory, num_episodes: int, seed: int = 0, *, seeds: Sequence[int] | None = None
) -> EvaluationReport:
    """Greedy rollout of the trained actor, decentralized (local observation only)."""
    return evaluate_policy(make_greedy_policy(algo, env_factory), env_factory, num_episodes, seed, seeds=seeds)


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
    criteria: GateCriteria = GateCriteria(),  # noqa: B008 - frozen dataclass default
) -> GateVerdict:
    """Apply C1-C4 (see module docstring) to a trained policy vs. the no-maneuver baseline.

    C2 reads the per-satellite budget fraction, so it needs no budget argument (a NaN fraction fails it).
    """
    checks = {
        "C1_safety": trained.collision_episodes <= baseline.collision_episodes,
        "C2_efficiency": trained.mean_budget_fraction_used <= criteria.max_delta_v_fraction,
        "C3_return": trained.mean_return_per_agent >= baseline.mean_return_per_agent,
        "C4_conditioning": (
            trained.threat_maneuver_rate >= criteria.min_threat_calm_ratio * trained.calm_maneuver_rate
            and trained.calm_maneuver_rate <= criteria.max_calm_maneuver_rate
        ),
    }
    return GateVerdict(passed=all(checks.values()), checks=checks)
