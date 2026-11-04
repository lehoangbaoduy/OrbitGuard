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

An episode that raises is logged to stderr with its seed and counted in
`failed_episodes`; `episodes` stays the number ATTEMPTED and every other field
(counts, rates, means) covers only the `ok_episodes` that finished. With no
failure the report is exactly what it was before failures were tracked.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from gymnasium.spaces.utils import flatten

from orbit_guard.rl.observation import THREATENING_RISK_THRESHOLD
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

if TYPE_CHECKING:  # annotations only: a runtime import would load torch and ray in every evaluation worker
    from ray.rllib.algorithms.algorithm import Algorithm

    from orbit_guard.rl.rllib_env import EnvFactory

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
    failed_episodes: int = 0  # attempted episodes that raised; every other field covers only the finished ones

    @property
    def ok_episodes(self) -> int:
        """Episodes that finished: the denominator of every per-episode rate (`episodes` counts attempts)."""
        return self.episodes - self.failed_episodes


def no_maneuver_policy(agent: str, observation: dict[str, np.ndarray]) -> int:
    return int(ManeuverIntent.NO_MANEUVER)


def make_greedy_policy(algo: Algorithm, env_factory: EnvFactory) -> Policy:
    """Decentralized execution: argmax of the actor from the LOCAL observation only.

    The module still expects the full ``[local | global]`` vector shape, so the
    global slice (alive mask and time remaining included) is zero-padded; the
    actor never reads it (see ctde_module).
    """
    import torch
    from ray.rllib.core.columns import Columns

    from orbit_guard.rl.ctde_module import POLICY_ID

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


# Per-episode integer counters, summed over finished episodes (field names of _EpisodeOutcome).
_COUNTERS = (
    "steps", "threat_steps", "threat_maneuvers", "calm_steps", "calm_maneuvers",
    "shield_steps", "shield_infeasible_steps", "shield_agent_steps", "shield_interventions",
)


@dataclass(frozen=True)
class _EpisodeOutcome:
    """Everything one finished episode contributes to an EvaluationReport (merged only if it finished)."""

    episode_return: float
    steps: int
    collided: bool
    unsafe: bool
    delta_v_used: float
    budget_fraction: float | None
    threat_steps: int
    threat_maneuvers: int
    calm_steps: int
    calm_maneuvers: int
    shield_steps: int
    shield_infeasible_steps: int
    shield_agent_steps: int
    shield_interventions: int


def _run_episode(policy: Policy, env, episode_seed: int) -> _EpisodeOutcome:
    """Roll one episode; raises whatever the env or the policy raises (the caller decides what a failure means)."""
    observations, _ = env.reset(seed=episode_seed)
    counts = dict.fromkeys(_COUNTERS, 0)
    episode_return, collided = 0.0, False
    while env.agents:
        actions = {agent: policy(agent, observations[agent]) for agent in env.agents}
        for agent, action in actions.items():
            maneuvered = int(action != ManeuverIntent.NO_MANEUVER)
            side = "threat" if observations[agent]["global_summary"][_MAX_RISK_INDEX] > THREATENING_RISK_THRESHOLD else "calm"
            counts[f"{side}_steps"] += 1
            counts[f"{side}_maneuvers"] += maneuvered
        observations, rewards, terminations, _, infos = env.step(actions)
        decisions = [info["shield"] for info in infos.values() if "shield" in info]
        if decisions:
            counts["shield_steps"] += 1
            counts["shield_infeasible_steps"] += not decisions[0]["feasible"]
            counts["shield_agent_steps"] += len(decisions)
            counts["shield_interventions"] += sum(bool(d["intervened"]) for d in decisions)
        episode_return += sum(rewards.values()) / max(len(rewards), 1)
        counts["steps"] += 1
        collided |= any(terminations.values())
    used = getattr(env, "delta_v_used_mps", None)
    summary = getattr(env, "episode_summary", None)
    return _EpisodeOutcome(
        episode_return=episode_return,
        collided=collided,
        unsafe=bool(summary and summary().unsafe_conjunction_count > 0),
        delta_v_used=float(np.mean(list(used().values()))) if used else math.nan,
        budget_fraction=_episode_budget_fraction(env),
        **counts,
    )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _report(outcomes: Sequence[_EpisodeOutcome], episodes: int) -> EvaluationReport:
    """Aggregate the FINISHED episodes; `episodes` is the attempted count, the difference is `failed_episodes`."""
    total = {name: sum(getattr(outcome, name) for outcome in outcomes) for name in _COUNTERS}
    fractions = [o.budget_fraction for o in outcomes if o.budget_fraction is not None]
    agent_steps = total["threat_steps"] + total["calm_steps"]
    return EvaluationReport(
        episodes=episodes,
        mean_return_per_agent=_mean([o.episode_return for o in outcomes]),
        collision_episodes=sum(o.collided for o in outcomes),
        maneuver_rate=_rate(total["threat_maneuvers"] + total["calm_maneuvers"], agent_steps),
        threat_maneuver_rate=_rate(total["threat_maneuvers"], total["threat_steps"]),
        calm_maneuver_rate=_rate(total["calm_maneuvers"], total["calm_steps"]),
        threat_step_fraction=_rate(total["threat_steps"], agent_steps),
        mean_delta_v_used_mps=_mean([o.delta_v_used for o in outcomes]),
        mean_episode_steps=_mean([o.steps for o in outcomes]),
        mean_budget_fraction_used=_mean(fractions),
        unsafe_episodes=sum(o.unsafe for o in outcomes),
        shield_intervention_rate=_rate(total["shield_interventions"], total["shield_agent_steps"]),
        shield_infeasible_step_rate=_rate(total["shield_infeasible_steps"], total["shield_steps"]),
        failed_episodes=episodes - len(outcomes),
    )


def evaluate_policy(
    policy: Policy,
    env_factory: EnvFactory,
    num_episodes: int,
    seed: int = 0,
    *,
    seeds: Sequence[int] | None = None,
) -> EvaluationReport:
    """Roll `policy` through `num_episodes` episodes on seeds `seed .. seed+num_episodes-1`, or on `seeds` if given.

    An episode that raises (any Exception, from the env or the policy) is logged to stderr with its seed, counted
    in `failed_episodes` and left out of every count, rate and mean; the env is rebuilt before the next episode.
    """
    if seeds is not None:
        num_episodes = len(seeds)
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    episode_seeds = tuple(seeds) if seeds is not None else tuple(seed + episode for episode in range(num_episodes))

    env = env_factory({})
    outcomes = []
    for episode_seed in episode_seeds:
        try:
            outcomes.append(_run_episode(policy, env, episode_seed))
        except Exception as error:  # noqa: BLE001 - one bad episode must not abort the evaluation (counted, logged)
            print(
                f"[evaluate_policy] episode seed {episode_seed} failed: {type(error).__name__}: {error}",
                file=sys.stderr, flush=True,
            )
            env = env_factory({})  # never reuse an env left mid-step
    return _report(outcomes, num_episodes)


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


def _no_less_safe(trained: EvaluationReport, baseline: EvaluationReport) -> bool:
    """C1. Without failed episodes: collision-episode COUNTS (the original, frozen rule, bit-identical).

    With failed episodes on either side: collision-episode RATES over the finished episodes, so crashing on seeds
    cannot make a policy look safer; a side that finished no episode cannot be judged and fails C1.
    """
    if trained.failed_episodes == 0 and baseline.failed_episodes == 0:
        return trained.collision_episodes <= baseline.collision_episodes
    if trained.ok_episodes < 1 or baseline.ok_episodes < 1:
        return False
    return trained.collision_episodes / trained.ok_episodes <= baseline.collision_episodes / baseline.ok_episodes


def gate_verdict(
    trained: EvaluationReport,
    baseline: EvaluationReport,
    criteria: GateCriteria = GateCriteria(),  # noqa: B008 - frozen dataclass default
) -> GateVerdict:
    """Apply C1-C4 (see module docstring) to a trained policy vs. the no-maneuver baseline.

    C2 reads the per-satellite budget fraction, so it needs no budget argument (a NaN fraction fails it).
    """
    checks = {
        "C1_safety": _no_less_safe(trained, baseline),
        "C2_efficiency": trained.mean_budget_fraction_used <= criteria.max_delta_v_fraction,
        "C3_return": trained.mean_return_per_agent >= baseline.mean_return_per_agent,
        "C4_conditioning": (
            trained.threat_maneuver_rate >= criteria.min_threat_calm_ratio * trained.calm_maneuver_rate
            and trained.calm_maneuver_rate <= criteria.max_calm_maneuver_rate
        ),
    }
    return GateVerdict(passed=all(checks.values()), checks=checks)
