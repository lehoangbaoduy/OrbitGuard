"""Re-measure the Fixed-Margin constant for a set of adaptive coefficients (progress Q30 protocol).

PRD §31 #3 makes Fixed-Margin MAPPO identical to Adaptive MAPPO except that `d_safe` is a constant. Q30 fixed that
constant as the mean adaptive d_safe over 12 validation scenarios at N=15 with no manoeuvres (1141.9 m, frozen as
`fixed_margin.d_safe_m: 1142.0`), measured under the DEFAULT coefficients. When the grid winner changes alpha, beta
or gamma, `tuning.grid_search.freeze_best` re-measures it here so both arms keep the same mean margin size and the
ablation still isolates adaptivity.

The protocol, reproduced exactly (it gives 1141.915 m under configs/default.yaml):
  * scenarios: the first PROTOCOL_SEED_COUNT seeds of the validation pool (configs/validation.yaml, checksum
    verified), condition `normal`, N = PROTOCOL_NUM_AGENTS;
  * the unshielded environment built from the config, with the config's adaptive margin, every satellite
    choosing NO_MANEUVER at every decision point until the episode ends;
  * every decision-time pair assessment (`env.snapshot()["pairs"]`: at reset and after each step while agents
    remain) contributes its d_safe; the result is the pooled mean (every episode has the same number of them).
With no manoeuvres and no shield, the margin changes neither the trajectories, nor the beliefs, nor any risk proxy,
so the margin INPUTS (sigma_rel, rho, risk) of those assessments are collected once and the mean is the adaptive
margin of the config evaluated on them; a collection under other coefficients yields identical inputs (tested).

VALIDATION SPLIT ONLY: the pool is read through `seed_pools.load_seed_pool(..., "validation")`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from orbit_guard.config import REPO_ROOT, env_kwargs_from_config, margin_from_config
from orbit_guard.evaluation.seed_pools import load_seed_pool
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

PROTOCOL_SEED_COUNT = 12
PROTOCOL_NUM_AGENTS = 15
# Spawned children re-import the parent's __main__; under the grid CLI that pulls in torch and Ray, so the
# default stays within the 6-process envelope the N=15 training already runs in (prereg RAM note).
DEFAULT_MAX_WORKERS = 6
_VALIDATION_CONFIG = REPO_ROOT / "configs" / "validation.yaml"


def validation_seeds(validation_config: str | Path = _VALIDATION_CONFIG) -> tuple[int, ...]:
    """The protocol's scenarios: the first PROTOCOL_SEED_COUNT seeds of the checksummed validation pool."""
    spec = yaml.safe_load(Path(validation_config).read_text())
    pool_path = Path(spec["seed_pool_file"])
    if not pool_path.is_absolute():
        pool_path = REPO_ROOT / pool_path
    return load_seed_pool(pool_path, "validation", spec.get("seed_pool_sha256"))[:PROTOCOL_SEED_COUNT]


def round_fixed_margin(raw_mean_m: float) -> float:
    """Nearest whole metre, halves up: how Q30 turned 1141.9 m into the frozen 1142.0."""
    return float(math.floor(raw_mean_m + 0.5))


def measure_fixed_margin(config: Mapping[str, Any], **protocol: Any) -> float:
    """The raw (unrounded) Q30 mean adaptive d_safe under `config`'s coefficients, in metres.

    `protocol` keywords go to `no_manoeuvre_margin_inputs` (seeds, num_agents, workers); the defaults are Q30's.
    """
    return mean_adaptive_margin(config, no_manoeuvre_margin_inputs(config, **protocol))


def mean_adaptive_margin(config: Mapping[str, Any], inputs: np.ndarray) -> float:
    """Mean of `config`'s adaptive margin over (sigma_rel_m, rho, risk) rows."""
    if len(inputs) == 0:
        raise ValueError("no margin inputs: the protocol produced no pair assessment")
    margin = margin_from_config(config, "adaptive")
    return float(np.mean([margin(float(s), float(rho), float(risk)) for s, rho, risk in inputs]))


def no_manoeuvre_margin_inputs(
    config: Mapping[str, Any],
    *,
    seeds: Sequence[int] | None = None,
    num_agents: int = PROTOCOL_NUM_AGENTS,
    max_decisions: int | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """(sigma_rel_m, rho, risk) of every decision-time pair assessment of the protocol, in seed order.

    `max_decisions` caps the decision points per episode (None = the whole episode, the protocol). Episodes run
    in `workers` spawned processes (default: one per seed, at most DEFAULT_MAX_WORKERS and the CPU count); the
    rows do not depend on it.
    """
    seeds = validation_seeds() if seeds is None else tuple(int(seed) for seed in seeds)
    jobs = [(dict(config), seed, num_agents, max_decisions) for seed in seeds]
    default = min(len(jobs), DEFAULT_MAX_WORKERS, os.cpu_count() or 1)
    count = default if workers is None else max(1, min(workers, len(jobs)))
    if count <= 1:
        episodes = [_episode_margin_inputs(job) for job in jobs]
    else:
        with get_context("spawn").Pool(count) as pool:
            episodes = pool.map(_episode_margin_inputs, jobs)
    rows = np.concatenate(episodes) if episodes else np.empty((0, 3))
    rows.setflags(write=False)
    return rows


class _RecordingMargin:
    """The config's margin, unchanged, remembering the inputs and value of every call since the last `take`."""

    def __init__(self, margin):
        self._margin = margin
        self._calls: list[tuple[float, float, float, float]] = []

    def __call__(self, sigma_rel_m: float, rho: float, risk: float) -> float:
        value = self._margin(sigma_rel_m, rho, risk)
        self._calls.append((float(sigma_rel_m), float(rho), float(risk), float(value)))
        return value

    def take_decision_inputs(self, pairs: Sequence[tuple]) -> list[tuple[float, float, float]]:
        """The inputs behind `pairs` (the env's decision-time assessments, computed by the last margin calls).

        The environment assesses the decision-time pairs last in reset and in step; the recorded values are
        checked against the assessments' own d_safe, so a change in that call order fails loudly.
        """
        recent = self._calls[len(self._calls) - len(pairs):] if pairs else []
        if [call[3] for call in recent] != [float(pair[3]) for pair in pairs]:
            raise RuntimeError("could not match the recorded margin calls to the decision-time pair assessments")
        self._calls = []
        return [call[:3] for call in recent]


def _episode_margin_inputs(job: tuple[dict, int, int, int | None]) -> np.ndarray:
    """One no-manoeuvre, unshielded episode's decision-time margin inputs (a picklable worker function)."""
    from orbit_guard.rl.environment import OrbitGuardEnv

    config, seed, num_agents, max_decisions = job
    recorder = _RecordingMargin(margin_from_config(config, "adaptive"))
    env = OrbitGuardEnv(num_agents=num_agents, margin_fn=recorder, **env_kwargs_from_config(config, shielded=False))
    env.reset(seed=seed)
    rows = recorder.take_decision_inputs(env.snapshot()["pairs"])
    decisions = 1
    while env.agents and (max_decisions is None or decisions < max_decisions):
        env.step({agent: int(ManeuverIntent.NO_MANEUVER) for agent in env.agents})
        if env.agents:
            rows.extend(recorder.take_decision_inputs(env.snapshot()["pairs"]))
            decisions += 1
    return np.asarray(rows, dtype=float).reshape(-1, 3)
