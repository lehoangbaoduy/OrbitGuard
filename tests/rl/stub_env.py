"""Minimal 2-agent PettingZoo ParallelEnv stub for bootstrapping Part C's
`orbit_guard.rl.training` (PRD §28/§29) BEFORE Part A's real
`orbit_guard.rl.environment` exists.

Frozen to exactly the observation/action SHAPES that `orbit_guard.rl.observation`
(Part B, PRD §24) and `orbit_guard.safety.maneuver_mapper` (Part B, PRD §23)
will produce/accept once implemented -- see tests/rl/test_observation.py and
tests/safety/test_maneuver_mapper.py for those frozen contracts. This stub
duplicates the raw numbers (10, 4, 10, 4, 6) rather than importing those
not-yet-implemented modules, by design: Part C must not depend on Part A's
or Part B's production code landing first, only on the shapes all three
parts have already agreed to freeze.

Dynamics here are DELIBERATELY trivial (random noise, no physics, no risk
model) -- this exists only to exercise an RLlib/PettingZoo training loop
mechanically end-to-end. It is NOT a substitute for the real OrbitGuardEnv
and must never be used for an actual convergence/evaluation run (that is
P3-M5, a separate, human-reviewed, non-CI-gated task -- see the Phase 3
Antigravity brief).

PRD §28 requires MAPPO/CTDE, not just a parameter-shared policy: a shared
actor that sees only its own agent's local observation, PLUS a centralized
critic that may see global information during training -- with the hard
rule that no critic-only feature leaks into the decentralized actor at
execution time. Parameter sharing alone (one network, same weights, each
agent's own local observation) is IPPO, not CTDE; it has no centralized
critic and satisfies PRD §28 only partially. This stub exposes the
centralized critic's input via `state()`/`state_space`, frozen to the
simplest documented MAPPO-CTDE convention -- concatenation of every
agent's own (flattened) local observation, in `possible_agents` order --
so Part C has a concrete, fixed-shape target to build a real centralized
critic against. Part A's real `OrbitGuardEnv` must expose the identical
`state()`/`state_space` interface so Part C's training code is a swap-in,
not a rewrite.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
from pettingzoo import ParallelEnv

K_NEIGHBORS = 4
OWN_DIM = 10
NEIGHBOR_DIM = 10
GLOBAL_SUMMARY_DIM = 4
NUM_ACTIONS = 6  # matches ManeuverIntent's 6 members (PRD §23)
EPISODE_HORIZON_STEPS = 10
AGENT_OBS_FLAT_DIM = OWN_DIM + K_NEIGHBORS * NEIGHBOR_DIM + GLOBAL_SUMMARY_DIM  # 54


class StubOrbitGuardEnv(ParallelEnv):
    metadata: ClassVar[dict] = {"name": "stub_orbit_guard_env_v0", "is_parallelizable": True}

    def __init__(self, num_agents: int = 2, episode_horizon_steps: int = EPISODE_HORIZON_STEPS):
        if num_agents < 2:
            raise ValueError("OrbitGuard is a multi-agent env; num_agents must be >= 2")
        self.possible_agents = [f"sat_{i}" for i in range(num_agents)]
        self.agents: list[str] = list(self.possible_agents)
        self._episode_horizon_steps = episode_horizon_steps
        self._step_count = 0
        self._rng = np.random.default_rng()
        # All agents share one homogeneous space -- cached once here (not via
        # lru_cache on the method, which ruff/B019 flags as a leak risk on
        # instance methods) so PettingZoo's identity check on repeated calls
        # (`observation_space(agent) is observation_space(agent)`) is satisfied.
        self._observation_space = spaces.Dict(
            {
                "own": spaces.Box(low=-np.inf, high=np.inf, shape=(OWN_DIM,), dtype=np.float32),
                "neighbors": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(K_NEIGHBORS, NEIGHBOR_DIM), dtype=np.float32
                ),
                "global_summary": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(GLOBAL_SUMMARY_DIM,), dtype=np.float32
                ),
            }
        )
        self._action_space = spaces.Discrete(NUM_ACTIONS)
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_agents * AGENT_OBS_FLAT_DIM,), dtype=np.float32
        )
        self._last_observations: dict = {}

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def state(self) -> np.ndarray:
        """Centralized-critic global state (PRD §28's CTDE requirement) --
        the concatenation of every agent's own flattened local observation,
        in fixed `possible_agents` order. This is the ONLY place global
        information may come from; the actor network must never see it."""
        flattened = [
            flatten(self._observation_space, self._last_observations[agent]) for agent in self.possible_agents
        ]
        return np.concatenate(flattened).astype(np.float32)

    def _sample_observation(self):
        return {
            "own": self._rng.standard_normal(OWN_DIM).astype(np.float32),
            "neighbors": self._rng.standard_normal((K_NEIGHBORS, NEIGHBOR_DIM)).astype(np.float32),
            "global_summary": self._rng.standard_normal(GLOBAL_SUMMARY_DIM).astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        observations = {agent: self._sample_observation() for agent in self.agents}
        self._last_observations = dict(observations)
        infos: dict = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        for agent, action in actions.items():
            if action not in range(NUM_ACTIONS):
                raise ValueError(f"action {action} out of range for agent {agent}")
        self._step_count += 1
        truncated = self._step_count >= self._episode_horizon_steps
        observations = {agent: self._sample_observation() for agent in self.agents}
        self._last_observations.update(observations)
        rewards = {agent: float(self._rng.standard_normal()) for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos: dict = {agent: {} for agent in self.agents}
        if truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        raise NotImplementedError(
            "StubOrbitGuardEnv has no rendering; it exists for training-loop bootstrapping only."
        )
