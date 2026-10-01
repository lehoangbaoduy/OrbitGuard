"""Pre-verified RLlib integration helper for wiring a Dict-observation
PettingZoo ParallelEnv (this project's shape, per PRD §24) into Ray 2.58's
CURRENT ("new API stack") RLlib.

Why this exists: RLlib's new RLModule/Catalog stack has no default encoder
for a `Dict` observation space containing multiple `Box` sub-spaces --
`PPOConfig().environment(...).build()` raises
`ValueError: No default encoder config for obs space=Dict(...)` immediately
on `algo.train()`, confirmed empirically against this project's installed
Ray 2.58 on 2026-09-30 (see docs/OrbitGuard_progress.md's Phase 3 section).
This is exactly the kind of RLlib-version-specific integration detail PRD
§28/design-decision-log item 6 warns must be re-verified against current
docs at Phase 3, rather than assumed from a cached tutorial written against
Ray 2.9.

The fix verified against 2.58: flatten each agent's Dict observation into a
single Box via `gymnasium.spaces.utils.flatten`/`flatten_space` in a thin
wrapper around `ray.rllib.env.wrappers.pettingzoo_env.ParallelPettingZooEnv`,
so the RLModule catalog only ever sees plain Box spaces. This throws away
the Dict's field boundaries for the NETWORK's input layer, not for the
codebase's own contract -- `orbit_guard.rl.observation.build_observation`
still returns the structured Dict; only the env-registration layer
flattens it before handing observations to RLlib.

If a future RLlib version restores/changes Dict-space auto-encoding, or if
Part C instead wants a custom `RLModule`/`Catalog` with separate encoder
branches per field (own/neighbors/global_summary) -- a reasonable
architecture choice, not required for the Phase 3 gate -- that decision
replaces this wrapper; it does not have to survive Phase 3's sanity check.

This module lives under `tests/` and is imported by
`tests/acceptance/test_acceptance_d_training_smoke.py` only. PRODUCTION
`orbit_guard/rl/training.py` must NOT import from `tests.rl.*` -- Part C
should port or extend this verified pattern into its own production module
(e.g. `orbit_guard/rl/rllib_env.py`), especially since the real centralized
critic (PRD §28's CTDE requirement, not satisfied by this flatten wrapper
alone -- see `tests/rl/stub_env.py`'s `state()`/`state_space`) will likely
need a different or extended wrapper anyway.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.spaces.utils import flatten, flatten_space
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


class FlattenedParallelPettingZooEnv(ParallelPettingZooEnv):
    """`ParallelPettingZooEnv` with each agent's Dict observation flattened
    to a single Box, so RLlib's new-API-stack default encoder can build."""

    def __init__(self, env):
        super().__init__(env)
        self._inner_observation_spaces = dict(self.observation_space.spaces)
        self.observation_space = gym.spaces.Dict(
            {agent_id: flatten_space(space) for agent_id, space in self._inner_observation_spaces.items()}
        )

    def _flatten_all(self, observations: dict) -> dict:
        return {
            agent_id: flatten(self._inner_observation_spaces[agent_id], obs)
            for agent_id, obs in observations.items()
        }

    def reset(self, *, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        return self._flatten_all(observations), infos

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = super().step(action_dict)
        return self._flatten_all(observations), rewards, terminations, truncations, infos
