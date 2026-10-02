"""Partial termination at N>2: one colliding pair ends, the rest play on (needed before N=15 training).

At N=2 any collision ends the episode, so this path is never exercised there.
Seed 14 at N=4 with zero sensing noise and a 60 m hard-body radius collides
sat_0/sat_1 at control step 4 under all-NO_MANEUVER; sat_2/sat_3 continue.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.rllib_env import CTDEParallelPettingZooEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel

_KW = {
    "episode_horizon_s": 900.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0,
    "hard_body_radius_m": 60.0, "noise_model": NoiseModel(0.0, 0.0),
}
_SEED = 14


@pytest.fixture(params=[OrbitGuardEnv, ShieldedOrbitGuardEnv], ids=["no_shield", "shielded"])
def env(request):
    return request.param(num_agents=4, **_KW)


def test_one_pair_terminates_and_the_others_continue_to_truncation(env):
    env.reset(seed=_SEED)
    state_shape = env.state().shape
    terminated_at = {}
    steps = 0
    while env.agents:
        _, rewards, terminations, truncations, _ = env.step({agent: 0 for agent in env.agents})
        steps += 1
        assert env.state().shape == state_shape
        for agent, done in terminations.items():
            if done:
                terminated_at.setdefault(agent, steps)
        assert set(rewards) == set(terminations) == set(truncations)

    assert set(terminated_at) == {"sat_0", "sat_1"}
    assert steps == round(_KW["episode_horizon_s"] / _KW["control_dt_s"])
    assert max(terminated_at.values()) < steps


def test_ctde_wrapper_keeps_fixed_dimensions_through_a_partial_termination(env):
    wrapped = CTDEParallelPettingZooEnv(env)
    observations, _ = wrapped.reset(seed=_SEED)
    dimension = next(iter(observations.values())).shape
    all_done = False
    while wrapped.par_env.agents:
        actions = {agent: 0 for agent in wrapped.par_env.agents}
        observations, _, terminations, truncations, _ = wrapped.step(actions)
        for obs in observations.values():
            assert obs.shape == dimension
            assert np.all(np.isfinite(obs))
        all_done = terminations["__all__"] or truncations["__all__"]
        assert all_done == (not wrapped.par_env.agents)

    assert all_done
