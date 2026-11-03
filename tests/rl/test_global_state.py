"""The centralized critic's global state is Markov for finite-horizon termination (PRD §28, critic-only).

Ray 2.58's MultiAgentEpisode marks an episode terminated once every agent is done, so each agent's final
row at the horizon is learned as TERMINATED and V(s_T+1) is never bootstrapped. `state()` therefore
carries the normalized time remaining (the forced terminal is then a Markov finite-horizon terminal for
the critic) and a per-agent alive mask, with a collided satellite's slot zeroed instead of frozen.

Layout: ``[slot_0 .. slot_{N-1} | alive_0 .. alive_{N-1} | time_remaining]``. Each slot is that agent's
flattened local observation (54 entries); time_remaining = 1 - steps_taken / horizon_steps, which is
1, 1-1/H, ..., 1/H at the H decision steps and 0 after the final step.

Collision fixture: seed 14 at N=4, zero sensing noise and a 60 m hard-body radius collides sat_0/sat_1
at control step 4 under all-NO_MANEUVER over a 15-step horizon (tests/rl/test_partial_termination.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces.utils import flatdim, flatten

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.rllib_env import CTDEParallelPettingZooEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel
from tests.rl.stub_env import (
    AGENT_OBS_FLAT_DIM,
    EPISODE_HORIZON_STEPS,
    StubOrbitGuardEnv,
)

_KW = {
    "episode_horizon_s": 900.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0,
    "hard_body_radius_m": 60.0, "noise_model": NoiseModel(0.0, 0.0),
}
_N = 4
_HORIZON = 15
_SEED = 14
_COLLIDED = ("sat_0", "sat_1")


@pytest.fixture(params=[OrbitGuardEnv, ShieldedOrbitGuardEnv], ids=["no_shield", "shielded"])
def env(request):
    return request.param(num_agents=_N, **_KW)


def _alive(state: np.ndarray, n: int) -> np.ndarray:
    return state[n * AGENT_OBS_FLAT_DIM : n * AGENT_OBS_FLAT_DIM + n]


def _slot(state: np.ndarray, index: int) -> np.ndarray:
    return state[index * AGENT_OBS_FLAT_DIM : (index + 1) * AGENT_OBS_FLAT_DIM]


def test_state_layout_is_slots_then_alive_mask_then_time_remaining(env):
    env.reset(seed=_SEED)
    assert flatdim(env.observation_space("sat_0")) == AGENT_OBS_FLAT_DIM
    expected_dim = _N * AGENT_OBS_FLAT_DIM + _N + 1
    assert env.state_space.shape == (expected_dim,)
    assert env.state().shape == (expected_dim,)
    assert env.state().dtype == np.float32


def test_time_remaining_runs_from_one_to_one_over_horizon_at_decisions_and_zero_after_the_last_step(env):
    env.reset(seed=_SEED)
    decision_times = []
    while env.agents:
        decision_times.append(float(env.state()[-1]))
        env.step({agent: 0 for agent in env.agents})

    np.testing.assert_allclose(decision_times, [1.0 - t / _HORIZON for t in range(_HORIZON)], atol=1e-6)
    assert decision_times[-1] == pytest.approx(1.0 / _HORIZON)
    assert float(env.state()[-1]) == 0.0


def test_alive_mask_flips_and_a_collided_slot_is_zeroed_while_survivors_keep_theirs(env):
    observations, _ = env.reset(seed=_SEED)
    np.testing.assert_array_equal(_alive(env.state(), _N), np.ones(_N))
    collided_seen = False
    while env.agents:
        observations, _, terminations, _, _ = env.step({agent: 0 for agent in env.agents})
        state = env.state()
        if any(terminations[a] for a in _COLLIDED if a in terminations):
            collided_seen = True
        if collided_seen:
            np.testing.assert_array_equal(_alive(state, _N), [0.0, 0.0, 1.0, 1.0])
            for index in (0, 1):
                np.testing.assert_array_equal(_slot(state, index), np.zeros(AGENT_OBS_FLAT_DIM))
            for index in (2, 3):
                expected = flatten(env.observation_space(f"sat_{index}"), observations[f"sat_{index}"])
                np.testing.assert_allclose(_slot(state, index), expected, atol=1e-6)
    assert collided_seen
    # Truncation at the horizon is not death: the survivors' final state keeps their slots and alive bits.
    assert np.any(_slot(env.state(), 2) != 0.0) and np.any(_slot(env.state(), 3) != 0.0)


def test_state_leaves_the_stored_last_observations_untouched_after_a_collision(env):
    env.reset(seed=_SEED)
    while env.agents:
        env.step({agent: 0 for agent in env.agents})
    for agent in _COLLIDED:
        assert np.any(flatten(env.observation_space(agent), env._last_observations[agent]) != 0.0)


def test_every_agents_wrapped_observation_carries_the_same_global_state_and_time(env):
    wrapped = CTDEParallelPettingZooEnv(env)
    observations, _ = wrapped.reset(seed=_SEED)
    step = 0
    while True:
        rows = list(observations.values())
        globals_ = [row[AGENT_OBS_FLAT_DIM:] for row in rows]
        for row in globals_[1:]:
            np.testing.assert_array_equal(row, globals_[0])
        assert float(rows[0][-1]) == pytest.approx(1.0 - step / _HORIZON)
        if not wrapped.par_env.agents:
            break
        observations, _, _, _, _ = wrapped.step({agent: 0 for agent in wrapped.par_env.agents})
        step += 1
    assert step == _HORIZON


def test_stub_env_mirrors_the_layout_with_time_remaining_and_an_all_alive_mask():
    stub = StubOrbitGuardEnv(num_agents=2)
    stub.reset(seed=0)
    assert stub.state_space.shape == (2 * AGENT_OBS_FLAT_DIM + 2 + 1,)
    times = [float(stub.state()[-1])]
    while stub.agents:
        stub.step({agent: 0 for agent in stub.agents})
        times.append(float(stub.state()[-1]))
        np.testing.assert_array_equal(_alive(stub.state(), 2), [1.0, 1.0])
    np.testing.assert_allclose(times, [1.0 - t / EPISODE_HORIZON_STEPS for t in range(EPISODE_HORIZON_STEPS + 1)])


def test_the_helper_rejects_mismatched_inputs_and_clips_time_remaining():
    from orbit_guard.rl.global_state import (
        assemble_global_state,
        global_state_dim,
        time_remaining,
    )

    assert global_state_dim(15, AGENT_OBS_FLAT_DIM) == 826
    assert time_remaining(0, 360) == 1.0 and time_remaining(360, 360) == 0.0 and time_remaining(400, 360) == 0.0
    with pytest.raises(ValueError):
        time_remaining(0, 0)
    with pytest.raises(ValueError):
        assemble_global_state([np.ones(2)], [True, False], 1.0)
    dead_inf = assemble_global_state([np.ones(2), np.full(2, np.inf)], [True, False], 0.5)
    np.testing.assert_array_equal(dead_inf, [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5])
