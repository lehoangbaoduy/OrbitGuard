"""The shield's MISSION_LIMIT input is the satellite's own maneuver deviation, not sensing noise.

Regression for the wave-4 review finding "Shield MISSION_LIMIT check is driven
by sensing noise": the shielded env used to pass the belief-vs-shadow deviation
(a fresh noisy belief against a shadow seeded from a noisy reset snapshot), so
under the noise_increase shift satellites that never burned exceeded the limit
from noise alone and had benign burns rejected. Only VALIDATION seeds are used.
"""

from __future__ import annotations

import numpy as np

from orbit_guard.evaluation.shifts import condition_env_kwargs
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.maneuver_mapper import ManeuverIntent
from orbit_guard.sensing.noise import NoiseModel

_VALIDATION_SEED = 100002  # the review's probe seed (p8): pure-noise deviation > 1.0 at N=15, 3x noise
_PHASE_BACKWARD = int(ManeuverIntent.PHASE_BACKWARD)
_RAISE_ORBIT = int(ManeuverIntent.RAISE_ORBIT)


def _violation_kinds(env: ShieldedOrbitGuardEnv, index: int) -> set[str]:
    return {v.kind.value for v in env.last_shield_result.proposal_evaluation.violations if index in v.agents}


def _no_op(env) -> dict:
    return {agent: 0 for agent in env.agents}


def test_noise_only_deviation_above_the_limit_no_longer_rejects_a_benign_burn():
    env = ShieldedOrbitGuardEnv(num_agents=15, **condition_env_kwargs("noise_increase"))
    env.reset(seed=_VALIDATION_SEED)
    while env.agents:
        used = env.delta_v_used_mps()
        noisy = [
            i for i, agent in enumerate(env.possible_agents)
            if agent in env.agents and used[agent] == 0.0 and env._pending_mission_deviations[i] > 1.0
        ]
        if noisy:
            break
        env.step(_no_op(env))
    assert env.agents and noisy, "the probe scenario no longer reaches a noise-only deviation above 1.0"
    index = noisy[0]

    env.step({**_no_op(env), env.possible_agents[index]: _PHASE_BACKWARD})

    assert "mission_limit" not in _violation_kinds(env, index)


def test_a_satellite_that_never_burned_has_exactly_zero_maneuver_deviation():
    env = ShieldedOrbitGuardEnv(num_agents=15, episode_horizon_s=600.0, **condition_env_kwargs("noise_increase"))
    env.reset(seed=_VALIDATION_SEED)
    assert env.maneuver_mission_deviations() == (0.0,) * 15
    applied = np.zeros(15)

    while env.agents:
        env.step(_no_op(env))
        applied += env.last_step["applied_dv_mps"]
        deviations = env.maneuver_mission_deviations()
        for i in range(15):
            if applied[i] == 0.0:
                assert deviations[i] == 0.0
            else:
                assert deviations[i] > 0.0
    assert max(env._pending_mission_deviations) > 0.0  # the noisy belief deviation the shield no longer reads


def test_maneuver_deviation_is_independent_of_the_noise_realisation():
    """Same shadow, same burns, two noise models: identical maneuver deviation; the belief deviation differs."""
    kw = {"num_agents": 3, "episode_horizon_s": 600.0}
    quiet = ShieldedOrbitGuardEnv(noise_model=NoiseModel(100.0, 0.1), **kw)
    loud = ShieldedOrbitGuardEnv(noise_model=NoiseModel(300.0, 0.3), **kw)
    quiet.reset(seed=_VALIDATION_SEED)
    loud.reset(seed=_VALIDATION_SEED)
    # The reset shadow is a noisy snapshot; give both the same one so only the noise stream differs.
    loud._shadow_states = [state.copy() for state in quiet._shadow_states]
    burns = (_RAISE_ORBIT, 0, int(ManeuverIntent.PHASE_FORWARD))

    while quiet.agents:
        actions = {agent: (burns[i] if quiet._step_count % 3 == 0 else 0) for i, agent in enumerate(quiet.agents)}
        quiet.step(actions)
        loud.step(actions)
        assert quiet.last_step["approved"] == loud.last_step["approved"]
        assert quiet.last_step["applied_dv_mps"] == loud.last_step["applied_dv_mps"]
        assert quiet.maneuver_mission_deviations() == loud.maneuver_mission_deviations()

    assert quiet.maneuver_mission_deviations()[0] > 0.0
    assert quiet._pending_mission_deviations != loud._pending_mission_deviations


def test_a_satellite_whose_own_burns_exceed_the_limit_is_still_rejected():
    limit = 0.15
    capability = SatelliteCapability(0.1, 2.0, True, limit)
    env = ShieldedOrbitGuardEnv(num_agents=3, capability_template=capability, episode_horizon_s=1800.0)
    env.reset(seed=_VALIDATION_SEED)
    outcomes = []
    while env.agents:
        before = env.maneuver_mission_deviations()[0]
        env.step({**_no_op(env), env.possible_agents[0]: _RAISE_ORBIT})
        outcomes.append((before, "mission_limit" in _violation_kinds(env, 0), env.last_shield_result.approved[0]))

    first_before, first_rejected, first_approved = outcomes[0]
    assert first_before == 0.0 and not first_rejected and first_approved == _RAISE_ORBIT
    rejected = [(before, approved) for before, hit, approved in outcomes if hit]
    assert rejected, "accumulated RAISE_ORBIT burns never reached the mission limit"
    before, approved = rejected[0]
    assert 0.0 < before <= limit  # the satellite's own burns, not noise, used up the allowance
    assert approved != _RAISE_ORBIT
    assert env.maneuver_mission_deviations()[0] > 0.0
