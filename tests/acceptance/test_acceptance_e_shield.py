"""Acceptance Test E (PRD §38): an intentionally unsafe proposal is replaced by the shield.

End to end on the real environment: every agent proposes NO_MANEUVER (the
"intentionally unsafe" joint proposal) in scenarios where that physically ends
in a collision. Without the shield the episode collides; with it the shield
replaces the proposal and the episode survives. Zero sensing noise and a 100 m
hard-body radius make the collision geometry deterministic.

The N=15 cases repeat this on the DEFAULT configuration (noise, hard-body radius, margin, shield; only the
horizon is shortened) and check the ShieldResult itself: the rejection names the injected collision-course
pair, and on every rejection with a feasible alternative the approved action differs from the proposal and its
predicted outcome has no violation involving an active agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import pytest

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.shield import ShieldResult
from orbit_guard.sensing.noise import NoiseModel

_KW = {
    "episode_horizon_s": 2400.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0,
    "noise_model": NoiseModel(0.0, 0.0), "hard_body_radius_m": 100.0,
}
_COLLIDING_SEEDS = (3, 9, 34)


def _collided(env_class, seed: int) -> tuple[bool, float]:
    env = env_class(num_agents=2, **_KW)
    env.reset(seed=seed)
    collided = False
    while env.agents:
        _, _, terminations, _, _ = env.step({agent: 0 for agent in env.agents})
        collided |= any(terminations.values())
    return collided, sum(env.delta_v_used_mps().values())


@pytest.mark.parametrize("seed", _COLLIDING_SEEDS)
def test_unsafe_no_maneuver_proposals_collide_unshielded_but_are_repaired_by_the_shield(seed):
    unshielded_collided, unshielded_dv = _collided(OrbitGuardEnv, seed)
    shielded_collided, shielded_dv = _collided(ShieldedOrbitGuardEnv, seed)

    assert unshielded_collided
    assert unshielded_dv == 0.0
    assert not shielded_collided
    assert shielded_dv > 0.0  # the avoidance burns came from the shield, not the proposal


# --- N=15 on the default configuration -------------------------------------------------------------------------
#
# Default sensing noise (100 m / 0.1 m/s), default 5 m hard-body radius, default margin and shield. Only the
# horizon is shortened, 21600 s -> 1800 s, for runtime. Injected TCAs are drawn as a fraction of the horizon, so
# these seeds hold only at this horizon. Offline scan: 34 of seeds 0-199 carry an injected collision-course
# conjunction (d_min < 10 m); of 23 run through both arms, all collide without the shield under all-NO_MANEUVER
# and none with it. Most rejections at N=15 defaults are infeasible (a 0.1 m/s burn cannot clear a ~1 km margin
# inside the lookahead); only 76, 102, 110, 116, 127 and 145 also reach a rejection with a feasible alternative,
# so the pinned seeds (76, 102) exercise the feasible branch too.

_N15 = 15
_N15_HORIZON_S = 1800.0
_N15_SEEDS = (76, 102)


@dataclass(frozen=True)
class _ShieldedStep:
    active: tuple[bool, ...]
    result: ShieldResult
    refiltered_approved_accepted: bool | None  # the approved action re-judged as a proposal (feasible rejections only)


def _violating_agents(violations, active) -> set[int]:
    return {agent for v in violations if any(active[a] for a in v.agents) for agent in v.agents}


@cache
def _n15_unshielded(seed: int):
    env = OrbitGuardEnv(num_agents=_N15, episode_horizon_s=_N15_HORIZON_S)
    env.reset(seed=seed)
    while env.agents:
        env.step({agent: 0 for agent in env.agents})
    return env.episode_summary(), sum(env.delta_v_used_mps().values())


@cache
def _n15_shielded(seed: int):
    env = ShieldedOrbitGuardEnv(num_agents=_N15, episode_horizon_s=_N15_HORIZON_S)
    env.reset(seed=seed)
    collision_course = tuple(
        tuple(sorted((c.primary_index, c.secondary_index)))
        for c in env._last_scenario.injected_conjunctions
        if c.achieved_d_min_m < 2.0 * env._hard_body_radius_m
    )
    steps = []
    while env.agents:
        actions = {agent: 0 for agent in env.agents}  # the intentionally unsafe joint proposal
        active = tuple(agent in actions for agent in env.possible_agents)
        decision_context = (
            list(env._pending_beliefs), list(env._capabilities),
            list(env._pending_mission_deviations), list(env._pending_predictions),
        )
        env.step(actions)
        result = env.last_shield_result
        refiltered = None
        if not result.accepted and result.feasible:
            beliefs, capabilities, deviations, predictions = decision_context
            refiltered = env._shield.filter(
                beliefs, list(result.approved), capabilities,
                controllable=active, mission_deviations=deviations, baseline_predictions=predictions,
            ).accepted
        steps.append(_ShieldedStep(active, result, refiltered))
    return collision_course, tuple(steps), env.episode_summary()


@pytest.mark.parametrize("seed", _N15_SEEDS)
def test_n15_defaults_the_shield_rejects_the_no_maneuver_proposal_because_of_the_injected_threat(seed):
    collision_course, steps, _ = _n15_shielded(seed)
    first = steps[0].result

    assert collision_course  # the scenario really is on a collision course
    assert not first.accepted
    violating_pairs = {tuple(sorted(v.agents)) for v in first.proposal_evaluation.violations if len(v.agents) == 2}
    assert set(collision_course) <= violating_pairs


@pytest.mark.parametrize("seed", _N15_SEEDS)
def test_n15_defaults_a_feasible_rejection_is_replaced_by_an_action_with_no_active_violation(seed):
    _, steps, _ = _n15_shielded(seed)
    feasible_rejections = [step for step in steps if not step.result.accepted and step.result.feasible]

    assert feasible_rejections  # the feasible branch is exercised, not passed vacuously
    for step in feasible_rejections:
        result = step.result
        blamed = _violating_agents(result.proposal_evaluation.violations, step.active)
        assert blamed
        assert result.approved != result.proposed
        assert any(result.approved[agent] != result.proposed[agent] for agent in blamed)
        # Recomputed from the evaluation itself with the mask built from env.agents, not read off `feasible`.
        assert not _violating_agents(result.approved_evaluation.violations, step.active)
        # Independently re-judged from the same decision-time beliefs, the approved action is acceptable.
        assert step.refiltered_approved_accepted is True


@pytest.mark.parametrize("seed", _N15_SEEDS)
def test_n15_defaults_unshielded_no_maneuver_collides_but_the_shielded_episode_does_not(seed):
    unshielded, unshielded_dv = _n15_unshielded(seed)
    _, steps, shielded = _n15_shielded(seed)

    assert unshielded.collision_count >= 1
    assert unshielded_dv == 0.0
    assert shielded.collision_count == 0
    assert shielded.total_delta_v_mps > 0.0  # every burn came from the shield; the proposal never moved
    assert all(step.result.proposed == (0,) * _N15 for step in steps)
