"""Acceptance Test E (PRD §38): an intentionally unsafe proposal is replaced by the shield.

End to end on the real environment: every agent proposes NO_MANEUVER (the
"intentionally unsafe" joint proposal) in scenarios where that physically ends
in a collision. Without the shield the episode collides; with it the shield
replaces the proposal and the episode survives. Zero sensing noise and a 100 m
hard-body radius make the collision geometry deterministic.
"""

from __future__ import annotations

import pytest

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
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
