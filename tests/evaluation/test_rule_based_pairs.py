"""Rule-Based on the injected conjunctions of the wave-4 probe seeds (validation pool, N=15, normal).

Both pair members must burn opposite tangential directions on every step where both burn, and
opening the encounter that way must leave the pair further apart than doing nothing. Validation
seeds only; the test pool is never touched.
"""

from __future__ import annotations

import pytest

from orbit_guard.evaluation.baselines import build_env_factory, build_policy

_PROBE_SEEDS = (114021, 110246, 111080)
_NUM_AGENTS = 15
_STEPS_PAST_TCA = 3


def _run_injected_pair(model: str, seed: int) -> tuple[float, int, int]:
    """(realized min separation of injected pair 0, co-burn steps in the same direction, in opposite ones)."""
    env = build_env_factory(model, _NUM_AGENTS)({})
    policy = build_policy(model)
    observations, _ = env.reset(seed=seed)
    conjunction = env._last_scenario.injected_conjunctions[0]
    pair = tuple(sorted((conjunction.primary_index, conjunction.secondary_index)))
    names = tuple(f"sat_{index}" for index in pair)
    last_step = int(conjunction.achieved_tca_s // env._control_dt_s) + _STEPS_PAST_TCA
    same = opposite = 0
    while env.agents and env._step_count <= last_step:
        actions = {agent: int(policy(agent, observations[agent])) for agent in env.agents}
        first, second = actions[names[0]], actions[names[1]]
        if first and second:
            same += first == second
            opposite += first != second
        observations, *_ = env.step(actions)
    return env._metrics._closest[pair][0], same, opposite


@pytest.mark.parametrize("seed", _PROBE_SEEDS)
def test_rule_based_pair_members_burn_opposite_ways_and_open_the_miss(seed):
    rule_min_m, same, opposite = _run_injected_pair("rule_based", seed)
    idle_min_m, _, _ = _run_injected_pair("no_maneuver", seed)

    assert same == 0
    assert opposite > 0
    assert rule_min_m > idle_min_m
