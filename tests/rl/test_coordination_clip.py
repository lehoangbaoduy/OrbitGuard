"""Coordination-term clipping (CORE-2b; PRD §25 "clipping/normalization must be fixed before training", Q13).

Each pair's risk proxy is in [0, 1], but the coordination term sums the one-sided degradation over every
candidate pair (up to 105 at N=15), so unclipped it is unbounded in N. `reward.coordination_clip` (1.0 by
default) caps it so it stays O(1) like the other bounded terms and one step cannot dominate the gradient.
`compute_coordination_term` itself is frozen (tests/rl/test_reward.py); the clip is a separate function.
"""

from __future__ import annotations

import math

import pytest

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.reward import (
    compute_clipped_coordination_term,
    compute_coordination_term,
)
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_KW = {"episode_horizon_s": 300.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0}
_SEED = 11  # N=3 conjunction: single burns raise the predicted summed risk by up to ~0.044
_W4_ONLY = {"w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 1.0, "collision_penalty": 0.0}
_SMALL_CLIP = 0.01


def test_clipped_coordination_term_caps_the_degradation_at_the_clip():
    assert compute_clipped_coordination_term(0.2, 3.7, clip=1.0) == 1.0


def test_clipped_coordination_term_leaves_a_degradation_below_the_clip_unchanged():
    assert compute_clipped_coordination_term(0.2, 0.5, clip=1.0) == compute_coordination_term(0.2, 0.5)


def test_clipped_coordination_term_scores_an_improvement_zero():
    assert compute_clipped_coordination_term(0.9, 0.1, clip=1.0) == 0.0


@pytest.mark.parametrize("clip", [0.0, -1.0, math.inf, math.nan])
def test_clipped_coordination_term_rejects_a_non_positive_or_non_finite_clip(clip):
    with pytest.raises(ValueError):
        compute_clipped_coordination_term(0.0, 1.0, clip=clip)


def risk_raising_proposal(env: OrbitGuardEnv) -> tuple[dict[str, int], float]:
    """The single-satellite burn that most raises the predicted summed risk from `env`'s current state, and its term."""
    pre = sum(a.risk_proxy for a in env._pending_assessments)
    best: tuple[dict[str, int], float] = ({}, 0.0)
    for agent in env.agents:
        for intent in ManeuverIntent:
            proposal = {**dict.fromkeys(env.agents, 0), agent: int(intent)}
            post = sum(a.risk_proxy for a in env._outcome_assessments(proposal, None))
            term = compute_coordination_term(pre, post)
            if term > best[1]:
                best = (proposal, term)
    return best


@pytest.mark.parametrize("env_class", [OrbitGuardEnv, ShieldedOrbitGuardEnv], ids=["no_shield", "shielded"])
def test_environment_applies_the_coordination_clip_in_both_shield_arms(env_class):
    probe = OrbitGuardEnv(num_agents=3, **_KW)
    probe.reset(seed=_SEED)
    proposal, raw_term = risk_raising_proposal(probe)
    assert raw_term > _SMALL_CLIP  # precondition: the clip binds
    clipped = env_class(num_agents=3, reward_weights=_W4_ONLY, coordination_clip=_SMALL_CLIP, **_KW)
    unclipped = env_class(num_agents=3, reward_weights=_W4_ONLY, **_KW)
    clipped.reset(seed=_SEED)
    unclipped.reset(seed=_SEED)

    _, clipped_rewards, _, _, _ = clipped.step(proposal)
    _, default_rewards, _, _, _ = unclipped.step(proposal)

    assert all(reward == -_SMALL_CLIP for reward in clipped_rewards.values())
    assert all(reward == pytest.approx(-raw_term, rel=1e-12) for reward in default_rewards.values())


def test_bare_environment_resolves_the_coordination_clip_from_the_config():
    assert OrbitGuardEnv(num_agents=2)._coordination_clip == 1.0


def test_environment_rejects_a_non_positive_coordination_clip():
    with pytest.raises(ValueError, match="coordination_clip"):
        OrbitGuardEnv(num_agents=2, coordination_clip=0.0)
