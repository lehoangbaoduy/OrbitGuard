"""Tests for orbit_guard.rl.shielded_environment (PRD §28 training-time shield, P4-M6)."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pettingzoo.test import parallel_api_test

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.reward import compute_pairwise_margin, compute_safety_term
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.shield import JointActionShield

# Long enough lookahead to see the injected conjunction; short episode keeps tests fast.
_KW = {"episode_horizon_s": 180.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0}
_CONJUNCTION_SEED = 10  # N=2: predicted d_min ~128 m, far inside any adaptive margin
_ONLY_WEIGHT = {"w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0, "collision_penalty": 0.0}


def _shielded(**overrides) -> ShieldedOrbitGuardEnv:
    return ShieldedOrbitGuardEnv(num_agents=2, **{**_KW, **overrides})


def _no_op(env):
    return {agent: 0 for agent in env.agents}


def test_shield_replaces_a_no_op_proposal_on_a_conjunction_and_applies_the_approved_burn():
    env = _shielded()
    env.reset(seed=_CONJUNCTION_SEED)
    budgets_before = [cap.remaining_delta_v_mps for cap in env._capabilities]

    _, _, _, _, infos = env.step(_no_op(env))

    assert any(infos[agent]["shield"]["intervened"] for agent in infos)
    assert any(
        cap.remaining_delta_v_mps < before for cap, before in zip(env._capabilities, budgets_before, strict=True)
    )
    assert env.last_shield_result is not None
    assert not env.last_shield_result.accepted


def test_delta_v_cost_is_charged_on_the_raw_request_not_the_shields_burn():
    env = _shielded(reward_weights={**_ONLY_WEIGHT, "w3": 1.0}, hard_body_radius_m=0.0)
    env.reset(seed=_CONJUNCTION_SEED)

    _, rewards, _, _, infos = env.step(_no_op(env))

    assert any(infos[agent]["shield"]["intervened"] for agent in infos)
    assert all(reward == pytest.approx(0.0) for reward in rewards.values())


def test_safety_term_is_scored_against_the_raw_proposals_predicted_outcome():
    env = _shielded(reward_weights={**_ONLY_WEIGHT, "w1": 1.0}, hard_body_radius_m=0.0)
    env.reset(seed=_CONJUNCTION_SEED)

    _, rewards, _, _, _ = env.step(_no_op(env))

    proposal_pairs = env.last_shield_result.proposal_evaluation.pairs
    for agent, reward in rewards.items():
        idx = env._agent_index[agent]
        margins = [compute_pairwise_margin(a.d_min_m, a.d_safe_m) for a in proposal_pairs if idx in (a.i, a.j)]
        assert reward == pytest.approx(compute_safety_term(margins))


def test_base_environment_never_calls_the_shield(monkeypatch):
    def forbidden(self, *args, **kwargs):
        raise AssertionError("the no-shield environment must not call the shield")

    monkeypatch.setattr(JointActionShield, "filter", forbidden)
    env = OrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)

    env.step(_no_op(env))  # must not raise

    shielded = _shielded()
    shielded.reset(seed=_CONJUNCTION_SEED)
    with pytest.raises(AssertionError):
        shielded.step(_no_op(shielded))


def test_importing_the_base_environment_does_not_import_the_shield_module():
    code = (
        "import sys, orbit_guard.rl.environment, orbit_guard.rl.training;"
        "assert 'orbit_guard.safety.shield' not in sys.modules, 'shield leaked into the no-shield path'"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_shielded_environment_is_deterministic_under_a_seed():
    def rollout():
        env = _shielded()
        env.reset(seed=_CONJUNCTION_SEED)
        trace = []
        while env.agents:
            _, rewards, _, _, _ = env.step(_no_op(env))
            trace.append((tuple(sorted(rewards.items())), env.last_shield_result.approved))
        return trace

    assert rollout() == rollout()


def test_shielded_environment_satisfies_the_parallel_api():
    parallel_api_test(_shielded(episode_horizon_s=120.0, lookahead_s=60.0), num_cycles=10)
