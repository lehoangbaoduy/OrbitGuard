"""One reward estimator for both shield arms (CORE-2a; PRD §25, §28; docs/OrbitGuard_progress.md Q18, Q25).

The shield-vs-no-shield ablation compares two training runs, so both must score R_safety and the
coordination term with the SAME estimator: the predicted outcome of the RAW joint proposal, from the
decision-time beliefs and baseline predictions, with burns mapped from the belief state and sized by each
satellite's own per-action limit. The base environment used to score both terms against a second,
independently-noised post-step belief assessment instead, so pure sensing noise (~0.095/step under
all-no-op actions at N=15) leaked into its coordination term but not into the shielded one.

The mission term, the delta-v term and the collision penalty stay on the realized transition.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from orbit_guard.physics.mission_deviation import compute_mission_deviation
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.reward import compute_mission_term
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_N = 15
_SEED = 3  # multi_conjunction at N=15: two pairs inside the margin (risk ~0.62, ~0.67) at reset
_KW = {"episode_horizon_s": 300.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0}
# A cheap repair search: the proposal's predicted outcome does not depend on the search settings.
_CHEAP_SHIELD = {"shield_exact_max_agents": 1, "shield_beam_width": 1}
_ZERO_WEIGHTS = {"w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0, "collision_penalty": 0.0}


def _weights(**only: float) -> dict[str, float]:
    return {**_ZERO_WEIGHTS, **only}


def _pair(seed: int = _SEED, **overrides) -> tuple[OrbitGuardEnv, ShieldedOrbitGuardEnv]:
    base = OrbitGuardEnv(num_agents=_N, **{**_KW, **overrides})
    shielded = ShieldedOrbitGuardEnv(num_agents=_N, **{**_KW, **_CHEAP_SHIELD, **overrides})
    base.reset(seed=seed)
    shielded.reset(seed=seed)
    return base, shielded


def _no_op(env) -> dict[str, int]:
    return dict.fromkeys(env.agents, int(ManeuverIntent.NO_MANEUVER))


def _random_proposals(agents: list[str], count: int) -> list[dict[str, int]]:
    rng = np.random.default_rng(2026)
    proposals = [{agent: int(rng.integers(0, len(ManeuverIntent))) for agent in agents} for _ in range(count)]
    # One proposal leaves two satellites out (non-controllable, e.g. already collided): they burn nothing.
    proposals.append({agent: action for agent, action in proposals[0].items() if agent not in agents[:2]})
    return proposals


def _set_capability(envs, index: int, **changes) -> None:
    for env in envs:
        env._capabilities[index] = replace(env._capabilities[index], **changes)


def test_base_env_outcome_pairs_equal_the_shields_proposal_evaluation_for_random_joint_proposals():
    base, shielded = _pair()
    # Per-satellite capabilities differ from the template: a smaller per-action burn, a disabled
    # satellite and an exhausted budget. The raw proposal is predicted as requested in every case.
    _set_capability((base, shielded), 3, max_delta_v_per_action_mps=0.05)
    _set_capability((base, shielded), 4, maneuver_enabled=False)
    _set_capability((base, shielded), 5, remaining_delta_v_mps=0.0)

    for proposal in _random_proposals(base.agents, 4):
        _, decision = shielded._approve_actions(proposal)

        outcome = base._outcome_assessments(proposal, None)

        assert outcome == list(decision.proposal_evaluation.pairs)


def test_all_no_op_steps_give_an_exactly_zero_coordination_term_without_the_shield():
    base, _ = _pair(reward_weights=_weights(w4=1.0))

    for _ in range(3):
        _, rewards, _, _, _ = base.step(_no_op(base))
        assert all(reward == 0.0 for reward in rewards.values())


def test_all_no_op_step_scores_the_same_safety_term_with_and_without_the_shield():
    base, shielded = _pair(reward_weights=_weights(w1=1.0))

    _, base_rewards, _, _, _ = base.step(_no_op(base))
    _, shielded_rewards, _, _, _ = shielded.step(_no_op(shielded))

    assert base_rewards == shielded_rewards


def test_all_no_op_step_scores_a_zero_coordination_term_with_the_shield_too():
    _, shielded = _pair(reward_weights=_weights(w4=1.0))

    _, rewards, _, _, _ = shielded.step(_no_op(shielded))

    assert all(reward == 0.0 for reward in rewards.values())


@pytest.mark.parametrize("env_class", [OrbitGuardEnv, ShieldedOrbitGuardEnv], ids=["no_shield", "shielded"])
def test_mission_term_is_scored_on_the_realized_post_step_belief(env_class):
    extra = _CHEAP_SHIELD if env_class is ShieldedOrbitGuardEnv else {}
    env = env_class(num_agents=_N, reward_weights=_weights(w2=1.0), **_KW, **extra)
    env.reset(seed=_SEED)
    proposal = {agent: int(ManeuverIntent.RAISE_ORBIT) for agent in env.agents}

    _, rewards, _, _, _ = env.step(proposal)

    for agent, reward in rewards.items():
        idx = env._agent_index[agent]
        components = compute_mission_deviation(env._pending_beliefs[idx].state_vector, env._shadow_states[idx])
        expected = compute_mission_term(components.d_a, components.d_e, components.d_i, components.d_m, **env._mission_weights)
        assert reward == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("env_class", [OrbitGuardEnv, ShieldedOrbitGuardEnv], ids=["no_shield", "shielded"])
def test_collision_penalty_is_charged_on_the_realized_ground_truth_collision(env_class):
    extra = _CHEAP_SHIELD if env_class is ShieldedOrbitGuardEnv else {}
    env = env_class(
        num_agents=_N, reward_weights=_weights(collision_penalty=-100.0), hard_body_radius_m=1.0e9, **_KW, **extra
    )
    env.reset(seed=_SEED)

    _, rewards, terminations, _, _ = env.step(_no_op(env))

    assert all(terminations.values())
    assert all(reward == -100.0 for reward in rewards.values())
