"""Contract tests for orbit_guard.rl.reward (PRD §25).

    R_i = w1*R_safety,i + w2*R_mission,i - w3*||dv_i|| - w4*C_coordination,i  [+ collision_penalty if collided]

Two of the four terms are themselves already reward-signed (higher = better)
and are ADDED in the top formula: `R_safety` (clipped worst-case pairwise
margin) and `R_mission` (negative of the weighted orbital-element deviation
D_mission -- this sign, R_mission = -D_mission, is an explicit freeze here;
the PRD states D_mission's formula but never states this sign itself). The
other two are raw, non-negative magnitudes and are SUBTRACTED in the top
formula exactly as PRD §25 literally writes it: `compute_delta_v_term`
returns ||dv|| itself (not pre-negated), and `compute_coordination_term`
returns the one-sided positive risk DEGRADATION (post - pre, floored at 0 --
PRD §25: "Only positive degradation ... is penalized; an improvement is not
double-rewarded here"). `compute_reward` is the only place that assembles
final signs -- do not pre-negate delta_v/coordination in their own
functions, and do not forget to negate mission deviation in its own
function. This asymmetry mirrors the PRD's literal formula 1:1 and is the
least error-prone convention for whoever wires this into the environment.

`compute_coordination_term`'s pre/post are the GLOBAL, whole-constellation
total risk (R_total in PRD §25) -- NOT the same quantity as
`rl.observation`'s per-agent local `total_risk` (sum over one agent's own
neighbor candidates). Do not conflate the two; the environment computes
this global quantity separately by summing risk_proxy over ALL active
conjunction pairs constellation-wide, once per control step, pre- and
post-joint-maneuver.

`compute_coordination_term`'s output is DELIBERATELY left unnormalized here
-- PRD §25 itself says "clipping/normalization must be fixed before
training" without specifying a formula, and with N=15 there are up to 105
pairs, so an unscaled degradation could swamp the other three terms once
real numbers are plugged in. This is an open Q-item (docs/OrbitGuard_progress.md)
owned by Part C's own reward-weight grid search (PRD §39/§48), not something
this frozen spec resolves -- do not silently invent a normalization inside
`compute_coordination_term` itself; whatever scaling is chosen belongs in
`compute_reward`'s weight `w4` or a documented follow-up parameter, so it
stays visible and tunable.

`compute_mission_term`'s four inputs (d_a, d_e, d_i, d_m) and
`rl.observation`'s `own_mission_deviation` scalar must be derived from the
IDENTICAL four normalized deviation components and the SAME q-weights by
Part A's environment, such that `own_mission_deviation ==
-compute_mission_term(d_a, d_e, d_i, d_m)` always holds for the same
satellite at the same instant -- see tests/rl/test_observation.py's module
docstring for the other half of this contract.

Every function here is a pure, framework-free function of plain floats/
arrays -- no orbit_guard.rl.environment, gymnasium, pettingzoo, ray, or
torch import, so this module is testable and reviewable in complete
isolation from the RLlib training stack (that stack lives in
rl/training.py, a separate Part C module).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from orbit_guard.rl.reward import (
    compute_coordination_term,
    compute_delta_v_term,
    compute_mission_term,
    compute_pairwise_margin,
    compute_reward,
    compute_safety_term,
)

_EPSILON_M = 1.0
_COLLISION_PENALTY = -100.0
_W1, _W2, _W3, _W4 = 1.0, 0.3, 0.1, 0.5


# --- compute_pairwise_margin --------------------------------------------------

def test_pairwise_margin_is_zero_when_exactly_at_d_safe():
    assert compute_pairwise_margin(d_min_m=1000.0, d_safe_m=1000.0, epsilon_m=_EPSILON_M) == pytest.approx(0.0)


def test_pairwise_margin_is_positive_when_comfortably_separated():
    margin = compute_pairwise_margin(d_min_m=2000.0, d_safe_m=1000.0, epsilon_m=_EPSILON_M)
    assert margin > 0.0
    assert margin == pytest.approx((2000.0 - 1000.0) / (1000.0 + _EPSILON_M))


def test_pairwise_margin_is_negative_when_inside_the_safety_bubble():
    margin = compute_pairwise_margin(d_min_m=500.0, d_safe_m=1000.0, epsilon_m=_EPSILON_M)
    assert margin < 0.0


# --- compute_safety_term: worst-case (min) pairwise margin, clipped ---------

def test_safety_term_is_the_minimum_margin_when_within_clip_bounds():
    margins = [0.5, -0.2, 0.1]
    assert compute_safety_term(margins, clip_min=-1.0, clip_max=1.0) == pytest.approx(-0.2)


def test_safety_term_clips_at_the_lower_bound():
    margins = [-5.0, 0.5]
    assert compute_safety_term(margins, clip_min=-1.0, clip_max=1.0) == pytest.approx(-1.0)


def test_safety_term_clips_at_the_upper_bound():
    margins = [5.0, 6.0]
    assert compute_safety_term(margins, clip_min=-1.0, clip_max=1.0) == pytest.approx(1.0)


def test_safety_term_with_no_active_pairs_is_maximally_safe():
    # No neighbor candidates at all -> nothing threatening -> clip_max, not 0
    # and not an error (a satellite can legitimately have zero conjunctions
    # in a control step).
    assert compute_safety_term([], clip_min=-1.0, clip_max=1.0) == pytest.approx(1.0)


def test_safety_term_a_single_bad_pair_dominates_many_good_pairs():
    # This is the whole point of "min" over "mean": one dangerous encounter
    # must not be washed out by nine benign ones.
    margins = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, -0.9]
    assert compute_safety_term(margins, clip_min=-1.0, clip_max=1.0) == pytest.approx(-0.9)


# --- compute_mission_term: R_mission = -D_mission ---------------------------

def test_mission_term_is_zero_when_all_deviations_are_zero():
    assert compute_mission_term(d_a=0.0, d_e=0.0, d_i=0.0, d_m=0.0) == pytest.approx(0.0)


def test_mission_term_is_the_negative_weighted_sum_of_deviations():
    term = compute_mission_term(d_a=0.4, d_e=0.2, d_i=0.1, d_m=0.3, q_a=0.25, q_e=0.25, q_i=0.25, q_m=0.25)
    expected_d_mission = 0.25 * 0.4 + 0.25 * 0.2 + 0.25 * 0.1 + 0.25 * 0.3
    assert term == pytest.approx(-expected_d_mission)


def test_mission_term_uses_prd_default_equal_weights_when_unspecified():
    term = compute_mission_term(d_a=1.0, d_e=0.0, d_i=0.0, d_m=0.0)
    assert term == pytest.approx(-0.25)


def test_mission_term_rejects_negative_deviation_components():
    # D_a/D_e/D_i/D_m are normalized deviation MAGNITUDES -- always >= 0.
    with pytest.raises(ValueError):
        compute_mission_term(d_a=-0.1, d_e=0.0, d_i=0.0, d_m=0.0)


# --- compute_delta_v_term: raw magnitude, NOT pre-negated -------------------

def test_delta_v_term_is_the_euclidean_norm():
    assert compute_delta_v_term([0.03, 0.04, 0.0]) == pytest.approx(0.05)


def test_delta_v_term_of_zero_vector_is_zero():
    assert compute_delta_v_term([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_delta_v_term_is_never_negative():
    assert compute_delta_v_term([-0.03, -0.04, 0.0]) >= 0.0


# --- compute_coordination_term: one-sided positive degradation only --------

def test_coordination_term_is_zero_when_risk_improves():
    assert compute_coordination_term(total_risk_pre=5.0, total_risk_post=3.0) == pytest.approx(0.0)


def test_coordination_term_is_zero_when_risk_is_unchanged():
    assert compute_coordination_term(total_risk_pre=5.0, total_risk_post=5.0) == pytest.approx(0.0)


def test_coordination_term_is_positive_degradation_when_risk_worsens():
    assert compute_coordination_term(total_risk_pre=3.0, total_risk_post=5.0) == pytest.approx(2.0)


# --- compute_reward: final assembly -----------------------------------------

def test_reward_assembles_all_four_terms_with_prd_default_weights_and_signs():
    safety_term = 0.5
    mission_term = -0.2
    delta_v_term = 0.05
    coordination_term = 1.0
    reward = compute_reward(
        safety_term=safety_term,
        mission_term=mission_term,
        delta_v_term=delta_v_term,
        coordination_term=coordination_term,
        collision_occurred=False,
        w1=_W1, w2=_W2, w3=_W3, w4=_W4,
        collision_penalty=_COLLISION_PENALTY,
    )
    expected = (
        _W1 * safety_term + _W2 * mission_term - _W3 * delta_v_term - _W4 * coordination_term
    )
    assert reward == pytest.approx(expected)


def test_reward_applies_collision_penalty_exactly_once_on_top_of_dense_terms():
    dense_only = compute_reward(
        safety_term=0.5, mission_term=-0.2, delta_v_term=0.05, coordination_term=1.0,
        collision_occurred=False, w1=_W1, w2=_W2, w3=_W3, w4=_W4, collision_penalty=_COLLISION_PENALTY,
    )
    with_collision = compute_reward(
        safety_term=0.5, mission_term=-0.2, delta_v_term=0.05, coordination_term=1.0,
        collision_occurred=True, w1=_W1, w2=_W2, w3=_W3, w4=_W4, collision_penalty=_COLLISION_PENALTY,
    )
    assert with_collision == pytest.approx(dense_only + _COLLISION_PENALTY)


def test_reward_uses_prd_default_weights_when_unspecified():
    reward = compute_reward(safety_term=1.0, mission_term=0.0, delta_v_term=0.0, coordination_term=0.0, collision_occurred=False)
    assert reward == pytest.approx(_W1 * 1.0)


def test_reward_is_deterministic():
    kwargs = {
        "safety_term": 0.3,
        "mission_term": -0.1,
        "delta_v_term": 0.02,
        "coordination_term": 0.5,
        "collision_occurred": False,
    }
    assert compute_reward(**kwargs) == compute_reward(**kwargs)


# --- Architectural boundary: reward.py stays framework-free -----------------

def test_reward_module_never_imports_rl_environment_or_training_frameworks():
    forbidden_modules = {
        "orbit_guard.rl.environment",
        "orbit_guard.rl.training",
        "gymnasium",
        "pettingzoo",
        "ray",
        "torch",
    }
    reward_file = Path(__file__).resolve().parents[2] / "orbit_guard" / "rl" / "reward.py"
    tree = ast.parse(reward_file.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = {node.module}
        else:
            continue
        for forbidden in forbidden_modules:
            hit = {name for name in names if name == forbidden or name.startswith(forbidden + ".")}
            assert not hit, f"reward.py imports {hit}, violating the framework-free pure-function design"
