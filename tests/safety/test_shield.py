"""Tests for orbit_guard.safety.shield (PRD §29, P4-M2/M4; Acceptance Test E).

Real physics throughout: the shield's whole job is predicting the joint
consequence of candidate burns, so mocking propagation would test nothing.
Beliefs are zero-noise snapshots of Phase 2 scenarios; `sigma_rel` still comes
from the (reported) covariance, as in the live pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v
from orbit_guard.safety.shield import JointActionShield, ViolationKind
from orbit_guard.scenarios.constellation import ShellDistribution
from orbit_guard.scenarios.small_n_scenario import generate_small_n_conjunction_scenario
from orbit_guard.sensing.belief import (
    AgentBelief,
    GroundTruthState,
    form_belief,
    observe,
)
from orbit_guard.sensing.noise import NoiseModel

_DT_S = 10.0
_LOOKAHEAD_S = 1800.0
_NOISE = NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1)
_ZERO_NOISE = NoiseModel(position_sigma_m=0.0, velocity_sigma_mps=0.0)
_SHELL = ShellDistribution(
    altitude_km_mean=463.0, altitude_km_std=1.5, altitude_km_min=460.0, altitude_km_max=465.0,
    inclination_deg_mean=53.06, inclination_deg_std=0.03, inclination_deg_min=53.04, inclination_deg_max=53.17,
    eccentricity_mean=0.0002, eccentricity_std=0.00005, eccentricity_min=0.0001, eccentricity_max=0.0004,
)
_CAPABILITY = SatelliteCapability(
    max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=2.0, maneuver_enabled=True, mission_deviation_limit=1.0
)
_SMALL_MARGIN_M = 50.0


def _fixed_margin(d_safe_m: float):
    return lambda sigma_rel_m, rho, risk: d_safe_m


def _scenario(episode_type: str, n: int = 2):
    for seed in range(100):
        scenario = generate_small_n_conjunction_scenario(
            n, seed=seed, shell=_SHELL, capability_template=_CAPABILITY,
            episode_horizon_s=3600.0, physics_dt_s=_DT_S,
        )
        if scenario.episode_type == episode_type:
            return scenario
    pytest.fail(f"no {episode_type} scenario in 100 seeds")


def _beliefs(scenario, noise: NoiseModel = _NOISE) -> list[AgentBelief]:
    rng = np.random.default_rng(0)
    return [
        form_belief(observe(GroundTruthState(sat.state.as_vector()), rng, _ZERO_NOISE), noise)
        for sat in scenario.satellites
    ]


def _capabilities(n: int, template: SatelliteCapability = _CAPABILITY) -> list[SatelliteCapability]:
    return [template] * n


def _shield(**overrides) -> JointActionShield:
    kwargs = {"lookahead_s": _LOOKAHEAD_S, "physics_dt_s": _DT_S}
    kwargs.update(overrides)
    return JointActionShield(**kwargs)


@pytest.fixture(scope="module")
def conjunction_beliefs():
    return _beliefs(_scenario("single_conjunction"))


@pytest.fixture(scope="module")
def benign_beliefs():
    return _beliefs(_scenario("benign"))


def test_a_safe_joint_proposal_is_accepted_unchanged(benign_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))

    result = shield.filter(benign_beliefs, (0, 0), _capabilities(2))

    assert result.accepted
    assert result.approved == result.proposed == (0, 0)
    assert result.intervened == (False, False)
    assert result.feasible
    assert result.repair is None


def test_unsafe_no_maneuver_proposal_on_a_conjunction_is_repaired_to_a_feasible_burn(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))

    result = shield.filter(conjunction_beliefs, (0, 0), _capabilities(2))

    assert not result.accepted
    assert any(v.kind is ViolationKind.D_SAFE for v in result.proposal_evaluation.violations)
    assert result.approved != (0, 0)
    assert any(result.intervened)
    assert result.feasible
    assert result.approved_evaluation.total_risk < result.proposal_evaluation.total_risk


def test_acceptance_e_an_intentionally_unsafe_maneuver_is_replaced_when_a_feasible_alternative_exists(
    conjunction_beliefs,
):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))
    # Brute-force the most dangerous single-satellite burn: the "intentionally unsafe" proposal.
    candidates = [(intent, 0) for intent in range(6)]
    worst = max(
        candidates,
        key=lambda proposal: shield.filter(conjunction_beliefs, proposal, _capabilities(2)).proposal_evaluation.total_risk,
    )

    result = shield.filter(conjunction_beliefs, worst, _capabilities(2))

    assert result.approved != result.proposed
    assert result.feasible
    assert result.approved_evaluation.total_risk < result.proposal_evaluation.total_risk
    assert not result.approved_evaluation.violations


def test_a_maneuver_that_creates_a_new_conjunction_is_flagged_and_replaced():
    # Seed 4: the baseline pair passes at 5645 m. Margin = 99.5% of that, so only a burn that
    # pulls the pair closer (LOWER_ORBIT here, checked below) crosses it.
    scenario = generate_small_n_conjunction_scenario(
        2, seed=4, shell=_SHELL, capability_template=_CAPABILITY, episode_horizon_s=3600.0, physics_dt_s=_DT_S
    )
    beliefs = _beliefs(scenario)
    baseline_d_min_m = _shield(margin_fn=_fixed_margin(1.0)).filter(
        beliefs, (0, 0), _capabilities(2)
    ).proposal_evaluation.pairs[0].d_min_m
    shield = _shield(margin_fn=_fixed_margin(0.995 * baseline_d_min_m))
    assert shield.filter(beliefs, (0, 0), _capabilities(2)).accepted

    result = shield.filter(beliefs, (int(ManeuverIntent.LOWER_ORBIT), 0), _capabilities(2))

    assert ViolationKind.NEW_CONJUNCTION in {v.kind for v in result.proposal_evaluation.violations}
    assert not result.accepted
    assert result.approved == (0, 0)
    assert result.intervened == (True, False)
    assert result.feasible


def test_proposal_evaluation_predicts_the_joint_effect_of_both_burns(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))
    proposal = (int(ManeuverIntent.RAISE_ORBIT), int(ManeuverIntent.LOWER_ORBIT))

    result = shield.filter(conjunction_beliefs, proposal, _capabilities(2))

    # Independent recomputation: apply BOTH burns to the belief states and propagate.
    trajectories = []
    for belief, intent in zip(conjunction_beliefs, proposal, strict=True):
        state = belief.state_vector.copy()
        state[3:6] += map_intent_to_delta_v(intent, state[:3], state[3:6], base_delta_v_mps=0.1)
        trajectories.append(propagate(state, _DT_S, round(_LOOKAHEAD_S / _DT_S), state_derivative))
    expected_d_min_m = float(np.min(np.linalg.norm(trajectories[0][:, :3] - trajectories[1][:, :3], axis=1)))
    (pair,) = result.proposal_evaluation.pairs
    assert pair.d_min_m == pytest.approx(expected_d_min_m, rel=0.02, abs=5.0)


def test_each_satellites_burn_is_sized_by_its_own_per_action_limit(conjunction_beliefs):
    small = SatelliteCapability(0.05, 2.0, True, 1.0)  # this satellite's burns are half the template's
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))
    proposal = (int(ManeuverIntent.RAISE_ORBIT), 0)

    result = shield.filter(conjunction_beliefs, proposal, [small, _CAPABILITY])

    assert not any(v.kind is ViolationKind.CAPABILITY for v in result.proposal_evaluation.violations)
    state = conjunction_beliefs[0].state_vector.copy()
    state[3:6] += map_intent_to_delta_v(proposal[0], state[:3], state[3:6], base_delta_v_mps=0.05)
    burned = propagate(state, _DT_S, round(_LOOKAHEAD_S / _DT_S), state_derivative)
    coasting = propagate(conjunction_beliefs[1].state_vector, _DT_S, round(_LOOKAHEAD_S / _DT_S), state_derivative)
    expected_d_min_m = float(np.min(np.linalg.norm(burned[:, :3] - coasting[:, :3], axis=1)))
    (pair,) = result.proposal_evaluation.pairs
    assert pair.d_min_m == pytest.approx(expected_d_min_m, rel=0.02, abs=5.0)


def test_a_phase_burn_above_the_per_action_limit_is_rejected(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M), phase_fraction=1.5)  # 0.15 m/s > the 0.1 limit

    result = shield.filter(conjunction_beliefs, (int(ManeuverIntent.PHASE_FORWARD), 0), _capabilities(2))

    assert any(v.kind is ViolationKind.CAPABILITY and v.agents == (0,) for v in result.proposal_evaluation.violations)
    assert result.approved[0] != int(ManeuverIntent.PHASE_FORWARD)
    assert result.intervened[0]


def test_insufficient_remaining_budget_is_rejected_not_silently_capped(benign_beliefs):
    capability = SatelliteCapability(0.1, 0.02, True, 1.0)
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))

    result = shield.filter(benign_beliefs, (int(ManeuverIntent.RAISE_ORBIT), 0), _capabilities(2, capability))

    assert result.approved == (0, 0)
    assert result.intervened == (True, False)
    assert any(v.kind is ViolationKind.CAPABILITY for v in result.proposal_evaluation.violations)


def test_maneuver_disabled_satellite_cannot_burn(benign_beliefs):
    capability = SatelliteCapability(0.1, 2.0, False, 1.0)
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))

    result = shield.filter(benign_beliefs, (0, int(ManeuverIntent.LOWER_ORBIT)), _capabilities(2, capability))

    assert result.approved == (0, 0)
    assert result.intervened == (False, True)


def test_mission_deviation_limit_violation_is_rejected(benign_beliefs):
    capability = SatelliteCapability(0.1, 2.0, True, 1e-9)
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))

    result = shield.filter(benign_beliefs, (int(ManeuverIntent.RAISE_ORBIT), 0), _capabilities(2, capability))

    assert any(v.kind is ViolationKind.MISSION_LIMIT for v in result.proposal_evaluation.violations)
    assert result.approved == (0, 0)


def test_current_mission_deviation_counts_against_the_limit(benign_beliefs):
    capability = SatelliteCapability(0.1, 2.0, True, 0.5)
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))
    proposal = (int(ManeuverIntent.PHASE_FORWARD), 0)

    within = shield.filter(benign_beliefs, proposal, _capabilities(2, capability), mission_deviations=(0.0, 0.0))
    exhausted = shield.filter(benign_beliefs, proposal, _capabilities(2, capability), mission_deviations=(0.5, 0.0))

    assert within.accepted
    assert not exhausted.accepted
    assert exhausted.approved == (0, 0)


def test_shield_is_deterministic(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))
    proposal = (int(ManeuverIntent.RAISE_ORBIT), 0)

    first = shield.filter(conjunction_beliefs, proposal, _capabilities(2))
    second = shield.filter(conjunction_beliefs, proposal, _capabilities(2))

    assert first == second


def test_no_feasible_alternative_returns_the_least_risky_flagged_infeasible(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(1.0e7))  # an unattainable margin

    result = shield.filter(conjunction_beliefs, (0, 0), _capabilities(2))

    assert not result.feasible
    assert result.approved_evaluation.total_risk <= result.proposal_evaluation.total_risk
    assert result.repair is not None


def test_uncontrollable_agents_are_never_given_a_burn(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))

    result = shield.filter(
        conjunction_beliefs, (0, int(ManeuverIntent.RAISE_ORBIT)), _capabilities(2), controllable=(True, False)
    )

    assert result.approved[1] == 0
    assert result.intervened[1] is False


def test_uninvolved_agents_keep_their_proposed_burns(conjunction_beliefs):
    # A third satellite on the same orbit rotated 90 degrees about z: far from the conjunction pair.
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    template = conjunction_beliefs[0]
    far = AgentBelief(
        np.concatenate([rotation @ template.state_vector[:3], rotation @ template.state_vector[3:6]]),
        template.covariance,
    )
    beliefs = [*conjunction_beliefs, far]
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))
    proposal = (0, 0, int(ManeuverIntent.PHASE_FORWARD))

    result = shield.filter(beliefs, proposal, _capabilities(3))

    assert result.approved[2] == int(ManeuverIntent.PHASE_FORWARD)
    assert result.intervened[2] is False
    assert 2 not in result.repair.stats.affected_agents


def test_beam_search_is_used_above_the_exact_enumeration_threshold(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M), exact_max_agents=1)

    result = shield.filter(conjunction_beliefs, (0, 0), _capabilities(2))

    assert result.repair.stats.mode == "beam"


def test_exact_enumeration_is_used_at_or_below_the_threshold(conjunction_beliefs):
    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M), exact_max_agents=4)

    result = shield.filter(conjunction_beliefs, (0, 0), _capabilities(2))

    assert result.repair.stats.mode == "exact"
    assert result.repair.stats.evaluated == 6**2


def test_precomputed_baseline_predictions_change_nothing(conjunction_beliefs):
    from orbit_guard.conjunction.pair_assessment import predict_all

    shield = _shield(margin_fn=_fixed_margin(_SMALL_MARGIN_M))
    predicted = predict_all(conjunction_beliefs, _LOOKAHEAD_S, _DT_S)

    reused = shield.filter(conjunction_beliefs, (0, 0), _capabilities(2), baseline_predictions=predicted)
    fresh = shield.filter(conjunction_beliefs, (0, 0), _capabilities(2))

    assert reused == fresh


def test_shield_rejects_non_belief_inputs():
    shield = _shield()
    raw_states = [np.zeros(6), np.zeros(6)]

    with pytest.raises(TypeError):
        shield.filter(raw_states, (0, 0), _capabilities(2))


def test_shield_rejects_an_invalid_intent_or_mismatched_lengths(benign_beliefs):
    shield = _shield()

    with pytest.raises(ValueError):
        shield.filter(benign_beliefs, (0, 9), _capabilities(2))
    with pytest.raises(ValueError):
        shield.filter(benign_beliefs, (0,), _capabilities(2))
    with pytest.raises(ValueError):
        shield.filter(benign_beliefs, (0, 0), _capabilities(3))


def test_shield_module_never_imports_ground_truth_machinery():
    import ast
    import inspect

    import orbit_guard.safety.shield as shield_module

    imported = {
        alias.name
        for node in ast.walk(ast.parse(inspect.getsource(shield_module)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "GroundTruthState" not in imported
    assert "observe" not in imported


def test_with_saturated_risk_and_no_feasible_option_the_shield_still_adds_separation():
    # Zero sensing noise saturates the risk proxy near 0, so risk alone cannot rank burns; the
    # fallback must still prefer a burn that reduces the margin shortfall over doing nothing.
    zero_noise = NoiseModel(position_sigma_m=0.0, velocity_sigma_mps=0.0)
    beliefs = _beliefs(_scenario("single_conjunction"), noise=zero_noise)
    shield = _shield(margin_fn=_fixed_margin(1000.0))

    result = shield.filter(beliefs, (0, 0), _capabilities(2))

    assert not result.feasible
    assert result.approved != (0, 0)
    assert result.approved_evaluation.shortfall_m < result.proposal_evaluation.shortfall_m
