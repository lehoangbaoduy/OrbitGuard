"""Tests for orbit_guard.conjunction.pair_assessment: the shared pair-scoring core."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.conjunction.pair_assessment import (
    assess_all_pairs,
    assess_pair,
    assess_proposal,
    intent_delta_v,
    predict_all,
    predict_proposal_trajectories,
)
from orbit_guard.physics.dynamics import state_derivative_batch
from orbit_guard.physics.integrator import propagate_batch
from orbit_guard.safety.adaptive_margin import (
    compute_d_safe,
    compute_sigma_norm,
    make_adaptive_margin,
)
from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v
from orbit_guard.sensing.belief import AgentBelief
from orbit_guard.sensing.noise import NoiseModel, estimated_covariance

_NOISE = NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1)
_DT_S = 10.0
_LOOKAHEAD_S = 300.0


def _belief(position_m, velocity_mps) -> AgentBelief:
    return AgentBelief(np.array([*position_m, *velocity_mps], dtype=float), estimated_covariance(_NOISE))


def _head_on_pair() -> list[AgentBelief]:
    radius_m = 6_900_000.0
    speed_mps = 7_600.0
    return [
        _belief((radius_m, 0.0, 0.0), (0.0, speed_mps, 0.0)),
        _belief((radius_m, 20_000.0, 0.0), (0.0, -speed_mps, 0.0)),
    ]


def test_adaptive_margin_factory_matches_compute_d_safe():
    margin = make_adaptive_margin(d_base_m=1000.0, d_min_m=200.0, d_max_m=5000.0)

    expected = compute_d_safe(compute_sigma_norm(141.0, 1000.0), 0.4, 0.3)

    assert margin(141.0, 0.4, 0.3) == pytest.approx(expected)


def test_assess_all_pairs_uses_an_injected_margin_function():
    beliefs = _head_on_pair()

    assessments = assess_all_pairs(beliefs, _LOOKAHEAD_S, _DT_S, margin_fn=lambda sigma, rho, risk: 1234.0)

    assert [a.d_safe_m for a in assessments] == [1234.0]


def test_precomputed_predictions_give_the_same_assessment_as_internal_prediction():
    beliefs = _head_on_pair()
    predicted = predict_all(beliefs, _LOOKAHEAD_S, _DT_S)

    reused = assess_all_pairs(beliefs, _LOOKAHEAD_S, _DT_S, predicted=predicted)
    fresh = assess_all_pairs(beliefs, _LOOKAHEAD_S, _DT_S)

    assert reused == fresh


def test_assess_pair_reports_the_predicted_minimum_separation_and_a_bounded_risk():
    beliefs = _head_on_pair()
    predicted = predict_all(beliefs, _LOOKAHEAD_S, _DT_S)

    assessment = assess_pair(
        0, 1, predicted[0].trajectory, predicted[1].trajectory, _DT_S, 141.0, 0.0, make_adaptive_margin()
    )

    assert assessment.d_min_m < 20_000.0
    assert 0.0 <= assessment.risk_proxy <= 1.0
    assert 200.0 <= assessment.d_safe_m <= 5000.0
    assert (assessment.i, assessment.j) == (0, 1)




_NO_OP = int(ManeuverIntent.NO_MANEUVER)
_RAISE = int(ManeuverIntent.RAISE_ORBIT)
_PHASE = int(ManeuverIntent.PHASE_FORWARD)


def test_intent_delta_v_maps_from_the_belief_state_sized_by_the_given_per_action_limit():
    belief = _head_on_pair()[0]
    state = belief.state_vector

    burn = intent_delta_v(belief, _PHASE, 0.05, 0.5)

    expected = map_intent_to_delta_v(_PHASE, state[:3], state[3:6], base_delta_v_mps=0.05, phase_fraction=0.5)
    np.testing.assert_array_equal(burn, expected)
    assert float(np.linalg.norm(burn)) == pytest.approx(0.025)


def test_an_all_no_op_proposal_reuses_the_baseline_and_reproduces_the_decision_time_assessment():
    beliefs = _head_on_pair()
    baseline = predict_all(beliefs, _LOOKAHEAD_S, _DT_S)

    outcome = assess_proposal(
        beliefs, (_NO_OP, _NO_OP), (0.1, 0.1), phase_fraction=0.5, lookahead_s=_LOOKAHEAD_S,
        physics_dt_s=_DT_S, margin_fn=make_adaptive_margin(), baseline_predictions=baseline,
    )

    assert outcome == assess_all_pairs(beliefs, _LOOKAHEAD_S, _DT_S, predicted=baseline)


def test_burning_satellites_are_propagated_from_belief_plus_their_own_burn():
    beliefs = _head_on_pair()
    baseline = predict_all(beliefs, _LOOKAHEAD_S, _DT_S)

    trajectories = predict_proposal_trajectories(
        beliefs, (_RAISE, _NO_OP), (0.05, 0.1), phase_fraction=0.5, lookahead_s=_LOOKAHEAD_S,
        physics_dt_s=_DT_S, baseline_predictions=baseline,
    )

    start = beliefs[0].state_vector.copy()
    start[3:6] += map_intent_to_delta_v(_RAISE, start[:3], start[3:6], base_delta_v_mps=0.05)
    (expected,) = propagate_batch(start[None, :], _DT_S, round(_LOOKAHEAD_S / _DT_S), state_derivative_batch)
    np.testing.assert_array_equal(trajectories[0], expected)
    assert trajectories[1] is baseline[1].trajectory


def test_proposal_prediction_rejects_one_entry_per_satellite_mismatches():
    beliefs = _head_on_pair()
    baseline = predict_all(beliefs, _LOOKAHEAD_S, _DT_S)

    with pytest.raises(ValueError):
        predict_proposal_trajectories(
            beliefs, (_RAISE,), (0.1, 0.1), phase_fraction=0.5, lookahead_s=_LOOKAHEAD_S,
            physics_dt_s=_DT_S, baseline_predictions=baseline,
        )
    with pytest.raises(ValueError):
        predict_proposal_trajectories(
            beliefs, (_RAISE, _NO_OP), (0.1,), phase_fraction=0.5, lookahead_s=_LOOKAHEAD_S,
            physics_dt_s=_DT_S, baseline_predictions=baseline,
        )
    with pytest.raises(ValueError):
        predict_proposal_trajectories(
            beliefs, (_RAISE, _NO_OP), (0.1, 0.1), phase_fraction=0.5, lookahead_s=_LOOKAHEAD_S,
            physics_dt_s=_DT_S, baseline_predictions=baseline[:1],
        )


def test_proposal_prediction_refuses_anything_but_a_belief():
    beliefs = _head_on_pair()
    baseline = predict_all(beliefs, _LOOKAHEAD_S, _DT_S)

    with pytest.raises(TypeError):
        predict_proposal_trajectories(
            [beliefs[0].state_vector, beliefs[1]], (_RAISE, _NO_OP), (0.1, 0.1), phase_fraction=0.5,
            lookahead_s=_LOOKAHEAD_S, physics_dt_s=_DT_S, baseline_predictions=baseline,
        )
