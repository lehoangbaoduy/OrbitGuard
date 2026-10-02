"""Tests for orbit_guard.conjunction.pair_assessment: the shared pair-scoring core."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.conjunction.pair_assessment import (
    assess_all_pairs,
    assess_pair,
    predict_all,
)
from orbit_guard.safety.adaptive_margin import (
    compute_d_safe,
    compute_sigma_norm,
    make_adaptive_margin,
)
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


