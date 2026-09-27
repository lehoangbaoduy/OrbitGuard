"""Contract tests for orbit_guard.sensing.belief (PRD §14).

This is the project's highest-risk correctness boundary: ground truth must
never reach the policy, and prediction must be built from the *estimated*
state, never ground truth. Every belief/prediction-producing function must
reject a GroundTruthState argument outright (TypeError), not just document
the rule in prose -- a leakage bug here silently invalidates every later
result, so the guard has to be structural, not a convention.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.sensing.belief import (
    AgentBelief,
    EvaluationTruth,
    GroundTruthState,
    PredictedState,
    SensorObservation,
    combined_relative_sigma,
    form_belief,
    observe,
    predict_belief,
    to_evaluation_truth,
)
from orbit_guard.sensing.noise import NoiseModel

_ZERO_NOISE_MODEL = NoiseModel(position_sigma_m=0.0, velocity_sigma_mps=0.0)
_REAL_NOISE_MODEL = NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1)

_CIRCULAR_LEO_VECTOR = np.array([6_838_137.0, 0.0, 0.0, 0.0, 0.0, 7_612.0])


def _ground_truth() -> GroundTruthState:
    return GroundTruthState(state_vector=_CIRCULAR_LEO_VECTOR.copy())


# --- Stage 1->2: ground truth -> observation -------------------------------

def test_observe_applies_configured_noise_deterministically():
    truth = _ground_truth()
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)

    obs1 = observe(truth, rng1, _REAL_NOISE_MODEL)
    obs2 = observe(truth, rng2, _REAL_NOISE_MODEL)

    assert isinstance(obs1, SensorObservation)
    assert np.array_equal(obs1.state_vector, obs2.state_vector)
    assert not np.array_equal(obs1.state_vector, truth.state_vector)


def test_observe_with_zero_noise_reproduces_ground_truth_exactly():
    truth = _ground_truth()
    rng = np.random.default_rng(0)

    obs = observe(truth, rng, _ZERO_NOISE_MODEL)

    assert np.array_equal(obs.state_vector, truth.state_vector)


def test_observe_rejects_non_ground_truth_input():
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        observe(_CIRCULAR_LEO_VECTOR, rng, _REAL_NOISE_MODEL)  # raw ndarray, not GroundTruthState


# --- Stage 2->3: observation -> belief --------------------------------------

def test_form_belief_state_matches_observation_and_reports_covariance():
    truth = _ground_truth()
    rng = np.random.default_rng(3)
    obs = observe(truth, rng, _REAL_NOISE_MODEL)

    belief = form_belief(obs, _REAL_NOISE_MODEL)

    assert isinstance(belief, AgentBelief)
    assert np.array_equal(belief.state_vector, obs.state_vector)
    assert belief.covariance.shape == (6, 6)
    assert belief.covariance[0, 0] == pytest.approx(100.0**2)


def test_form_belief_rejects_ground_truth_state():
    # THE leakage guard: you cannot form a belief directly from ground truth,
    # even by mistake -- it must go through observe() first.
    truth = _ground_truth()
    with pytest.raises(TypeError):
        form_belief(truth, _REAL_NOISE_MODEL)


def test_form_belief_rejects_raw_ndarray():
    with pytest.raises(TypeError):
        form_belief(_CIRCULAR_LEO_VECTOR, _REAL_NOISE_MODEL)


# --- Stage 3->4: belief -> prediction ----------------------------------------

def test_predict_belief_propagates_from_belief_state():
    truth = _ground_truth()
    belief = form_belief(observe(truth, np.random.default_rng(1), _ZERO_NOISE_MODEL), _ZERO_NOISE_MODEL)

    prediction = predict_belief(belief, dt=10.0, n_steps=5, derivative_fn=state_derivative)

    assert isinstance(prediction, PredictedState)
    assert prediction.trajectory.shape == (6, 6)
    assert np.array_equal(prediction.trajectory[0], belief.state_vector)


def test_predict_belief_rejects_ground_truth_state():
    truth = _ground_truth()
    with pytest.raises(TypeError):
        predict_belief(truth, dt=10.0, n_steps=5, derivative_fn=state_derivative)


def test_predict_belief_differs_from_ground_truth_propagation_when_noise_is_nonzero():
    # Proves prediction genuinely uses the noisy belief, not a silent
    # ground-truth shortcut: with real noise, predicting from belief must
    # diverge from propagating ground truth directly.
    from orbit_guard.physics.integrator import propagate

    truth = _ground_truth()
    rng = np.random.default_rng(9)
    belief = form_belief(observe(truth, rng, _REAL_NOISE_MODEL), _REAL_NOISE_MODEL)

    prediction = predict_belief(belief, dt=10.0, n_steps=5, derivative_fn=state_derivative)
    truth_trajectory = propagate(truth.state_vector, 10.0, 5, state_derivative)

    assert not np.allclose(prediction.trajectory, truth_trajectory)


# --- Stage 5: evaluation truth -----------------------------------------------

def test_to_evaluation_truth_wraps_ground_truth_trajectory():
    from orbit_guard.physics.integrator import propagate

    truth = _ground_truth()
    trajectory = propagate(truth.state_vector, 10.0, 3, state_derivative)

    eval_truth = to_evaluation_truth(trajectory)

    assert isinstance(eval_truth, EvaluationTruth)
    assert np.array_equal(eval_truth.trajectory, trajectory)


def test_to_evaluation_truth_rejects_non_array_input():
    with pytest.raises(TypeError):
        to_evaluation_truth([[1.0, 2.0]])  # not an ndarray


# --- Combined relative uncertainty (feeds Part B's risk_proxy) --------------

def test_combined_relative_sigma_matches_quadrature_formula():
    truth = _ground_truth()
    belief_i = form_belief(observe(truth, np.random.default_rng(1), _REAL_NOISE_MODEL), _REAL_NOISE_MODEL)
    belief_j = form_belief(observe(truth, np.random.default_rng(2), _REAL_NOISE_MODEL), _REAL_NOISE_MODEL)

    sigma_rel = combined_relative_sigma(belief_i, belief_j)

    assert sigma_rel == pytest.approx((100.0**2 + 100.0**2) ** 0.5, rel=1e-9)


def test_combined_relative_sigma_is_symmetric():
    truth = _ground_truth()
    belief_i = form_belief(observe(truth, np.random.default_rng(1), _REAL_NOISE_MODEL), _REAL_NOISE_MODEL)
    belief_j = form_belief(
        observe(truth, np.random.default_rng(2), NoiseModel(position_sigma_m=50.0, velocity_sigma_mps=0.1)),
        NoiseModel(position_sigma_m=50.0, velocity_sigma_mps=0.1),
    )

    assert combined_relative_sigma(belief_i, belief_j) == pytest.approx(
        combined_relative_sigma(belief_j, belief_i)
    )


def test_combined_relative_sigma_rejects_ground_truth_input():
    truth = _ground_truth()
    belief = form_belief(observe(truth, np.random.default_rng(1), _REAL_NOISE_MODEL), _REAL_NOISE_MODEL)
    with pytest.raises(TypeError):
        combined_relative_sigma(truth, belief)


# --- Structural leakage guarantee -------------------------------------------

def _field_type_names(cls) -> set[str]:
    return {f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
            for f in dataclasses.fields(cls)}


@pytest.mark.parametrize("policy_facing_type", [AgentBelief, PredictedState])
def test_policy_facing_types_never_declare_a_ground_truth_field(policy_facing_type):
    field_types = _field_type_names(policy_facing_type)
    assert "GroundTruthState" not in field_types


@pytest.mark.parametrize("cls", [GroundTruthState, SensorObservation, AgentBelief, PredictedState, EvaluationTruth])
def test_pipeline_stage_types_are_immutable(cls, request):
    truth = _ground_truth()
    rng = np.random.default_rng(0)
    instance = {
        GroundTruthState: truth,
        SensorObservation: observe(truth, rng, _REAL_NOISE_MODEL),
        AgentBelief: form_belief(observe(truth, rng, _REAL_NOISE_MODEL), _REAL_NOISE_MODEL),
        PredictedState: predict_belief(
            form_belief(observe(truth, rng, _REAL_NOISE_MODEL), _REAL_NOISE_MODEL),
            dt=10.0, n_steps=2, derivative_fn=state_derivative,
        ),
        EvaluationTruth: to_evaluation_truth(truth.state_vector.reshape(1, 6)),
    }[cls]

    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.__setattr__("extra_field", 1)
