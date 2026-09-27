"""Ground truth -> observation -> belief -> prediction -> evaluation-truth
pipeline (PRD §14) -- the project's highest-risk correctness boundary.

Five distinct, non-interchangeable stage types. The policy must never see a
GroundTruthState; every function past `observe()` accepts only the stage
type immediately before it and raises TypeError on anything else (including
a raw ndarray), so a leakage mistake fails loudly at the call site instead
of silently producing a policy input derived from ground truth.

    GroundTruthState --observe()--> SensorObservation --form_belief()-->
    AgentBelief --predict_belief()--> PredictedState

    GroundTruthState (trajectory) --to_evaluation_truth()--> EvaluationTruth
    (used only for collision determination and final metrics, §14 point 5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.physics.integrator import StateDerivative, propagate
from orbit_guard.sensing.noise import NoiseModel, apply_noise, estimated_covariance


@dataclass(frozen=True)
class GroundTruthState:
    """Exact simulated physical state. Never exposed to the policy."""

    state_vector: np.ndarray


@dataclass(frozen=True)
class SensorObservation:
    """Noisy measurement of ground truth (PRD §15)."""

    state_vector: np.ndarray


@dataclass(frozen=True)
class AgentBelief:
    """The estimate actually exposed to the RL policy: observed state + reported covariance."""

    state_vector: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class PredictedState:
    """Future trajectory propagated from the *belief* state, never ground truth."""

    trajectory: np.ndarray
    dt_s: float


@dataclass(frozen=True)
class EvaluationTruth:
    """Ground-truth trajectory, used only for collision determination and final metrics."""

    trajectory: np.ndarray


def observe(ground_truth: GroundTruthState, rng: np.random.Generator, model: NoiseModel) -> SensorObservation:
    if not isinstance(ground_truth, GroundTruthState):
        raise TypeError(
            f"observe() requires a GroundTruthState, got {type(ground_truth).__name__} "
            "(PRD §14: this is the only function allowed to touch ground truth)"
        )
    noisy = apply_noise(ground_truth.state_vector, rng, model)
    return SensorObservation(state_vector=noisy)


def form_belief(observation: SensorObservation, model: NoiseModel) -> AgentBelief:
    if not isinstance(observation, SensorObservation):
        raise TypeError(
            f"form_belief() requires a SensorObservation, got {type(observation).__name__} "
            "-- never call this with GroundTruthState (PRD §14 leakage boundary)"
        )
    return AgentBelief(state_vector=observation.state_vector, covariance=estimated_covariance(model))


def predict_belief(
    belief: AgentBelief, dt: float, n_steps: int, derivative_fn: StateDerivative
) -> PredictedState:
    if not isinstance(belief, AgentBelief):
        raise TypeError(
            f"predict_belief() requires an AgentBelief, got {type(belief).__name__} "
            "-- prediction must use the estimated state, never ground truth (PRD §14 point 4)"
        )
    trajectory = propagate(belief.state_vector, dt, n_steps, derivative_fn)
    return PredictedState(trajectory=trajectory, dt_s=dt)


def to_evaluation_truth(ground_truth_trajectory: np.ndarray) -> EvaluationTruth:
    if not isinstance(ground_truth_trajectory, np.ndarray):
        raise TypeError(
            f"to_evaluation_truth() requires an ndarray trajectory, got {type(ground_truth_trajectory).__name__}"
        )
    return EvaluationTruth(trajectory=ground_truth_trajectory)


def combined_relative_sigma(belief_i: AgentBelief, belief_j: AgentBelief) -> float:
    """g(Sigma_i, Sigma_j) of PRD §21: combined relative position uncertainty, in meters.

    Position noise is isotropic per axis in the MVP model (§15), so the [0,0]
    diagonal entry alone represents each satellite's position variance.
    Combines two independent Gaussians via quadrature -- this is the plain
    float `sigma_rel_m` handed to Part B's `conjunction.risk.risk_proxy`.
    """
    if not isinstance(belief_i, AgentBelief) or not isinstance(belief_j, AgentBelief):
        raise TypeError("combined_relative_sigma() requires AgentBelief inputs, never GroundTruthState")
    sigma_i = float(np.sqrt(belief_i.covariance[0, 0]))
    sigma_j = float(np.sqrt(belief_j.covariance[0, 0]))
    return float(np.sqrt(sigma_i**2 + sigma_j**2))
