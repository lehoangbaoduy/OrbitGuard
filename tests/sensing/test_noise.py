"""Contract tests for orbit_guard.sensing.noise (PRD §15).

Configurable Gaussian state-estimation noise, independently settable per
position/velocity axis. MVP `covariance_mode` is `true_estimate`: the
covariance reported to the agent is exactly the configured noise model's
covariance -- an explicit simplification, not a real filter estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.sensing.noise import NoiseModel, apply_noise, estimated_covariance

_POSITION_SIGMA_M = 100.0
_VELOCITY_SIGMA_MPS = 0.1


def _model() -> NoiseModel:
    return NoiseModel(
        position_sigma_m=_POSITION_SIGMA_M,
        velocity_sigma_mps=_VELOCITY_SIGMA_MPS,
        covariance_mode="true_estimate",
    )


def test_apply_noise_returns_new_array_without_mutating_input():
    true_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    original = true_state.copy()
    rng = np.random.default_rng(0)

    noisy = apply_noise(true_state, rng, _model())

    assert np.array_equal(true_state, original)
    assert noisy is not true_state


def test_apply_noise_is_deterministic_for_a_given_rng_state():
    true_state = np.zeros(6)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    assert np.array_equal(apply_noise(true_state, rng1, _model()), apply_noise(true_state, rng2, _model()))


def test_apply_noise_differs_across_seeds():
    true_state = np.zeros(6)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)

    assert not np.array_equal(apply_noise(true_state, rng1, _model()), apply_noise(true_state, rng2, _model()))


def test_apply_noise_sample_statistics_match_configured_sigma():
    # Large-sample check: sampled additive noise should have ~zero mean and
    # standard deviation close to the configured per-axis sigma.
    true_state = np.zeros(6)
    rng = np.random.default_rng(7)
    model = _model()

    samples = np.array([apply_noise(true_state, rng, model) for _ in range(20_000)])

    position_std = samples[:, 0:3].std(axis=0)
    velocity_std = samples[:, 3:6].std(axis=0)
    assert np.allclose(position_std, _POSITION_SIGMA_M, rtol=0.05)
    assert np.allclose(velocity_std, _VELOCITY_SIGMA_MPS, rtol=0.05)
    assert np.allclose(samples.mean(axis=0), 0.0, atol=5.0)


def test_apply_noise_rejects_wrong_shape_state():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        apply_noise(np.zeros(5), rng, _model())


@pytest.mark.parametrize("bad_kwargs", [
    {"position_sigma_m": -1.0, "velocity_sigma_mps": 0.1, "covariance_mode": "true_estimate"},
    {"position_sigma_m": 100.0, "velocity_sigma_mps": -0.1, "covariance_mode": "true_estimate"},
    {"position_sigma_m": 100.0, "velocity_sigma_mps": 0.1, "covariance_mode": "kalman_filtered"},
])
def test_noise_model_rejects_invalid_construction(bad_kwargs):
    with pytest.raises(ValueError):
        NoiseModel(**bad_kwargs)


def test_estimated_covariance_matches_configured_sigma_in_true_estimate_mode():
    cov = estimated_covariance(_model())

    assert cov.shape == (6, 6)
    expected_diag = np.array([_POSITION_SIGMA_M**2] * 3 + [_VELOCITY_SIGMA_MPS**2] * 3)
    assert np.allclose(np.diag(cov), expected_diag)
    off_diagonal = cov - np.diag(np.diag(cov))
    assert np.allclose(off_diagonal, 0.0)


def test_estimated_covariance_is_deterministic_and_pure():
    model = _model()
    assert np.array_equal(estimated_covariance(model), estimated_covariance(model))
