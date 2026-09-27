"""Gaussian telemetry & uncertainty model (PRD §15).

MVP: independent, isotropic-per-axis Gaussian noise on position and velocity.
`covariance_mode="true_estimate"` means the covariance reported to the agent
is exactly the configured noise model's own covariance -- an explicit MVP
simplification (stated as such in the paper), not a real filter estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SUPPORTED_COVARIANCE_MODES = ("true_estimate",)


@dataclass(frozen=True)
class NoiseModel:
    position_sigma_m: float
    velocity_sigma_mps: float
    covariance_mode: str = "true_estimate"

    def __post_init__(self) -> None:
        if self.position_sigma_m < 0:
            raise ValueError(f"position_sigma_m must be >= 0, got {self.position_sigma_m}")
        if self.velocity_sigma_mps < 0:
            raise ValueError(f"velocity_sigma_mps must be >= 0, got {self.velocity_sigma_mps}")
        if self.covariance_mode not in _SUPPORTED_COVARIANCE_MODES:
            raise ValueError(
                f"Unsupported covariance_mode {self.covariance_mode!r}; "
                f"MVP only supports {_SUPPORTED_COVARIANCE_MODES}"
            )


def apply_noise(true_state_vector: np.ndarray, rng: np.random.Generator, model: NoiseModel) -> np.ndarray:
    """Return true_state_vector + independent Gaussian noise (a new array).

    `rng` is an explicit, caller-owned generator -- no hidden global RNG state,
    matching the project's determinism requirements for scenario generation.
    """
    if true_state_vector.shape != (6,):
        raise ValueError(f"Expected a 6-vector [r, v], got shape {true_state_vector.shape}")

    sigma = np.array([model.position_sigma_m] * 3 + [model.velocity_sigma_mps] * 3)
    noise = rng.normal(loc=0.0, scale=sigma)
    return true_state_vector + noise


def estimated_covariance(model: NoiseModel) -> np.ndarray:
    """Diagonal 6x6 covariance reported to the agent (PRD §15 `true_estimate` mode)."""
    diag = np.array([model.position_sigma_m**2] * 3 + [model.velocity_sigma_mps**2] * 3)
    return np.diag(diag)
