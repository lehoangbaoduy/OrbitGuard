"""Zero-shot distribution-shift conditions (PRD §32, §35) as environment keyword arguments.

Each condition is a dict of `OrbitGuardEnv` keyword arguments read from the
`shifts:` block of the config, so every model -- baselines included -- is
evaluated against byte-identical conditions, and Acceptance Test G's "all four
shifts from one experiment-config interface" is `condition_env_kwargs(name)`.
Training, hyperparameters, margin coefficients and shield thresholds are frozen;
nothing here touches them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from orbit_guard.rl.environment import CapabilityDegradation
from orbit_guard.sensing.noise import NoiseModel

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
CONDITION_NAMES = (
    "normal",
    "noise_increase",
    "density_increase",
    "simultaneous_conjunctions",
    "capability_reduction",
)


def condition_env_kwargs(name: str, config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    """`OrbitGuardEnv` kwargs for one shift condition (empty for `normal`)."""
    config = yaml.safe_load(Path(config_path).read_text())
    spec = config["shifts"][name]  # KeyError on an unknown condition, deliberately
    kwargs: dict = {}
    if "noise_scale" in spec:
        sensing = config["sensing"]
        kwargs["noise_model"] = NoiseModel(
            position_sigma_m=float(sensing["position_sigma_m"]) * float(spec["noise_scale"]),
            velocity_sigma_mps=float(sensing["velocity_sigma_mps"]) * float(spec["noise_scale"]),
        )
    if "phase_spread_rad" in spec:
        kwargs["phase_spread_rad"] = float(spec["phase_spread_rad"])
    if "distribution_mix" in spec:
        kwargs["distribution_mix"] = {key: float(value) for key, value in spec["distribution_mix"].items()}
    if "disabled_count" in spec or "reduced_count" in spec:
        kwargs["capability_degradation"] = CapabilityDegradation(
            disabled_count=int(spec.get("disabled_count", 0)),
            reduced_count=int(spec.get("reduced_count", 0)),
            reduced_budget_scale=float(spec.get("reduced_budget_scale", 0.25)),
        )
    return kwargs
