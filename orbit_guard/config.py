"""Single source of truth for run configuration (CORE-1): configs/default.yaml plus an optional tuned overlay.

Every simulation parameter used to be duplicated as a Python constant; this module is the one place the
YAML is read into environment arguments, margin functions and provenance digests. Only PyYAML and the
standard library are imported at module level; the environment value types are imported inside the
functions that build them, so the loader stays light for physics-free layers.

An overlay (`tuning/grid_search.freeze_best` writes one) may override only the `reward`, `adaptive_margin`,
`fixed_margin` and `rl` blocks, key by key. `fixed_margin` is overridable so freeze_best can re-match the
Fixed-Margin constant to tuned adaptive coefficients (tuning/fixed_margin.py, progress Q30 protocol). Reward overlay keys use the `compute_reward` names (w1..w4, collision_penalty).
When an overlay carries `config_hash` it must have been produced against these base config bytes:
sha256(base bytes + canonical JSON of the overlay body without the hash), the PRD section 33 freeze rule.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

OVERLAY_BLOCKS = ("reward", "adaptive_margin", "fixed_margin", "rl")
OVERLAY_METADATA_KEYS = ("selected", "selected_on", "validation", "config_hash")
REWARD_OVERLAY_KEYS = {
    "w1": "w1_safety",
    "w2": "w2_mission",
    "w3": "w3_delta_v",
    "w4": "w4_coordination",
    "collision_penalty": "collision_penalty",
}
# compute_mission_term / compute_weighted_mission_deviation keyword -> the yaml key under reward.mission_weights.
MISSION_WEIGHT_KEYS = {"q_a": "q_a", "q_e": "q_e", "q_i": "q_i", "q_m": "q_M"}
MARGIN_KINDS = ("adaptive", "fixed")
_ADAPTIVE_COEFFICIENTS = ("d_base_m", "d_min_m", "d_max_m", "alpha", "beta", "gamma")


def overlay_digest(base_bytes: bytes, body: Mapping[str, Any]) -> str:
    """sha256 of the base config bytes followed by the canonical JSON of an overlay body (no `config_hash`)."""
    return hashlib.sha256(base_bytes + json.dumps(dict(body), sort_keys=True).encode()).hexdigest()


def config_digest(config_path: str | Path = DEFAULT_CONFIG_PATH, overlay_path: str | Path | None = None) -> str:
    """Provenance hash of the configuration a run used: base bytes, then the TLE data, then the overlay.

    The TLE file named by `data.tle_file` defines the scenario distribution, so its bytes are part of the
    configuration: it is resolved exactly as `env_kwargs_from_config` resolves it (relative to the repository
    root), and a missing file raises FileNotFoundError. An overlay cannot change `data`, so only the base
    config is read for it. A config with no `data.tle_file` key (not runnable by the environment) digests
    its own bytes and the overlay only.
    """
    base_bytes = Path(config_path).read_bytes()
    digest = hashlib.sha256(base_bytes)
    tle_path = _tle_path(base_bytes, Path(config_path))
    if tle_path is not None:
        digest.update(tle_path.read_bytes())
    if overlay_path is not None:
        digest.update(Path(overlay_path).read_bytes())
    return digest.hexdigest()


def _tle_path(base_bytes: bytes, source: Path) -> Path | None:
    """The TLE file `data.tle_file` names (resolved like the environment's tle_path), or None if unset."""
    data = _parse_mapping(base_bytes, source).get("data")
    tle_file = data.get("tle_file") if isinstance(data, Mapping) else None
    if tle_file is None:
        return None
    path = _repo_path(str(tle_file))
    if not path.is_file():
        raise FileNotFoundError(
            f"config {source}: data.tle_file {tle_file!r} resolves to {path}, which does not exist; "
            "the TLE data is part of the configuration digest"
        )
    return path


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH, overlay_path: str | Path | None = None) -> dict:
    """The configuration as a fresh dict: the YAML at `config_path`, with `overlay_path` merged on top if given."""
    base_path = Path(config_path)
    base_bytes = base_path.read_bytes()
    config = _parse_mapping(base_bytes, base_path)
    if overlay_path is None:
        return config

    overlay_source = Path(overlay_path)
    overlay = _parse_mapping(overlay_source.read_bytes(), overlay_source)
    if "config_hash" in overlay:
        body = {key: value for key, value in overlay.items() if key != "config_hash"}
        if overlay_digest(base_bytes, body) != overlay["config_hash"]:
            raise ValueError(
                f"overlay {overlay_source} config_hash does not match base config {base_path}: "
                "it was produced against a different configuration"
            )
    return apply_overlay(config, overlay)


def apply_overlay(config: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict:
    """Merge `overlay` into a copy of `config`; only OVERLAY_BLOCKS may change, and only to known keys."""
    unknown = sorted(set(overlay) - set(OVERLAY_BLOCKS) - set(OVERLAY_METADATA_KEYS))
    if unknown:
        raise ValueError(f"overlay may only override {list(OVERLAY_BLOCKS)}; unknown top-level keys {unknown}")
    merged = copy.deepcopy(dict(config))
    for block in OVERLAY_BLOCKS:
        if block not in overlay:
            continue
        values = overlay[block] or {}
        if not isinstance(values, Mapping):
            raise TypeError(f"overlay block {block!r} must be a mapping")
        merged[block] = _merge_block(block, merged[block], values)
    return merged


def reward_weights_from_config(config: Mapping[str, Any]) -> dict[str, float]:
    """`compute_reward` keyword weights from the `reward:` block (PRD section 25/39)."""
    reward = config["reward"]
    return {
        "w1": reward["w1_safety"],
        "w2": reward["w2_mission"],
        "w3": reward["w3_delta_v"],
        "w4": reward["w4_coordination"],
        "collision_penalty": reward["collision_penalty"],
    }


def margin_from_config(config: Mapping[str, Any], model_margin: str = "adaptive"):
    """The safety margin a model uses: the PRD section 22 adaptive margin, or the Fixed-Margin constant."""
    from orbit_guard.safety.adaptive_margin import FixedMargin, make_adaptive_margin

    if model_margin == "fixed":
        return FixedMargin(float(config["fixed_margin"]["d_safe_m"]))
    if model_margin != "adaptive":
        raise ValueError(f"model_margin must be one of {list(MARGIN_KINDS)}, got {model_margin!r}")
    block = config["adaptive_margin"]
    return make_adaptive_margin(**{key: float(block[key]) for key in _ADAPTIVE_COEFFICIENTS})


def env_kwargs_from_config(config: Mapping[str, Any], *, shielded: bool) -> dict:
    """Every OrbitGuardEnv constructor argument the config determines (the shield search too when `shielded`).

    `num_agents` and `margin_fn` are not included: the caller chooses the constellation size and the model's
    margin (`margin_from_config`). Unsupported modelling choices and malformed values raise ValueError here,
    so a config can never silently run something other than what the code implements.
    """
    from orbit_guard.physics.state import SatelliteCapability
    from orbit_guard.sensing.noise import NoiseModel

    sim, sensing, risk = config["simulation"], config["sensing"], config["risk"]
    maneuver, capability, adaptive = config["maneuver"], config["capability"], config["adaptive_margin"]
    physics_dt_s, control_dt_s = _check_timestep(sim["physics_dt_s"], sim["control_dt_s"])
    _choice(config["dynamics"]["model"], ("two_body_j2",), "dynamics.model")
    _choice(config["dynamics"]["integrator"], ("rk4",), "dynamics.integrator")
    _choice(sensing["noise_model"], ("gaussian",), "sensing.noise_model")
    _choice(sensing["covariance_mode"], ("true_estimate",), "sensing.covariance_mode")
    _choice(risk["model"], ("normalized_separation_proxy",), "risk.model")
    _choice(config["shield"]["repair_mode"], ("bounded_joint_search",), "shield.repair_mode")
    mix = {str(key): float(value) for key, value in config["experiment"]["training_distribution_mix"].items()}
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"experiment.training_distribution_mix must sum to 1.0, got {sum(mix.values())}")
    reward_mission = config["reward"]["mission_weights"]

    kwargs: dict[str, Any] = {
        "episode_horizon_s": _positive(sim["episode_horizon_s"], "simulation.episode_horizon_s"),
        "physics_dt_s": physics_dt_s,
        "control_dt_s": control_dt_s,
        "lookahead_s": _positive(sim["lookahead_s"], "simulation.lookahead_s"),
        "k_neighbors": _positive_int(config["observation"]["k_neighbors"], "observation.k_neighbors"),
        "hard_body_radius_m": _positive(config["collision"]["hard_body_radius_m"], "collision.hard_body_radius_m"),
        "capability_template": SatelliteCapability(
            max_delta_v_per_action_mps=_non_negative(maneuver["base_delta_v_mps"], "maneuver.base_delta_v_mps"),
            remaining_delta_v_mps=_non_negative(capability["remaining_delta_v_mps"], "capability.remaining_delta_v_mps"),
            maneuver_enabled=True,
            mission_deviation_limit=_positive(capability["mission_deviation_limit"], "capability.mission_deviation_limit"),
        ),
        "remaining_delta_v_range_mps": budget_range(
            capability.get("remaining_delta_v_range_mps"), "capability.remaining_delta_v_range_mps"
        ),
        "distribution_mix": mix,
        "noise_model": NoiseModel(
            position_sigma_m=_non_negative(sensing["position_sigma_m"], "sensing.position_sigma_m"),
            velocity_sigma_mps=_non_negative(sensing["velocity_sigma_mps"], "sensing.velocity_sigma_mps"),
            covariance_mode=sensing["covariance_mode"],
        ),
        "tle_path": _repo_path(config["data"]["tle_file"]),
        "reward_weights": reward_weights_from_config(config),
        "coordination_clip": _positive(config["reward"]["coordination_clip"], "reward.coordination_clip"),
        "phase_fraction": _non_negative(maneuver["phase_fraction"], "maneuver.phase_fraction"),
        "risk_epsilon_m": _positive(risk["epsilon_m"], "risk.epsilon_m"),
        "risk_z0": _non_negative(risk["z0"], "risk.z0"),
        "risk_k": _non_negative(risk["k"], "risk.k"),
        "density_radius_m": _positive(adaptive["density_radius_km"], "adaptive_margin.density_radius_km") * 1000.0,
        "density_ref_count": _positive(adaptive["density_ref_count"], "adaptive_margin.density_ref_count"),
        "mission_weights": {
            kwarg: _non_negative(reward_mission[key], f"reward.mission_weights.{key}")
            for kwarg, key in MISSION_WEIGHT_KEYS.items()
        },
    }
    if shielded:
        kwargs["shield_exact_max_agents"] = _positive_int(
            config["shield"]["exact_enumeration_max_agents"], "shield.exact_enumeration_max_agents"
        )
        kwargs["shield_beam_width"] = _positive_int(config["shield"]["beam_width"], "shield.beam_width")
    return kwargs


def budget_range(value: Any, name: str = "remaining_delta_v_range_mps") -> tuple[float, float] | None:
    """A per-episode Δv budget range (PRD §17): None stays None, otherwise two finite numbers with 0 <= low <= high."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError(f"{name} must be null or [low, high], got {value!r}")
    low, high = (_non_negative(bound, name) for bound in value)
    if low > high:
        raise ValueError(f"{name} must have low <= high, got {value!r}")
    return low, high


def _parse_mapping(raw: bytes, source: Path) -> dict:
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise TypeError(f"{source} must contain a YAML mapping")
    return parsed


def _merge_block(block: str, base_block: Mapping[str, Any], values: Mapping[str, Any]) -> dict:
    if block == "reward":
        unknown = sorted(set(values) - set(REWARD_OVERLAY_KEYS))
        if unknown:
            raise ValueError(f"unknown reward overlay keys {unknown}; expected {sorted(REWARD_OVERLAY_KEYS)}")
        translated = {REWARD_OVERLAY_KEYS[key]: value for key, value in values.items()}
    else:
        unknown = sorted(set(values) - set(base_block))
        if unknown:
            raise ValueError(f"unknown keys {unknown} in overlay block {block!r}")
        translated = dict(values)
    return {**base_block, **translated}


def _choice(value: Any, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def _non_negative(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a whole number >= 1, got {value!r}")
    return int(value)


def _check_timestep(physics: Any, control: Any) -> tuple[float, float]:
    physics_dt_s = _positive(physics, "simulation.physics_dt_s")
    control_dt_s = _positive(control, "simulation.control_dt_s")
    ratio = control_dt_s / physics_dt_s
    if ratio < 1.0 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"simulation.control_dt_s must be a whole multiple of simulation.physics_dt_s (>= 1x), "
            f"got control {control_dt_s} and physics {physics_dt_s}"
        )
    return physics_dt_s, control_dt_s


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate
