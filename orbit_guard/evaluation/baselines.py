"""The five compared models, buildable from configuration alone (PRD §31, §35; FR14).

`experiment.model` in the config names one of `MODEL_NAMES`; the `models:`
block says whether that model runs behind the shield and with which margin.
Nothing else differs between baselines, so "baseline switch via config diff
only, no code edit" is literally `build_env_factory(model, ...)` +
`build_policy(model, ...)`.

Rule-Based (frozen heuristic, PRD §31 #2). Fully specified here and never
tuned. It reads only the agent's own observation:

1. A satellite that cannot manoeuvre, or has no budget left, does nothing.
2. Take the top neighbour slot (observations are already ranked by risk). If it
   is not valid, or its risk proxy is below the FIXED threshold
   `RULE_RISK_THRESHOLD`, do nothing.
3. Take that neighbour's relative position (R, T, N) in the agent's own RTN
   frame and pick the component of largest magnitude. Exact ties go to R, then
   T, then N.
4. If that magnitude is below `RELATIVE_POSITION_DEADBAND` (the neighbour sits
   within 5 m of me on every axis), fall back to RADIAL_ADJUSTMENT.
5. Otherwise map the sign of the dominant component to a tangential burn:
   - radial-dominant: neighbour below me (R < 0) -> RAISE_ORBIT, above -> LOWER_ORBIT;
   - along-track-dominant: neighbour ahead (T > 0) -> RAISE_ORBIT (prograde: a
     longer period, so I fall behind), behind -> LOWER_ORBIT;
   - cross-track-dominant: N > 0 -> RAISE_ORBIT, N < 0 -> LOWER_ORBIT.

Burn magnitudes are the fixed ones of the maneuver mapper. The policy is
stateless and deterministic.

Conflicts are resolved by antisymmetry, not by arbitration. The two members of
a pair see each other's relative position with opposite sign on each axis, up
to the small rotation between their two RTN frames. That rotation adds only
second-order same-sign terms: chord curvature (about -d^2/2r) on R, and the
crossing-plane tilt on T, at most tan(theta/2) of N. So the dominant component
flips sign between the members, and they burn opposite tangential directions.
The sign of the along-track component alone is NOT antisymmetric for
crossing-plane encounters: both members see the same T sign. The previous
along-track-only rule therefore made both burn the same way. Limits: the two
members can pick different axes when two components are nearly equal in
magnitude, and the rule only pairs up when each member's top neighbour is the
other. The deadband fallback (+R for both) is symmetric, but it is reachable
only when the belief already puts the pair inside the hard body. Direction
mapping PROVISIONAL pending astrodynamics sign-off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from orbit_guard.config import load_config, margin_from_config
from orbit_guard.rl.actor_policy import NumpyActorPolicy, load_actor_weights
from orbit_guard.rl.env_factory import EnvFactory, make_env_factory_from_config
from orbit_guard.rl.evaluation import Policy, no_maneuver_policy
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
MODEL_NAMES = (
    "no_maneuver",
    "rule_based",
    "fixed_margin_mappo",
    "adaptive_mappo",
    "adaptive_mappo_no_shield",
)

RULE_RISK_THRESHOLD = 0.5
RELATIVE_POSITION_DEADBAND = 1.0e-3  # in observation units (relative position / 5 km)

# Observation indices (rl/observation.py layout).
_OWN_REMAINING_DV, _OWN_MANEUVER_ENABLED = 6, 9
_NEIGHBOR_POSITION, _NEIGHBOR_RISK, _NEIGHBOR_VALID = slice(0, 3), 6, 9
_RADIAL_AXIS = 0


def _tangential_burn_away(relative_position_rtn: np.ndarray) -> ManeuverIntent:
    """Burn direction from the sign of the neighbour's dominant relative-position component (module docstring)."""
    axis = int(np.argmax(np.abs(relative_position_rtn)))  # first maximum: ties go R, then T, then N
    component = float(relative_position_rtn[axis])
    if abs(component) < RELATIVE_POSITION_DEADBAND:
        return ManeuverIntent.RADIAL_ADJUSTMENT
    toward_raise = component < 0.0 if axis == _RADIAL_AXIS else component > 0.0
    return ManeuverIntent.RAISE_ORBIT if toward_raise else ManeuverIntent.LOWER_ORBIT


def rule_based_policy(agent: str, observation: dict) -> int:
    """Frozen fixed-threshold heuristic (module docstring)."""
    own = observation["own"]
    if own[_OWN_MANEUVER_ENABLED] < 0.5 or own[_OWN_REMAINING_DV] <= 0.0:
        return int(ManeuverIntent.NO_MANEUVER)
    top = observation["neighbors"][0]
    if top[_NEIGHBOR_VALID] < 0.5 or top[_NEIGHBOR_RISK] < RULE_RISK_THRESHOLD:
        return int(ManeuverIntent.NO_MANEUVER)
    return int(_tangential_burn_away(np.asarray(top[_NEIGHBOR_POSITION], dtype=float)))


def _spec_from_config(model: str, config: dict) -> dict:
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    return config["models"][model]


def _model_spec(model: str, config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    return _spec_from_config(model, load_config(config_path))


def model_from_config(config_path: str | Path = _DEFAULT_CONFIG) -> str:
    """The model named by `experiment.model`."""
    model = yaml.safe_load(Path(config_path).read_text())["experiment"]["model"]
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    return model


def build_env_factory(
    model: str,
    num_agents: int,
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
    **env_kwargs,
) -> EnvFactory:
    """Environment for `model`: shield on/off and the margin come from the config's `models:` block.

    Every other argument comes from the config (plus the optional tuned overlay); explicit `env_kwargs`,
    such as a shift condition's, win over it.
    """
    config = load_config(config_path, overlay_path)
    spec = _spec_from_config(model, config)
    return make_env_factory_from_config(
        config, num_agents, shielded=bool(spec["shield"]), margin_fn=margin_from_config(config, spec["margin"]),
        **env_kwargs,
    )


def build_policy(
    model: str,
    actor_weights_path: str | Path | None = None,
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
) -> Policy:
    """Decentralized policy for `model`; the learned models need their exported actor weights."""
    kind = _spec_from_config(model, load_config(config_path, overlay_path))["policy"]
    if kind == "no_maneuver":
        return no_maneuver_policy
    if kind == "rule_based":
        return rule_based_policy
    if actor_weights_path is None:
        raise ValueError(f"{model} is a learned model and needs actor weights (an exported actor_weights.npz)")
    return NumpyActorPolicy(load_actor_weights(actor_weights_path))
