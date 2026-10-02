"""The five compared models, buildable from configuration alone (PRD §31, §35; FR14).

`experiment.model` in the config names one of `MODEL_NAMES`; the `models:`
block says whether that model runs behind the shield and with which margin.
Nothing else differs between baselines, so "baseline switch via config diff
only, no code edit" is literally `build_env_factory(model, ...)` +
`build_policy(model, ...)`.

Rule-Based (frozen heuristic, PRD §31 #2). Fully specified here and never
tuned: from the agent's own observation, take the highest-risk valid neighbour
(observations are already ranked by risk). If its risk proxy is at or above the
FIXED threshold `RULE_RISK_THRESHOLD`, burn along-track away from it -- neighbour
ahead of me (+T) -> RAISE_ORBIT (prograde: a longer period, so I fall behind);
neighbour behind me -> LOWER_ORBIT; neighbour level with me within
`ALONG_TRACK_DEADBAND` -> RADIAL_ADJUSTMENT. Satellites that cannot manoeuvre,
or have no budget, do nothing. Conflicts need no arbitration: when both members
of a pair run the rule they see each other with opposite along-track signs and
so burn in opposite directions. The policy is stateless and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from orbit_guard.rl.actor_policy import NumpyActorPolicy, load_actor_weights
from orbit_guard.rl.env_factory import (
    EnvFactory,
    load_margin_fn,
    load_reward_weights,
    make_env_factory,
)
from orbit_guard.rl.evaluation import Policy, no_maneuver_policy
from orbit_guard.safety.adaptive_margin import FixedMargin
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
ALONG_TRACK_DEADBAND = 1.0e-3  # in observation units (relative position / 5 km)

# Observation indices (rl/observation.py layout).
_OWN_REMAINING_DV, _OWN_MANEUVER_ENABLED = 6, 9
_NEIGHBOR_ALONG_TRACK, _NEIGHBOR_RISK, _NEIGHBOR_VALID = 1, 6, 9


def rule_based_policy(agent: str, observation: dict) -> int:
    """Frozen fixed-threshold heuristic (module docstring)."""
    own = observation["own"]
    if own[_OWN_MANEUVER_ENABLED] < 0.5 or own[_OWN_REMAINING_DV] <= 0.0:
        return int(ManeuverIntent.NO_MANEUVER)
    top = observation["neighbors"][0]
    if top[_NEIGHBOR_VALID] < 0.5 or top[_NEIGHBOR_RISK] < RULE_RISK_THRESHOLD:
        return int(ManeuverIntent.NO_MANEUVER)
    along_track = top[_NEIGHBOR_ALONG_TRACK]
    if abs(along_track) < ALONG_TRACK_DEADBAND:
        return int(ManeuverIntent.RADIAL_ADJUSTMENT)
    return int(ManeuverIntent.RAISE_ORBIT if along_track > 0 else ManeuverIntent.LOWER_ORBIT)


def _model_spec(model: str, config_path: str | Path) -> dict:
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    return yaml.safe_load(Path(config_path).read_text())["models"][model]


def model_from_config(config_path: str | Path = _DEFAULT_CONFIG) -> str:
    """The model named by `experiment.model`."""
    model = yaml.safe_load(Path(config_path).read_text())["experiment"]["model"]
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_NAMES}")
    return model


def build_env_factory(
    model: str, num_agents: int, *, config_path: str | Path = _DEFAULT_CONFIG, **env_kwargs
) -> EnvFactory:
    """Environment for `model`: shield on/off and the margin come from the config's `models:` block."""
    spec = _model_spec(model, config_path)
    config = yaml.safe_load(Path(config_path).read_text())
    margin_fn = (
        FixedMargin(float(config["fixed_margin"]["d_safe_m"]))
        if spec["margin"] == "fixed"
        else load_margin_fn(config_path)
    )
    return make_env_factory(
        num_agents, load_reward_weights(config_path), shielded=bool(spec["shield"]), margin_fn=margin_fn, **env_kwargs
    )


def build_policy(
    model: str, actor_weights_path: str | Path | None = None, *, config_path: str | Path = _DEFAULT_CONFIG
) -> Policy:
    """Decentralized policy for `model`; the learned models need their exported actor weights."""
    kind = _model_spec(model, config_path)["policy"]
    if kind == "no_maneuver":
        return no_maneuver_policy
    if kind == "rule_based":
        return rule_based_policy
    if actor_weights_path is None:
        raise ValueError(f"{model} is a learned model and needs actor weights (an exported actor_weights.npz)")
    return NumpyActorPolicy(load_actor_weights(actor_weights_path))
