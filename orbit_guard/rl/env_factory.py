"""Environment construction from config -- light (no Ray/torch), so evaluation workers can import it.

Every value comes from `configs/default.yaml` (optionally with a tuned overlay) through `orbit_guard.config`
(CORE-1). `make_env_factory` picks the shield-active environment or the fully separate no-shield one
(PRD §28, P4-M6). The shield module is imported only on the shielded branch, so a no-shield run never loads it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from orbit_guard.config import (
    DEFAULT_CONFIG_PATH,
    env_kwargs_from_config,
    load_config,
    reward_weights_from_config,
)
from orbit_guard.safety.adaptive_margin import MarginFn, make_adaptive_margin

_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
EnvFactory = Callable[[dict], Any]
_MARGIN_COEFFICIENTS = ("d_base_m", "d_min_m", "d_max_m", "alpha", "beta", "gamma")


def load_reward_weights(
    config_path: str | Path = _DEFAULT_CONFIG_PATH, overlay_path: str | Path | None = None, **overrides: float
) -> dict[str, float]:
    """Reward weights from the `reward:` block (PRD §25/§39), as `compute_reward` keyword args."""
    weights = reward_weights_from_config(load_config(config_path, overlay_path))
    unknown = set(overrides) - set(weights)
    if unknown:
        raise ValueError(f"unknown reward weights {sorted(unknown)}; expected {sorted(weights)}")
    weights.update(overrides)
    return weights


def load_margin_fn(
    config_path: str | Path = _DEFAULT_CONFIG_PATH, overlay_path: str | Path | None = None, **overrides: float
) -> MarginFn:
    """The PRD §22 adaptive margin from the `adaptive_margin:` block; `overrides` (alpha/beta/gamma/d_*) win."""
    block = load_config(config_path, overlay_path)["adaptive_margin"]
    coefficients = {key: float(block[key]) for key in _MARGIN_COEFFICIENTS}
    unknown = set(overrides) - set(coefficients)
    if unknown:
        raise ValueError(f"unknown margin coefficients {sorted(unknown)}; expected {sorted(coefficients)}")
    coefficients.update(overrides)
    return make_adaptive_margin(**coefficients)


def _build_env(config: dict, *, shielded: bool, num_agents: int, reward_weights: dict, margin_fn, env_kwargs: dict):
    """Module-level so a `partial` of it pickles into evaluation worker processes."""
    if shielded:
        from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv

        return ShieldedOrbitGuardEnv(
            num_agents=num_agents, reward_weights=reward_weights, margin_fn=margin_fn, **env_kwargs
        )

    from orbit_guard.rl.environment import OrbitGuardEnv

    return OrbitGuardEnv(num_agents=num_agents, reward_weights=reward_weights, margin_fn=margin_fn, **env_kwargs)


def make_env_factory(
    num_agents: int,
    reward_weights: dict[str, float],
    shielded: bool,
    margin_fn: MarginFn | None = None,
    **env_kwargs: Any,
) -> EnvFactory:
    """Env factory for a training/eval run: shield-active or a fully separate no-shield environment.

    The shield module is imported only on the shielded branch, so a no-shield run never loads it
    (PRD §28, P4-M6). Extra `env_kwargs` (shift-condition arguments etc.) go straight to the env.
    The returned factory is picklable.
    """
    return partial(
        _build_env,
        shielded=shielded, num_agents=num_agents, reward_weights=reward_weights,
        margin_fn=margin_fn, env_kwargs=env_kwargs,
    )


def make_env_factory_from_config(
    config: Mapping[str, Any],
    num_agents: int,
    *,
    shielded: bool,
    margin_fn: MarginFn | None,
    **env_kwargs: Any,
) -> EnvFactory:
    """`make_env_factory` with every environment argument taken from `config` (explicit `env_kwargs` win)."""
    merged = {**env_kwargs_from_config(config, shielded=shielded), **env_kwargs}
    reward_weights = merged.pop("reward_weights")
    return make_env_factory(num_agents, reward_weights, shielded=shielded, margin_fn=margin_fn, **merged)
