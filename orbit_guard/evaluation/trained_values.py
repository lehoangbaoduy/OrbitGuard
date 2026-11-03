"""The margin and reward weights a checkpoint was trained with, against the ones its evaluation will use.

config_digest covers the config and overlay files, but the training CLI's --margin-alpha/--margin-beta/
--margin-gamma-margin and --w1..--w4/--collision-penalty flags change the training environment without changing
it, while evaluation (baselines.build_env_factory, the replay recorder) rebuilds every environment from config +
overlay only. The run_config.json sidecar records the margin (`margin_record`) and reward weights the run really
used; `trained_value_mismatches` compares them with what the evaluation environment for the model will build, so
a checkpoint trained under CLI overrides is refused instead of evaluated under a margin or reward it never saw.
`run_shape_mismatches` likewise compares the recorded `shield` and `num_agents` with the model's shield setting and
the constellation size the evaluation runs at.

Ray-free (evaluation workers never import Ray): only the config loader and the margin value types are used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from orbit_guard.config import (
    MARGIN_KINDS,
    margin_from_config,
    reward_weights_from_config,
)
from orbit_guard.safety.adaptive_margin import margin_record

RECORDED_KEYS = ("margin", "reward_weights")
OVERLAY_ADVICE = (
    "training CLI overrides (--margin-alpha/--margin-beta/--margin-gamma-margin, --w1..--w4, --collision-penalty) "
    "never reach evaluation, which rebuilds the environment from config + overlay: put tuned values in the overlay "
    "(tuning/grid_search tuned.yaml), retrain with --overlay, and evaluate with the same --overlay"
)


def evaluation_values(config: Mapping[str, Any], margin_kind: str) -> dict[str, dict]:
    """The margin record and reward weights an evaluation environment built from `config` uses for `margin_kind`."""
    return {
        "margin": margin_record(margin_from_config(config, margin_kind)),
        "reward_weights": dict(reward_weights_from_config(config)),
    }


def recorded_margin_kind(recorded: Mapping[str, Any]) -> str:
    """The margin kind a sidecar says the run trained with ('adaptive' when absent or unreadable: then the
    comparison reports the missing or malformed record itself)."""
    margin = recorded.get("margin")
    kind = margin.get("kind") if isinstance(margin, Mapping) else None
    return kind if kind in MARGIN_KINDS else "adaptive"


def trained_value_mismatches(recorded: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """One line per recorded value (margin, reward weights) that is missing or differs from `expected`."""
    problems = []
    for key in RECORDED_KEYS:
        label = key.replace("_", " ")
        if key not in recorded:
            problems.append(f"it records no {key}")
        elif not _same_values(recorded[key], expected[key]):
            problems.append(f"{label}: trained with {recorded[key]!r}, evaluation would use {dict(expected[key])!r}")
    return problems


RUN_SHAPE_ADVICE = (
    "register each checkpoint under the model it was trained as (shield on/off) and evaluate it at the N it was "
    "trained at (--n-agents; the queue trains every model at N=15)"
)


def run_shape_mismatches(
    recorded: Mapping[str, Any], *, shield: bool | None, num_agents: int | None, model: str | None
) -> list[str]:
    """One line per problem with the sidecar's `shield` / `num_agents`: missing, malformed, or (when the expected
    value is given) different. The record must be a real bool / int: 1 is not True and 15.0 is not 15."""
    problems = []
    if "shield" not in recorded:
        problems.append("it records no shield")
    elif not isinstance(recorded["shield"], bool):
        problems.append(f"its recorded shield {recorded['shield']!r} is not true/false")
    elif shield is not None and recorded["shield"] != shield:
        problems.append(
            f"shield: trained with the shield {'on' if recorded['shield'] else 'off'}, but {model} evaluates with "
            f"it {'on' if shield else 'off'}"
        )
    count = recorded.get("num_agents")
    if "num_agents" not in recorded:
        problems.append("it records no num_agents")
    elif isinstance(count, bool) or not isinstance(count, int):
        problems.append(f"its recorded num_agents {count!r} is not an integer")
    elif num_agents is not None and count != num_agents:
        problems.append(f"num_agents: trained with {count} agents, evaluation would run {num_agents}")
    return problems


def _same_values(recorded: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(recorded, Mapping) or set(recorded) != set(expected):
        return False
    return all(_same_value(recorded[key], expected[key]) for key in expected)


def _same_value(recorded: Any, expected: Any) -> bool:
    numbers = (int, float)
    if isinstance(expected, numbers) and not isinstance(expected, bool):
        if not isinstance(recorded, numbers) or isinstance(recorded, bool):
            return False
        return math.isclose(float(recorded), float(expected), rel_tol=1e-12, abs_tol=0.0)
    return recorded == expected
