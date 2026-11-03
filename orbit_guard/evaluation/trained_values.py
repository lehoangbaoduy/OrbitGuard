"""The margin and reward weights a checkpoint was trained with, against the ones its evaluation will use.

config_digest covers the config and overlay files, but the training CLI's --margin-alpha/--margin-beta/
--margin-gamma-margin and --w1..--w4/--collision-penalty flags change the training environment without changing
it, while evaluation (baselines.build_env_factory, the replay recorder) rebuilds every environment from config +
overlay only. The run_config.json sidecar records the margin (`margin_record`) and reward weights the run really
used; `trained_value_mismatches` compares them with what the evaluation environment for the model will build, so
a checkpoint trained under CLI overrides is refused instead of evaluated under a margin or reward it never saw.

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
