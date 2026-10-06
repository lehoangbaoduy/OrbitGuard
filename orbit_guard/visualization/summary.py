"""Episode totals for the summary strip, from the summary the environment recorded in the trace header."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from orbit_guard.visualization.replay_meta import ReplayMeta, ReplayMetaError

_FIELDS = ("total_delta_v_mps", "maneuver_count", "shield_intervention_count", "collision_count", "mean_mission_deviation")


@dataclass(frozen=True)
class SummaryTotals:
    delta_v_mps: float
    maneuvers: int
    shield: int | None  # None when the shield was off: "not applicable", never 0
    collisions: int
    mean_deviation: float


def _number(summary: Mapping[str, Any], field: str, kind: type) -> Any:
    """A number of `kind`, or a `ReplayMetaError` naming the field (a header is operator-supplied data, not trusted).

    A non-finite float is a recorded "no value" and is kept (the UI prints a dash); counts must be finite integers.
    """
    value = summary[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and not math.isfinite(value)):
        raise ReplayMetaError(f"trace summary field {field} must be a number, got {value!r}"[:200])
    return kind(value)


def summary_totals(summary: Mapping[str, Any], meta: ReplayMeta) -> SummaryTotals:
    missing = [field for field in _FIELDS if field not in summary]
    if missing:
        raise ReplayMetaError(f"trace summary is missing field(s): {', '.join(missing)}")
    return SummaryTotals(
        delta_v_mps=_number(summary, "total_delta_v_mps", float),
        maneuvers=_number(summary, "maneuver_count", int),
        shield=_number(summary, "shield_intervention_count", int) if meta.shield else None,
        collisions=_number(summary, "collision_count", int),
        mean_deviation=_number(summary, "mean_mission_deviation", float),
    )
