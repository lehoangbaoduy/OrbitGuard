"""Episode totals for the summary strip, from the summary the environment recorded in the trace header."""

from __future__ import annotations

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


def summary_totals(summary: Mapping[str, Any], meta: ReplayMeta) -> SummaryTotals:
    missing = [field for field in _FIELDS if field not in summary]
    if missing:
        raise ReplayMetaError(f"trace summary is missing field(s): {', '.join(missing)}")
    return SummaryTotals(
        delta_v_mps=float(summary["total_delta_v_mps"]),
        maneuvers=int(summary["maneuver_count"]),
        shield=int(summary["shield_intervention_count"]) if meta.shield else None,
        collisions=int(summary["collision_count"]),
        mean_deviation=float(summary["mean_mission_deviation"]),
    )
