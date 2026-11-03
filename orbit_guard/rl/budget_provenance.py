"""Per-satellite Δv budget provenance for one episode (replay fix b).

Since CORE-3a every satellite's initial budget is drawn per episode, so a starting tank below the nominal budget
no longer means the capability_reduction shift touched that satellite. This records the DRAWN budget (before any
degradation) and a mask of the satellites whose budget the degradation actually cut, so the replay can flag
"limited" satellites without guessing. A thruster the degradation disabled is not marked here: it is reported as
offline through `maneuver_enabled`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from orbit_guard.physics.state import SatelliteCapability

# Budgets are scaled, never rounded, so any genuine cut is far larger than this; it only absorbs float noise.
REDUCED_EPS_MPS = 1e-9


@dataclass(frozen=True)
class BudgetProvenance:
    drawn_mps: tuple[float, ...]  # each satellite's initial budget as drawn, before capability degradation
    degraded: tuple[bool, ...]  # the capability degradation reduced this satellite's budget


def budget_provenance(
    drawn: Sequence[SatelliteCapability], after_degradation: Sequence[SatelliteCapability]
) -> BudgetProvenance:
    """Compare the fleet's capabilities before and after degradation, satellite by satellite."""
    if len(drawn) != len(after_degradation):
        raise ValueError(
            f"before/after degradation must list the same satellites ({len(drawn)} != {len(after_degradation)})"
        )
    return BudgetProvenance(
        drawn_mps=tuple(float(c.remaining_delta_v_mps) for c in drawn),
        degraded=tuple(
            bool(after.remaining_delta_v_mps < before.remaining_delta_v_mps - REDUCED_EPS_MPS)
            for before, after in zip(drawn, after_degradation, strict=True)
        ),
    )
