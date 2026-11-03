"""Reduced-capability distribution shift (PRD §32 #4): which satellites lose or limit their thrusters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from orbit_guard.physics.state import SatelliteCapability


def select_victims(
    num_satellites: int, pairs: Sequence[tuple[int, int]], disabled_count: int, reduced_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(disabled, reduced) satellite indices for the injected (primary, secondary) pairs, in pair order.

    PROVISIONAL (astro sign-off): disabled = injected-pair primaries first, then uninvolved satellites by
    index; reduced = injected-pair secondaries first, then the remaining uninvolved satellites by index. A
    secondary is never disabled, so every injected pair keeps at least one manoeuvrable member and the shift
    tests compensation (PRD §32 #4) rather than forcing the encounter's outcome. When too few eligible
    satellites exist (N - #pairs < disabled_count, never at N=15), fewer are disabled; pairs beyond the counts
    keep full capability.
    """
    involved = {index for pair in pairs for index in pair}
    uninvolved = [i for i in range(num_satellites) if i not in involved]
    disabled = tuple(([primary for primary, _ in pairs] + uninvolved)[:disabled_count])
    reduced_order = [secondary for _, secondary in pairs] + [i for i in uninvolved if i not in disabled]
    return disabled, tuple(reduced_order[:reduced_count])


@dataclass(frozen=True)
class CapabilityDegradation:
    """Reduced-capability distribution shift (PRD §32 #4), deterministic given the scenario.

    `select_victims` picks the satellites from the injected conjunctions alone, so every model faces the
    identical handicap: the first `disabled_count` get `maneuver_enabled=False`, the next `reduced_count`
    keep it but start with `reduced_budget_scale` of their drawn delta-v budget. A primary loses its
    thruster while its secondary keeps a reduced one, so no injected encounter is uncontrollable.
    """

    disabled_count: int = 0
    reduced_count: int = 0
    reduced_budget_scale: float = 0.25

    def __post_init__(self):
        if self.disabled_count < 0 or self.reduced_count < 0:
            raise ValueError("disabled_count and reduced_count must be >= 0")
        if not 0.0 < self.reduced_budget_scale <= 1.0:
            raise ValueError("reduced_budget_scale must be in (0, 1]")

    def apply(self, capabilities: list[SatelliteCapability], conjunctions) -> list[SatelliteCapability]:
        pairs = [(spec.primary_index, spec.secondary_index) for spec in conjunctions]
        disabled, reduced = select_victims(len(capabilities), pairs, self.disabled_count, self.reduced_count)
        updated = list(capabilities)
        for index in disabled:
            updated[index] = replace(updated[index], maneuver_enabled=False)
        for index in reduced:
            capability = updated[index]
            updated[index] = capability.with_remaining_delta_v(
                capability.remaining_delta_v_mps * self.reduced_budget_scale
            )
        return updated
