"""`ReplaySource`: the one interface the replay UI talks to (handoff prompt, "ReplaySource layer").

Wraps a recorded `EpisodeTrace`; every derived value is a pure function over its arrays, computed once on first
use. Nothing here runs ML, so a live feed could implement the same interface later.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property, wraps
from typing import Any

import numpy as np

from orbit_guard.safety.maneuver_mapper import IntentSpec, intent_catalog
from orbit_guard.visualization.derived.capability import Capability, capability_flags
from orbit_guard.visualization.derived.density import event_density
from orbit_guard.visualization.derived.fuel import FleetFuel, fleet_fuel
from orbit_guard.visualization.derived.key_steps import key_steps
from orbit_guard.visualization.derived.metrics import RunningMetrics, running_metrics
from orbit_guard.visualization.derived.partners import ClosestPartner, closest_partner
from orbit_guard.visualization.derived.runs import Run, event_runs
from orbit_guard.visualization.derived.start_step import default_start_step
from orbit_guard.visualization.derived.status import (
    pair_status_matrix,
    satellite_event_matrix,
    satellite_status_matrix,
)
from orbit_guard.visualization.replay import EpisodeTrace, Frame, frame_at
from orbit_guard.visualization.replay_meta import (
    ReplayMeta,
    ReplayMetaError,
    validate_replay_meta,
)
from orbit_guard.visualization.summary import SummaryTotals, summary_totals


def _freeze(value: Any) -> Any:
    """Make every array reachable from a cached derived value read-only: consumers share these, none may write."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze(item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _freeze(getattr(value, field.name))
    return value


def _frozen_cached(func):
    @cached_property
    @wraps(func)
    def wrapper(self):
        return _freeze(func(self))

    return wrapper


@dataclass(frozen=True)
class FuelAtStep:
    total_mps: float
    remaining_mps: np.ndarray  # (N,)
    empty_count: int
    usable_count: int


@dataclass(frozen=True)
class PartnerAtStep:
    partner: int
    d_min_m: float
    d_safe_m: float


class ReplaySource:
    def __init__(self, trace: EpisodeTrace):
        self.trace = trace
        self.meta: ReplayMeta = validate_replay_meta(trace.meta)
        agents = trace.num_agents
        if self.meta.n_agents != agents or trace.num_pairs != agents * (agents - 1) // 2:
            raise ReplayMetaError(f"meta n_agents={self.meta.n_agents} does not match the trace ({agents} satellites, {trace.num_pairs} pairs)")

    @property
    def num_steps(self) -> int:
        return self.trace.num_steps

    @property
    def num_agents(self) -> int:
        return self.trace.num_agents

    @property
    def num_pairs(self) -> int:
        return self.trace.num_pairs

    def frame_at(self, t: int) -> Frame:
        return frame_at(self.trace, t)

    @_frozen_cached
    def pair_status(self) -> np.ndarray:
        return pair_status_matrix(self.trace)

    @_frozen_cached
    def sat_status(self) -> np.ndarray:
        return satellite_status_matrix(self.trace, self.pair_status)

    @_frozen_cached
    def events(self) -> np.ndarray:
        return satellite_event_matrix(self.trace)

    @_frozen_cached
    def density(self) -> dict[int, np.ndarray]:
        return event_density(self.events)

    @_frozen_cached
    def runs(self) -> dict[int, list[Run]]:
        return event_runs(self.events, self.trace.pair_index, self.trace.collided_pair)

    @_frozen_cached
    def key_steps(self) -> list[int]:
        return key_steps(self.pair_status, self.trace.remaining_dv_mps, self.runs)

    @_frozen_cached
    def fuel(self) -> FleetFuel:
        return fleet_fuel(self.trace)

    @_frozen_cached
    def partners(self) -> ClosestPartner:
        return closest_partner(self.trace, self.pair_status)

    @_frozen_cached
    def running(self) -> RunningMetrics:
        return running_metrics(self.trace, shield_enabled=self.meta.shield)

    @_frozen_cached
    def capability(self) -> Capability:
        return capability_flags(self.trace, self.meta.delta_v_budget_mps)

    @_frozen_cached
    def start_step(self) -> int:
        return default_start_step(self.trace, self.pair_status, self.key_steps)

    @_frozen_cached
    def catalog(self) -> tuple[IntentSpec, ...]:
        return intent_catalog(self.meta.max_delta_v_per_action_mps)

    @_frozen_cached
    def summary(self) -> SummaryTotals:
        return summary_totals(self.trace.summary, self.meta)

    def _check_step(self, t: int) -> None:
        if not 0 <= t <= self.num_steps:
            raise IndexError(f"step {t} outside 0..{self.num_steps}")

    def fleet_fuel(self, t: int) -> FuelAtStep:
        self._check_step(t)
        fuel = self.fuel
        return FuelAtStep(
            total_mps=float(fuel.total_mps[t]),
            remaining_mps=self.trace.remaining_dv_mps[t].copy(),
            empty_count=int(fuel.empty_count[t]),
            usable_count=fuel.usable_count,
        )

    def closest_partner(self, t: int, satellite: int) -> PartnerAtStep | None:
        self._check_step(t)
        partner = int(self.partners.partner[t, satellite])
        if partner < 0:
            return None
        return PartnerAtStep(
            partner=partner,
            d_min_m=float(self.partners.d_min_m[t, satellite]),
            d_safe_m=float(self.partners.d_safe_m[t, satellite]),
        )
