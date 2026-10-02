"""Fleet fuel: remaining delta-v per satellite over the episode (EC-35)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbit_guard.visualization.replay import EpisodeTrace

EMPTY_EPS_MPS = 1e-9


@dataclass(frozen=True)
class FleetFuel:
    budget_mps: np.ndarray  # (N,) each satellite's own starting tank (a limited satellite has a shorter one)
    total_mps: np.ndarray  # (T+1,) delta-v left in the USABLE tanks (an offline thruster cannot spend its fuel)
    total_budget_mps: float  # starting delta-v of the usable tanks
    stranded_mps: float  # fuel held by offline thrusters, shown separately so the fleet total never overstates
    empty: np.ndarray  # (T+1, N) usable satellites whose tank is empty
    empty_count: np.ndarray  # (T+1,) number of empty usable tanks
    usable_count: int  # satellites whose thrusters work (offline ones hold fuel they can never use)
    all_empty_step: int | None  # first step at which every usable tank is empty


def fleet_fuel(trace: EpisodeTrace) -> FleetFuel:
    remaining = trace.remaining_dv_mps
    usable = np.asarray(trace.maneuver_enabled, dtype=bool)
    empty = (remaining <= EMPTY_EPS_MPS) & usable[None, :]
    empty_count = empty.sum(axis=1).astype(np.int32)
    usable_count = int(usable.sum())
    full = np.nonzero(empty_count == usable_count)[0] if usable_count else np.array([], dtype=int)
    return FleetFuel(
        budget_mps=remaining[0].copy(),
        total_mps=remaining[:, usable].sum(axis=1),
        total_budget_mps=float(remaining[0][usable].sum()),
        stranded_mps=float(remaining[0][~usable].sum()),
        empty=empty,
        empty_count=empty_count,
        usable_count=usable_count,
        all_empty_step=int(full[0]) if full.size else None,
    )
