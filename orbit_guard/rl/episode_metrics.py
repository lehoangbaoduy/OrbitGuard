"""Per-episode outcome metrics and failure-mode labels (PRD §34, §36, §40).

The tracker lives inside the environment so every model -- learned or
heuristic, shielded or not -- is scored by the same accounting. It uses ground
truth ONLY for the evaluation-side facts the PRD reserves it for (true minimum
pair separation, §14 point 5); everything else comes from the same belief
assessments the agents see.

Definitions:
  * collision pair        -- a pair whose true separation fell below the combined hard-body radius.
  * unsafe conjunction    -- a pair that did NOT collide but whose true minimum separation over the
                             episode was below the margin `d_safe` in force at that closest approach.
  * maneuver              -- one applied (non-zero, capability-approved) burn by one satellite.
  * residual risk         -- the largest belief risk proxy over all pairs after each step, averaged.
  * new conjunction / worsened -- after a step in which a pair member burned, a pair that was
                             outside `d_safe` is now inside it / its risk rose by more than `_WORSENING_RISK`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

_WORSENING_RISK = 0.05
_BURN_EPSILON_MPS = 1.0e-12

Pair = tuple[int, int]

LABEL_COLLISION = "physical_collision"
LABEL_UNSAFE_REMAINED = "unsafe_conjunction_remained"
LABEL_SHIELD_REJECTED = "unsafe_proposal_rejected_by_shield"
LABEL_MISSION_COSTLY = "feasible_proposal_mission_costly"
LABEL_BUDGET_EXHAUSTED = "maneuver_budget_exhausted"
LABEL_UNABLE_TO_MANEUVER = "satellite_unable_to_maneuver"
LABEL_NEW_CONJUNCTION = "new_conjunction_created"
LABEL_WORSENED = "existing_conjunction_worsened"
LABEL_SIMULATION_FAILURE = "numerical_simulation_failure"

_MISSION_COSTLY_DEVIATION = 0.5


@dataclass(frozen=True)
class EpisodeSummary:
    steps: int = 0
    collision_count: int = 0
    unsafe_conjunction_count: int = 0
    total_delta_v_mps: float = 0.0
    mean_remaining_delta_v_mps: float = 0.0
    maneuver_count: int = 0
    mean_mission_deviation: float = 0.0
    shield_intervention_count: int = 0
    shield_rejected_proposal_steps: int = 0
    shield_infeasible_steps: int = 0
    new_conjunction_steps: int = 0
    worsened_conjunction_steps: int = 0
    mean_residual_risk: float = 0.0
    failure_labels: tuple[str, ...] = ()
    max_mission_deviation: float = 0.0  # largest post-step deviation of any active satellite (§40)


@dataclass(frozen=True)
class ShieldStep:
    """What the shield did on one control step (None for environments without one)."""

    interventions: int
    proposal_accepted: bool
    approved_feasible: bool


@dataclass
class EpisodeMetricsTracker:
    """Mutable accumulator; `reset()` per episode, `record_step()` per control step."""

    steps: int = 0
    collided_pairs: set[Pair] = field(default_factory=set)
    _closest: dict[Pair, tuple[float, float]] = field(default_factory=dict)  # pair -> (min sep, d_safe then)
    _delta_v_total: float = 0.0
    _maneuvers: int = 0
    _deviation_sum: float = 0.0
    _deviation_samples: int = 0
    _deviation_max: float = 0.0
    _risk_sum: float = 0.0
    _shield_interventions: int = 0
    _shield_rejected: int = 0
    _shield_infeasible: int = 0
    _new_conjunction_steps: int = 0
    _worsened_steps: int = 0

    def reset(self) -> None:
        self.__init__()

    def record_step(
        self,
        *,
        applied_delta_v_mps: Sequence[float],
        collided: set[Pair],
        min_separation_m: Mapping[Pair, float],
        pre: Mapping[Pair, tuple[float, float, float]],
        post: Mapping[Pair, tuple[float, float, float]],
        mission_deviations: Sequence[float],
        shield: ShieldStep | None,
    ) -> None:
        """`pre`/`post` map pair -> (d_min_m, d_safe_m, risk) from the decision-time / post-step assessment."""
        self.steps += 1
        self.collided_pairs |= collided
        for pair, separation in min_separation_m.items():
            d_safe = pre[pair][1]
            if pair not in self._closest or separation < self._closest[pair][0]:
                self._closest[pair] = (separation, d_safe)

        burned = [dv > _BURN_EPSILON_MPS for dv in applied_delta_v_mps]
        self._delta_v_total += float(sum(applied_delta_v_mps))
        self._maneuvers += sum(burned)
        self._deviation_sum += float(sum(mission_deviations))
        self._deviation_samples += len(mission_deviations)
        self._deviation_max = max(self._deviation_max, *map(float, mission_deviations), 0.0)
        self._risk_sum += max((risk for _, _, risk in post.values()), default=0.0)

        new_conjunction = worsened = False
        for (i, j), (pre_d_min, pre_d_safe, pre_risk) in pre.items():
            if not (burned[i] or burned[j]):
                continue
            post_d_min, post_d_safe, post_risk = post[(i, j)]
            new_conjunction |= pre_d_min >= pre_d_safe and post_d_min < post_d_safe
            worsened |= post_risk > pre_risk + _WORSENING_RISK
        self._new_conjunction_steps += new_conjunction
        self._worsened_steps += worsened

        if shield is not None:
            self._shield_interventions += shield.interventions
            self._shield_rejected += not shield.proposal_accepted
            self._shield_infeasible += not shield.approved_feasible

    def summary(
        self,
        *,
        remaining_delta_v_mps: Sequence[float],
        smallest_burn_mps: Sequence[float],
        maneuver_enabled: Sequence[bool],
    ) -> EpisodeSummary:
        unsafe_pairs = {
            pair
            for pair, (separation, d_safe) in self._closest.items()
            if pair not in self.collided_pairs and separation < d_safe
        }
        involved = {agent for pair in unsafe_pairs | self.collided_pairs for agent in pair}
        mean_deviation = self._deviation_sum / self._deviation_samples if self._deviation_samples else 0.0

        labels = []
        if self.collided_pairs:
            labels.append(LABEL_COLLISION)
        if unsafe_pairs:
            labels.append(LABEL_UNSAFE_REMAINED)
        if self._shield_rejected:
            labels.append(LABEL_SHIELD_REJECTED)
        if self._maneuvers and mean_deviation > _MISSION_COSTLY_DEVIATION:
            labels.append(LABEL_MISSION_COSTLY)
        if any(remaining < smallest for remaining, smallest in zip(remaining_delta_v_mps, smallest_burn_mps, strict=True)):
            labels.append(LABEL_BUDGET_EXHAUSTED)
        if any(not enabled and agent in involved for agent, enabled in enumerate(maneuver_enabled)):
            labels.append(LABEL_UNABLE_TO_MANEUVER)
        if self._new_conjunction_steps:
            labels.append(LABEL_NEW_CONJUNCTION)
        if self._worsened_steps:
            labels.append(LABEL_WORSENED)

        return EpisodeSummary(
            steps=self.steps,
            collision_count=len(self.collided_pairs),
            unsafe_conjunction_count=len(unsafe_pairs),
            total_delta_v_mps=self._delta_v_total,
            mean_remaining_delta_v_mps=float(sum(remaining_delta_v_mps) / max(len(remaining_delta_v_mps), 1)),
            maneuver_count=self._maneuvers,
            mean_mission_deviation=mean_deviation,
            shield_intervention_count=self._shield_interventions,
            shield_rejected_proposal_steps=self._shield_rejected,
            shield_infeasible_steps=self._shield_infeasible,
            new_conjunction_steps=self._new_conjunction_steps,
            worsened_conjunction_steps=self._worsened_steps,
            mean_residual_risk=self._risk_sum / self.steps if self.steps else 0.0,
            failure_labels=tuple(labels),
            max_mission_deviation=self._deviation_max,
        )
