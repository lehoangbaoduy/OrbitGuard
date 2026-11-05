"""Which recorded episode to show: scenario x policy x shield (PR2; EC-03, EC-13, EC-14).

Pure functions over an `EpisodeIndex`, so every transition is testable without a browser. A combination with no
trace is never selected; changing the scenario keeps the policy and shield when the new scenario has them and
otherwise falls back deterministically (same policy with the other shield, then any recorded policy). A policy
recorded under one shield setting only (the do-nothing baseline is shield-off only) opens that setting when chosen.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from orbit_guard.visualization.episodes import EpisodeIndex


@dataclass(frozen=True)
class Selection:
    scenario_id: str | None
    policy: str | None
    shield: bool


def _preferences(index: EpisodeIndex, policy: str | None, shield: bool) -> Iterator[tuple[str, bool]]:
    if policy is not None:
        yield policy, shield
        yield policy, not shield
    for other in index.policies():
        yield other, shield
        yield other, not shield


def _first_available(index: EpisodeIndex, scenario_id: str, policy: str | None, shield: bool) -> Selection | None:
    for candidate_policy, candidate_shield in _preferences(index, policy, shield):
        if index.find(scenario_id, candidate_policy, candidate_shield) is not None:
            return Selection(scenario_id, candidate_policy, candidate_shield)
    return None


def initial_selection(index: EpisodeIndex) -> Selection:
    """First scenario, BEFORE training (the first policy), shield ON where recorded."""
    scenarios = index.scenarios()
    if not scenarios:
        return Selection(None, None, True)
    return _first_available(index, scenarios[0].scenario_id, None, True) or Selection(None, None, True)


def resolve_selection(
    index: EpisodeIndex,
    current: Selection,
    *,
    scenario_id: str | None = None,
    policy: str | None = None,
    shield: bool | None = None,
) -> Selection:
    """Apply one requested change; an unavailable request leaves the selection as it was."""
    if scenario_id is not None and scenario_id not in {s.scenario_id for s in index.scenarios()}:
        return current
    target = replace(
        current,
        scenario_id=current.scenario_id if scenario_id is None else scenario_id,
        policy=current.policy if policy is None else policy,
        shield=current.shield if shield is None else shield,
    )
    if target.scenario_id is None or index.find(target.scenario_id, target.policy, target.shield) is not None:
        return target
    if policy is not None and shield is None and scenario_id is None:  # a policy click: its other shield setting, if any
        other = replace(target, shield=not target.shield)
        return other if target.scenario_id is not None and index.find(other.scenario_id, other.policy, other.shield) else current
    if scenario_id is None:  # an explicit shield click on a missing trace is ignored (the control is disabled)
        return current
    return _first_available(index, target.scenario_id, current.policy, current.shield) or current
