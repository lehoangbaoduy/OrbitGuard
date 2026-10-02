"""Bounded joint-action repair search (PRD §29 step 6, P4-M3).

Pure search over a joint assignment (one maneuver intent per satellite
index); it knows nothing about physics -- the caller supplies `evaluate`, a
deterministic callable scoring a full assignment. That separation is what lets
the exact-vs-beam threshold, the beam width, and the tie-breaking be pinned by
tests with fake evaluators.

  * `<= exact_max_agents` affected agents (default 4): exact enumeration of
    every combination (6^4 = 1296 at most).
  * more: beam search, one agent per expansion level, keeping the top
    `beam_width` (default 50) PARTIAL assignments by predicted total risk;
    agents not yet expanded sit at their proposed action while a partial
    assignment is scored.
  * agents outside `affected` keep their proposed action (see the shield's
    module docstring for why this reads PRD §29's "leave uninvolved agents at
    NO_MANEUVER" as "do not search them" rather than "cancel their burn").

Selection among FULL candidates is lexicographic and fully deterministic:
  1. a feasible assignment beats any infeasible one;
  2. lower total predicted risk (compared after quantizing to `risk_tolerance`,
     so risk differences too small to matter fall through to the next rule
     instead of buying a pointless extra burn);
  3. lower `shortfall_m` -- how far inside the safety margin the candidate
     still leaves its pairs. It is 0 for every feasible assignment, so it only
     matters when nothing is feasible: the risk proxy saturates (e.g. near 0
     when sensing noise is tiny), and without this rule the fallback would
     fall through to "burn nothing" while a burn could still add separation;
  4. lower total delta-v;
  5. the maneuvering satellites with the lowest indices (the sorted tuple of
     indices whose intent is not NO_MANEUVER, compared lexicographically);
  6. the assignment tuple itself (a total order, so nothing is left to
     iteration order).
If no candidate is feasible the same ordering picks the least risky one and
the outcome's score says `feasible=False` -- the caller decides how to report.
`NO_MANEUVER` is never a fallback: it wins only where it ranks best.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

NO_MANEUVER_INTENT = 0
DEFAULT_EXACT_MAX_AGENTS = 4
DEFAULT_BEAM_WIDTH = 50
DEFAULT_RISK_TOLERANCE = 1.0e-3
_DELTA_V_RESOLUTION_MPS = 1.0e-9
_SHORTFALL_RESOLUTION_M = 1.0

Assignment = tuple[int, ...]


@dataclass(frozen=True)
class AssignmentScore:
    """Predicted quality of one full joint assignment."""

    feasible: bool
    risk: float
    delta_v_mps: float
    shortfall_m: float = 0.0


@dataclass(frozen=True)
class RepairStats:
    mode: Literal["exact", "beam"]
    affected_agents: tuple[int, ...]
    evaluated: int
    max_beam_size: int  # 0 for exact enumeration


@dataclass(frozen=True)
class RepairOutcome:
    assignment: Assignment
    score: AssignmentScore
    stats: RepairStats


Evaluator = Callable[[Assignment], AssignmentScore]


def _quantized(value: float, resolution: float) -> int:
    return round(value / resolution)


def _moving_agents(assignment: Assignment) -> tuple[int, ...]:
    return tuple(agent for agent, intent in enumerate(assignment) if intent != NO_MANEUVER_INTENT)


def _final_key(score: AssignmentScore, assignment: Assignment, risk_tolerance: float):
    return (
        not score.feasible,
        _quantized(score.risk, risk_tolerance),
        _quantized(score.shortfall_m, _SHORTFALL_RESOLUTION_M),
        _quantized(score.delta_v_mps, _DELTA_V_RESOLUTION_MPS),
        _moving_agents(assignment),
        assignment,
    )


def _partial_key(score: AssignmentScore, assignment: Assignment, risk_tolerance: float):
    return (
        _quantized(score.risk, risk_tolerance),
        _quantized(score.shortfall_m, _SHORTFALL_RESOLUTION_M),
        _quantized(score.delta_v_mps, _DELTA_V_RESOLUTION_MPS),
        _moving_agents(assignment),
        assignment,
    )


def _with_intents(base: Assignment, agents: Sequence[int], intents: Sequence[int]) -> Assignment:
    updated = list(base)
    for agent, intent in zip(agents, intents, strict=True):
        updated[agent] = intent
    return tuple(updated)


def search_repair(
    proposal: Assignment,
    affected: Sequence[int],
    candidate_intents: Mapping[int, Sequence[int]],
    evaluate: Evaluator,
    *,
    order: Sequence[int] | None = None,
    exact_max_agents: int = DEFAULT_EXACT_MAX_AGENTS,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    risk_tolerance: float = DEFAULT_RISK_TOLERANCE,
) -> RepairOutcome:
    """Search the affected agents' alternatives for the best joint assignment.

    `order` fixes the beam's expansion order (highest-priority conjunction
    first, PRD §26); it defaults to ascending satellite index and has no effect
    on exact enumeration's result.
    """
    affected_set = set(affected)
    if not affected_set:
        raise ValueError("search_repair needs at least one affected agent")
    for agent in affected_set:
        if not candidate_intents.get(agent):
            raise ValueError(f"affected agent {agent} has no candidate intents")
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    expansion_order = tuple(order) if order is not None else tuple(sorted(affected_set))
    if set(expansion_order) != affected_set:
        raise ValueError("order must list exactly the affected agents")

    if len(affected_set) <= exact_max_agents:
        return _exact(proposal, expansion_order, candidate_intents, evaluate, risk_tolerance)
    return _beam(proposal, expansion_order, candidate_intents, evaluate, beam_width, risk_tolerance)


def _exact(proposal, agents, candidate_intents, evaluate, risk_tolerance) -> RepairOutcome:
    best: tuple | None = None
    evaluated = 0
    for intents in itertools.product(*(sorted(set(candidate_intents[agent])) for agent in agents)):
        assignment = _with_intents(proposal, agents, intents)
        score = evaluate(assignment)
        evaluated += 1
        key = _final_key(score, assignment, risk_tolerance)
        if best is None or key < best[0]:
            best = (key, assignment, score)
    assert best is not None  # candidate lists are non-empty, so the product is too
    return RepairOutcome(
        assignment=best[1],
        score=best[2],
        stats=RepairStats("exact", tuple(agents), evaluated, 0),
    )


def _beam(proposal, agents, candidate_intents, evaluate, beam_width, risk_tolerance) -> RepairOutcome:
    beam: list[tuple[Assignment, AssignmentScore]] = [(proposal, evaluate(proposal))]
    evaluated = 1
    max_beam_size = 0
    for agent in agents:
        expanded: dict[Assignment, AssignmentScore] = {}
        for assignment, _ in beam:
            for intent in sorted(set(candidate_intents[agent])):
                child = _with_intents(assignment, (agent,), (intent,))
                if child not in expanded:
                    expanded[child] = evaluate(child)
                    evaluated += 1
        ranked = sorted(expanded.items(), key=lambda item: _partial_key(item[1], item[0], risk_tolerance))
        beam = ranked[:beam_width]
        max_beam_size = max(max_beam_size, len(beam))

    best_assignment, best_score = min(beam, key=lambda item: _final_key(item[1], item[0], risk_tolerance))
    return RepairOutcome(
        assignment=best_assignment,
        score=best_score,
        stats=RepairStats("beam", tuple(agents), evaluated, max_beam_size),
    )
