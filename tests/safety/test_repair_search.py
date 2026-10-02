"""Tests for orbit_guard.safety.repair_search (PRD §29 step 6, P4-M3).

The search is pure: it knows nothing about physics, only an `evaluate`
callable mapping a full joint assignment (one intent per satellite index) to
an `AssignmentScore`. Fake evaluators therefore pin the search's own contract
-- exact-vs-beam threshold, beam width, determinism, tie-breaking -- without
any propagation.
"""

from __future__ import annotations

import itertools

import pytest

from orbit_guard.safety.repair_search import (
    AssignmentScore,
    RepairOutcome,
    search_repair,
)

_ALL_INTENTS = tuple(range(6))


def _candidates(agents):
    return {agent: _ALL_INTENTS for agent in agents}


def _no_op_proposal(num_agents):
    return (0,) * num_agents


def _separable_risk(target):
    """Risk = number of affected agents NOT on their target intent; always feasible."""

    def evaluate(assignment):
        misses = sum(1 for agent, intent in target.items() if assignment[agent] != intent)
        dv = sum(0.1 for intent in assignment if intent != 0)
        return AssignmentScore(feasible=True, risk=float(misses), delta_v_mps=dv)

    return evaluate


def test_four_affected_agents_use_exact_enumeration_of_all_6_to_the_4_combinations():
    evaluate_calls = []
    target = {0: 1, 1: 2, 2: 3, 3: 4}
    base = _separable_risk(target)

    def counting_evaluate(assignment):
        evaluate_calls.append(assignment)
        return base(assignment)

    outcome = search_repair(_no_op_proposal(6), (0, 1, 2, 3), _candidates(range(4)), counting_evaluate)

    assert outcome.stats.mode == "exact"
    assert outcome.stats.evaluated == 6**4 == len(evaluate_calls)
    assert len(set(evaluate_calls)) == 6**4
    assert outcome.stats.max_beam_size == 0


def test_five_affected_agents_switch_to_beam_search():
    target = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

    outcome = search_repair(_no_op_proposal(8), (0, 1, 2, 3, 4), _candidates(range(5)), _separable_risk(target))

    assert outcome.stats.mode == "beam"
    assert outcome.stats.evaluated < 6**5


def test_beam_keeps_exactly_fifty_partial_assignments():
    target = {agent: 1 for agent in range(6)}

    outcome = search_repair(_no_op_proposal(8), tuple(range(6)), _candidates(range(6)), _separable_risk(target))

    assert outcome.stats.mode == "beam"
    assert outcome.stats.max_beam_size == 50


def test_beam_width_is_configurable_but_defaults_to_fifty():
    target = {agent: 1 for agent in range(6)}

    narrow = search_repair(
        _no_op_proposal(6), tuple(range(6)), _candidates(range(6)), _separable_risk(target), beam_width=7
    )

    assert narrow.stats.max_beam_size == 7


def test_exact_threshold_is_configurable():
    target = {0: 1, 1: 2, 2: 3}

    outcome = search_repair(
        _no_op_proposal(3), (0, 1, 2), _candidates(range(3)), _separable_risk(target), exact_max_agents=2
    )

    assert outcome.stats.mode == "beam"


def test_exact_search_finds_the_global_optimum_of_a_coupled_objective():
    # Risk couples agents: agents 0 and 1 must pick DIFFERENT non-zero intents.
    def evaluate(assignment):
        clash = assignment[0] == assignment[1]
        risk = 10.0 if clash else 0.0
        risk += 0.5 * (assignment[2] != 5)
        return AssignmentScore(feasible=True, risk=risk, delta_v_mps=0.01 * sum(1 for i in assignment if i))

    outcome = search_repair((1, 1, 0), (0, 1, 2), _candidates(range(3)), evaluate)

    assert outcome.assignment[0] != outcome.assignment[1]
    assert outcome.assignment[2] == 5
    brute = min(
        itertools.product(_ALL_INTENTS, repeat=3),
        key=lambda a: (evaluate(a).risk, evaluate(a).delta_v_mps),
    )
    assert evaluate(outcome.assignment).risk == evaluate(brute).risk


def test_beam_search_finds_the_optimum_of_a_separable_objective():
    target = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1}

    outcome = search_repair(_no_op_proposal(6), tuple(range(6)), _candidates(range(6)), _separable_risk(target))

    assert dict(enumerate(outcome.assignment)) == target
    assert outcome.score.risk == 0.0


def test_uninvolved_agents_keep_their_proposed_actions():
    proposal = (0, 3, 0, 4, 2)
    target = {1: 1, 3: 2}

    outcome = search_repair(proposal, (1, 3), _candidates((1, 3)), _separable_risk(target))

    assert outcome.assignment[0] == 0
    assert outcome.assignment[2] == 0
    assert outcome.assignment[4] == 2
    assert (outcome.assignment[1], outcome.assignment[3]) == (1, 2)


def test_result_is_identical_across_repeated_runs():
    target = {agent: (agent % 5) + 1 for agent in range(6)}
    runs = [
        search_repair(_no_op_proposal(6), tuple(range(6)), _candidates(range(6)), _separable_risk(target))
        for _ in range(3)
    ]

    assert runs[0] == runs[1] == runs[2]


def test_result_does_not_depend_on_the_order_affected_agents_are_listed_exactly():
    target = {0: 1, 1: 2, 2: 3}
    forward = search_repair(_no_op_proposal(3), (0, 1, 2), _candidates(range(3)), _separable_risk(target))
    backward = search_repair(_no_op_proposal(3), (2, 1, 0), _candidates(range(3)), _separable_risk(target))

    assert forward.assignment == backward.assignment


def test_equal_risk_ties_break_to_lower_total_delta_v():
    def evaluate(assignment):
        # Every option has identical risk; intent 2 is the cheap one.
        dv = {0: 0.3, 1: 0.2, 2: 0.05}.get(assignment[0], 0.4)
        return AssignmentScore(feasible=True, risk=1.0, delta_v_mps=dv)

    outcome = search_repair((1,), (0,), {0: (1, 2)}, evaluate)

    assert outcome.assignment == (2,)


def test_equal_risk_and_delta_v_ties_break_to_the_lowest_satellite_index():
    # Exactly one of the two agents must maneuver (any intent 1); either choice is equivalent.
    def evaluate(assignment):
        moving = [agent for agent, intent in enumerate(assignment) if intent != 0]
        risk = 0.0 if len(moving) == 1 else 5.0
        return AssignmentScore(feasible=True, risk=risk, delta_v_mps=0.1 * len(moving))

    outcome = search_repair((0, 0), (0, 1), {0: (0, 1), 1: (0, 1)}, evaluate)

    assert outcome.assignment == (1, 0)


def test_risk_differences_below_tolerance_are_treated_as_ties():
    def evaluate(assignment):
        intent = assignment[0]
        return AssignmentScore(feasible=True, risk=1e-6 * intent, delta_v_mps=0.1 if intent else 0.0)

    outcome = search_repair((1,), (0,), {0: (0, 1, 2, 3)}, evaluate, risk_tolerance=1e-3)

    assert outcome.assignment == (0,)


def test_a_feasible_assignment_beats_a_lower_risk_infeasible_one():
    def evaluate(assignment):
        if assignment[0] == 1:
            return AssignmentScore(feasible=False, risk=0.0, delta_v_mps=0.1)
        return AssignmentScore(feasible=True, risk=0.9, delta_v_mps=0.1)

    outcome = search_repair((0,), (0,), {0: (1, 2)}, evaluate)

    assert outcome.assignment == (2,)
    assert outcome.score.feasible


def test_with_no_feasible_assignment_the_least_risky_is_returned_and_flagged_infeasible():
    def evaluate(assignment):
        return AssignmentScore(feasible=False, risk=0.1 * assignment[0], delta_v_mps=0.1)

    outcome = search_repair((0,), (0,), {0: (2, 3, 4)}, evaluate)

    assert outcome.assignment == (2,)
    assert not outcome.score.feasible


def test_no_maneuver_is_chosen_only_when_it_is_actually_the_safest_option():
    safest_is_burn = {0: 3, 1: 4}

    def evaluate(assignment):
        misses = sum(1 for agent, intent in safest_is_burn.items() if assignment[agent] != intent)
        return AssignmentScore(feasible=True, risk=float(misses), delta_v_mps=0.1 * sum(1 for i in assignment if i))

    burn = search_repair((0, 0), (0, 1), _candidates((0, 1)), evaluate)
    assert burn.assignment == (3, 4)

    def no_op_is_safest(assignment):
        return AssignmentScore(feasible=True, risk=float(sum(1 for i in assignment if i)), delta_v_mps=0.0)

    safest = search_repair((1, 1), (0, 1), _candidates((0, 1)), no_op_is_safest)
    assert safest.assignment == (0, 0)


def test_empty_affected_set_is_rejected():
    with pytest.raises(ValueError):
        search_repair((0, 0), (), {}, lambda a: AssignmentScore(True, 0.0, 0.0))


def test_affected_agent_without_candidates_is_rejected():
    with pytest.raises(ValueError):
        search_repair((0, 0), (0,), {0: ()}, lambda a: AssignmentScore(True, 0.0, 0.0))


def test_outcome_reports_the_affected_agents_in_search_order():
    outcome = search_repair((0, 0, 0), (2, 0), _candidates((0, 2)), _separable_risk({0: 1, 2: 1}))

    assert isinstance(outcome, RepairOutcome)
    assert set(outcome.stats.affected_agents) == {0, 2}
