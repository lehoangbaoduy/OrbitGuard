"""Selection of the episode to show (scenario x policy x shield): a pure function, so every transition is testable
without a browser (PR2; EC-03, EC-13, EC-14). Unavailable combinations are never selected."""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit_guard.visualization.episodes import EpisodeEntry, EpisodeIndex
from orbit_guard.visualization.replay_meta import (
    build_replay_meta,
    validate_replay_meta,
)
from orbit_guard.visualization.selection import (
    Selection,
    initial_selection,
    resolve_selection,
)


def _entry(condition: str, seed: int, policy: str, shield: bool) -> EpisodeEntry:
    extra = {"weights_sha256": "a" * 64} if policy == "trained" else {"init_seed": 0} if policy == "untrained" else {}
    meta = build_replay_meta(condition=condition, seed=seed, policy=policy, shield=shield, model="m", n_agents=15,
                             delta_v_budget_mps=2.0, max_delta_v_per_action_mps=0.1, **extra)
    return EpisodeEntry(Path(f"{condition}-{seed}-{policy}-{shield}.npz"), validate_replay_meta(meta))


def _index(*combos) -> EpisodeIndex:
    return EpisodeIndex(tuple(_entry(*combo) for combo in combos))


def _full(condition: str, seed: int):
    return [(condition, seed, policy, shield) for policy in ("untrained", "trained") for shield in (True, False)]


def test_initial_selection_is_the_first_scenario_before_training_with_the_shield_on():
    index = _index(*_full("simultaneous_conjunctions", 10), *_full("normal", 5))
    assert initial_selection(index) == Selection("normal:5", "untrained", True)


def test_initial_selection_falls_back_when_shield_on_is_not_recorded():
    index = _index(("normal", 5, "trained", False))
    assert initial_selection(index) == Selection("normal:5", "trained", False)


def test_empty_index_gives_an_empty_selection():
    assert initial_selection(_index()) == Selection(None, None, True)


def test_switching_policy_keeps_scenario_and_shield():
    index = _index(*_full("normal", 5))
    current = Selection("normal:5", "untrained", False)
    assert resolve_selection(index, current, policy="trained") == Selection("normal:5", "trained", False)


def test_switching_shield_keeps_scenario_and_policy():
    index = _index(*_full("normal", 5))
    current = Selection("normal:5", "trained", True)
    assert resolve_selection(index, current, shield=False) == Selection("normal:5", "trained", False)


def test_switching_scenario_keeps_policy_and_shield_when_available():
    index = _index(*_full("normal", 5), *_full("capability_reduction", 10))
    current = Selection("normal:5", "trained", False)
    assert resolve_selection(index, current, scenario_id="capability_reduction:10") == Selection(
        "capability_reduction:10", "trained", False
    )


def test_scenario_change_falls_back_to_same_policy_other_shield_first():
    index = _index(*_full("normal", 5), ("capability_reduction", 10, "trained", True), ("capability_reduction", 10, "untrained", False))
    current = Selection("normal:5", "trained", False)
    assert resolve_selection(index, current, scenario_id="capability_reduction:10") == Selection(
        "capability_reduction:10", "trained", True
    )


def test_scenario_change_then_other_policy_when_the_policy_is_missing():
    index = _index(*_full("normal", 5), ("capability_reduction", 10, "untrained", False))
    current = Selection("normal:5", "trained", False)
    assert resolve_selection(index, current, scenario_id="capability_reduction:10") == Selection(
        "capability_reduction:10", "untrained", False
    )


def test_a_request_for_an_unavailable_combination_is_ignored():
    index = _index(("normal", 5, "untrained", True), ("normal", 5, "trained", True))
    current = Selection("normal:5", "trained", True)
    assert resolve_selection(index, current, shield=False) == current  # no shield-off trace exists


def test_unknown_scenario_is_ignored():
    index = _index(*_full("normal", 5))
    current = Selection("normal:5", "trained", True)
    assert resolve_selection(index, current, scenario_id="nope:1") == current


def test_no_change_requested_returns_the_current_selection():
    index = _index(*_full("normal", 5))
    current = Selection("normal:5", "untrained", True)
    assert resolve_selection(index, current) == current


def test_selection_is_immutable():
    with pytest.raises(AttributeError):
        Selection("a", "b", True).shield = False  # frozen dataclass
