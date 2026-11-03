"""Validation runs across shift conditions, and the explicit test-split condition check (PRD §33, §35).

The validation split may be run under any subset of the five conditions (the R3 power check) and reuses
the same validation seeds for each: it is validation, not test. The final-test split always runs the
five conditions of the frozen pool, and a pool without exactly those conditions is refused with a clear
error rather than a bare assert (which `python -O` strips).
"""

from __future__ import annotations

import pytest

from orbit_guard.evaluation import experiment
from orbit_guard.evaluation.experiment import (
    require_all_conditions,
    validation_condition_seeds,
)
from orbit_guard.evaluation.shifts import CONDITION_NAMES


def _captured_run(tmp_path, monkeypatch, *extra: str) -> list:
    registry = tmp_path / "reg.json"
    registry.write_text("{}")
    captured: list = []
    monkeypatch.setattr(experiment, "run_tasks", lambda tasks, out, **kwargs: captured.extend(tasks) or [])
    monkeypatch.setattr(experiment, "read_records", lambda path: [])
    experiment.main(
        [
            "run", "--split", "validation", "--max-seeds", "3", "--models", "no_maneuver",
            "--registry", str(registry), "--out", str(tmp_path / "e.jsonl"), *extra,
        ]
    )
    return captured


def test_validation_defaults_to_the_normal_condition_only(tmp_path, monkeypatch):
    tasks = _captured_run(tmp_path, monkeypatch)

    assert {task.condition for task in tasks} == {"normal"}
    assert len(tasks) == 3


def test_validation_with_all_conditions_reuses_the_same_seeds_for_each(tmp_path, monkeypatch):
    tasks = _captured_run(tmp_path, monkeypatch, "--conditions", "all")

    per_condition = {
        condition: sorted(task.scenario_seed for task in tasks if task.condition == condition)
        for condition in CONDITION_NAMES
    }
    assert len(tasks) == 3 * len(CONDITION_NAMES)
    assert len({tuple(seeds) for seeds in per_condition.values()}) == 1


def test_validation_accepts_a_named_subset_of_conditions(tmp_path, monkeypatch):
    tasks = _captured_run(tmp_path, monkeypatch, "--conditions", "noise_increase", "normal")

    assert {task.condition for task in tasks} == {"normal", "noise_increase"}


def test_an_unknown_condition_name_is_rejected_by_the_cli(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _captured_run(tmp_path, monkeypatch, "--conditions", "solar_storm")


def test_conditions_cannot_be_chosen_for_the_test_split(tmp_path, capsys):
    registry = tmp_path / "reg.json"
    registry.write_text("{}")

    with pytest.raises(SystemExit):
        experiment.main(
            [
                "run", "--split", "test", "--freeze-snapshot", str(tmp_path / "frozen.json"),
                "--registry", str(registry), "--out", str(tmp_path / "e.jsonl"), "--conditions", "all",
            ]
        )

    assert "--conditions" in capsys.readouterr().err


def test_condition_seeds_follow_the_canonical_condition_order_without_duplicates():
    seeds = (5, 6)

    result = validation_condition_seeds(seeds, ("capability_reduction", "normal", "normal"))

    assert list(result) == ["normal", "capability_reduction"]
    assert all(value == seeds for value in result.values())


def test_all_expands_to_the_five_conditions():
    assert tuple(validation_condition_seeds((1,), ("all",))) == CONDITION_NAMES


def test_an_unknown_condition_raises_a_value_error():
    with pytest.raises(ValueError, match="solar_storm"):
        validation_condition_seeds((1,), ("solar_storm",))


def test_an_empty_condition_list_raises_a_value_error():
    with pytest.raises(ValueError, match="at least one"):
        validation_condition_seeds((1,), ())


def test_a_test_pool_missing_a_condition_is_refused_with_a_clear_error():
    pool = {name: (1,) for name in CONDITION_NAMES if name != "density_increase"}

    with pytest.raises(ValueError, match="density_increase"):
        require_all_conditions(pool)


def test_a_test_pool_in_another_order_is_refused():
    pool = {name: (1,) for name in reversed(CONDITION_NAMES)}

    with pytest.raises(ValueError, match="conditions"):
        require_all_conditions(pool)


def test_the_complete_test_pool_passes_the_condition_check():
    require_all_conditions({name: (1,) for name in CONDITION_NAMES})
