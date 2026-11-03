"""`experiment run --step-logs/--no-step-logs`: per-step §40 logs on for the test split, off for validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit_guard.evaluation import experiment
from orbit_guard.evaluation.experiment import build_tasks, step_log_dir_for


def _captured_run(tmp_path, monkeypatch, split: str, *extra: str) -> list:
    registry = tmp_path / "reg.json"
    registry.write_text("{}")
    captured: list = []
    monkeypatch.setattr(experiment, "run_tasks", lambda tasks, out, **kwargs: captured.extend(tasks) or [])
    monkeypatch.setattr(experiment, "read_records", lambda path: [])
    # The final-test pool is guarded by the freeze (tested elsewhere); only the task wiring is under test here.
    monkeypatch.setattr(experiment, "_test_condition_seeds", lambda args, registry: {"normal": (11, 12)})
    split_args = ["--max-seeds", "2"] if split == "validation" else ["--freeze-snapshot", str(tmp_path / "f.json")]
    experiment.main(
        [
            "run", "--split", split, *split_args, "--models", "no_maneuver",
            "--registry", str(registry), "--out", str(tmp_path / "out" / "e.jsonl"), *extra,
        ]
    )
    return captured


@pytest.mark.parametrize(
    ("split", "flag", "expected_on"),
    [("test", None, True), ("validation", None, False), ("test", False, False), ("validation", True, True)],
)
def test_the_step_log_directory_defaults_by_split_and_can_be_overridden(split, flag, expected_on):
    directory = step_log_dir_for(split, "results/run1/episodes.jsonl", flag)

    assert directory == (str(Path("results/run1/steps")) if expected_on else None)


def test_the_test_split_logs_steps_next_to_the_episode_log_by_default(tmp_path, monkeypatch):
    tasks = _captured_run(tmp_path, monkeypatch, "test")

    assert tasks
    assert {task.step_log_dir for task in tasks} == {str(tmp_path / "out" / "steps")}


def test_the_validation_split_does_not_log_steps_by_default(tmp_path, monkeypatch):
    tasks = _captured_run(tmp_path, monkeypatch, "validation")

    assert tasks
    assert {task.step_log_dir for task in tasks} == {None}


def test_no_step_logs_turns_the_test_split_logging_off(tmp_path, monkeypatch):
    tasks = _captured_run(tmp_path, monkeypatch, "test", "--no-step-logs")

    assert {task.step_log_dir for task in tasks} == {None}


def test_step_logs_turns_validation_logging_on(tmp_path, monkeypatch):
    tasks = _captured_run(tmp_path, monkeypatch, "validation", "--step-logs")

    assert {task.step_log_dir for task in tasks} == {str(tmp_path / "out" / "steps")}


def test_build_tasks_leaves_step_logging_off_unless_asked():
    plain = build_tasks(["no_maneuver"], {"normal": (1,)}, {}, n_agents=3, experiment_id="x")
    logged = build_tasks(["no_maneuver"], {"normal": (1,)}, {}, n_agents=3, experiment_id="x", step_log_dir="d")

    assert [task.step_log_dir for task in plain] == [None]
    assert [task.step_log_dir for task in logged] == ["d"]
