"""The episode log survives a killed sweep, and failed episodes can be retried (PRD §40; Acceptance F).

A sweep killed mid-write leaves a torn LAST line. Reading tolerates it with a warning, and the next
run_tasks cuts it off before appending, so it never becomes a corrupt mid-file line. An invalid line
in the middle of the file is real corruption and raises. When a key is logged more than once (a
retried episode), the last record wins.
"""

from __future__ import annotations

import json

import pytest

from orbit_guard.evaluation import experiment, runner
from orbit_guard.evaluation.runner import (
    EpisodeTask,
    _completed_keys,
    read_records,
    record_key,
    run_tasks,
)


def _task(seed: int) -> EpisodeTask:
    return EpisodeTask(model="rule_based", condition="normal", scenario_seed=seed, n_agents=3, experiment_id="unit")


def _record(seed: int, status: str = "ok", marker: str = "") -> dict:
    return {
        "experiment_id": "unit", "model_name": "rule_based", "training_seed": None, "shift_type": "normal",
        "scenario_seed": seed, "status": status, "marker": marker,
    }


def _write(path, *lines: str) -> None:
    path.write_text("".join(lines))


def _line(record: dict) -> str:
    return json.dumps(record) + "\n"


@pytest.fixture
def fake_runner(monkeypatch):
    """run_episode_task replaced by a recorder that returns an ok record; workers=1 looks it up at call time."""
    ran: list[int] = []

    def fake(task: EpisodeTask) -> dict:
        ran.append(task.scenario_seed)
        return _record(task.scenario_seed, marker="rerun")

    monkeypatch.setattr(runner, "run_episode_task", fake)
    return ran


def test_record_key_matches_the_task_key():
    task = _task(7)

    assert record_key(_record(7)) == task.key()


def test_a_torn_last_line_is_skipped_with_a_warning(tmp_path, capsys):
    log = tmp_path / "e.jsonl"
    _write(log, _line(_record(1)), '{"experiment_id": "unit", "model_na')

    records = read_records(log)

    assert [r["scenario_seed"] for r in records] == [1]
    assert "torn" in capsys.readouterr().err


def test_an_invalid_line_in_the_middle_of_the_log_raises(tmp_path):
    log = tmp_path / "e.jsonl"
    _write(log, _line(_record(1)), "{not json\n", _line(_record(2)))

    with pytest.raises(ValueError, match="line 2"):
        read_records(log)


def test_completed_keys_tolerate_a_torn_last_line(tmp_path):
    log = tmp_path / "e.jsonl"
    _write(log, _line(_record(1)), '{"experiment_id": ')

    assert _completed_keys(log) == {_task(1).key()}


def test_the_last_record_for_a_key_wins(tmp_path):
    log = tmp_path / "e.jsonl"
    _write(log, _line(_record(1, "failed")), _line(_record(2)), _line(_record(1, "ok", marker="retry")))

    records = read_records(log)

    assert [(r["scenario_seed"], r["status"]) for r in records] == [(2, "ok"), (1, "ok")]
    assert records[-1]["marker"] == "retry"


def test_run_tasks_cuts_a_torn_tail_before_appending(tmp_path, fake_runner, capsys):
    log = tmp_path / "e.jsonl"
    _write(log, _line(_record(1)), '{"experiment_id": "unit", "mod')

    run_tasks([_task(1), _task(2)], log, workers=1)

    assert fake_runner == [2]
    assert [r["scenario_seed"] for r in read_records(log)] == [1, 2]
    assert all(json.loads(line) for line in log.read_text().splitlines())
    assert "torn" in capsys.readouterr().err


def test_run_tasks_keeps_a_valid_last_line_that_lacks_its_newline(tmp_path, fake_runner):
    log = tmp_path / "e.jsonl"
    _write(log, _line(_record(1)), json.dumps(_record(2)))

    run_tasks([_task(1), _task(2), _task(3)], log, workers=1)

    assert fake_runner == [3]
    assert [r["scenario_seed"] for r in read_records(log)] == [1, 2, 3]


def test_failed_episodes_are_not_rerun_by_default(tmp_path, fake_runner):
    log = tmp_path / "e.jsonl"
    _write(log, _line(_record(1, "failed")), _line(_record(2)))

    run_tasks([_task(1), _task(2)], log, workers=1)

    assert fake_runner == []


def test_retry_failed_reruns_only_episodes_whose_latest_record_failed(tmp_path, fake_runner):
    log = tmp_path / "e.jsonl"
    _write(
        log,
        _line(_record(1, "failed")),
        _line(_record(2)),
        _line(_record(3, "failed")),
        _line(_record(3, "ok")),  # already retried successfully
    )

    run_tasks([_task(1), _task(2), _task(3)], log, workers=1, retry_failed=True)

    assert fake_runner == [1]
    latest = {r["scenario_seed"]: r for r in read_records(log)}
    assert latest[1]["status"] == "ok" and latest[1]["marker"] == "rerun"


def test_the_cli_passes_retry_failed_to_run_tasks(tmp_path, monkeypatch):
    registry = tmp_path / "reg.json"
    registry.write_text("{}")
    seen: dict = {}
    monkeypatch.setattr(experiment, "run_tasks", lambda tasks, out, **kwargs: seen.update(kwargs) or [])
    monkeypatch.setattr(experiment, "read_records", lambda path: [])

    experiment.main(
        [
            "run", "--split", "validation", "--max-seeds", "1", "--models", "no_maneuver",
            "--registry", str(registry), "--out", str(tmp_path / "e.jsonl"), "--retry-failed",
        ]
    )

    assert seen["retry_failed"] is True
