"""A sweep that fails or is interrupted stops its pool instead of draining the queue, and run_tasks creates
the output directory when it is missing.

The pool is replaced by a fake, so no worker processes start: these tests check which pool calls run and
what reaches the episode log. tests/evaluation/test_runner.py covers the real parallel run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.evaluation import runner
from orbit_guard.evaluation.runner import EpisodeTask, read_records, run_tasks


def _task(seed: int) -> EpisodeTask:
    return EpisodeTask(model="rule_based", condition="normal", scenario_seed=seed, n_agents=3, experiment_id="unit")


def _record(seed: int) -> dict:
    return {
        "experiment_id": "unit", "model_name": "rule_based", "training_seed": None, "shift_type": "normal",
        "scenario_seed": seed, "status": "ok", "config_hash": config_digest(DEFAULT_CONFIG_PATH),
        "actor_weights_sha256": None,
    }


class _FakePool:
    """Yields the given records, then raises `error` (if any) the way a Ctrl-C or a failed worker does."""

    def __init__(self, records: list[dict], error: BaseException | None = None) -> None:
        self.calls: list[str] = []
        self._records = records
        self._error = error

    def imap_unordered(self, func, tasks):
        yield from self._records
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.calls.append("close")

    def terminate(self) -> None:
        self.calls.append("terminate")

    def join(self) -> None:
        self.calls.append("join")


def _use_pool(monkeypatch, pool: _FakePool) -> None:
    monkeypatch.setattr(runner, "get_context", lambda method: SimpleNamespace(Pool=lambda workers: pool))


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("worker crashed")])
def test_an_interrupted_sweep_terminates_the_pool_instead_of_draining_it(tmp_path, monkeypatch, error):
    out = tmp_path / "e.jsonl"
    pool = _FakePool([_record(1)], error)
    _use_pool(monkeypatch, pool)

    with pytest.raises(type(error)):
        run_tasks([_task(1), _task(2)], out, workers=2)

    assert pool.calls == ["terminate", "join"]


def test_records_written_before_an_interrupt_survive_intact(tmp_path, monkeypatch):
    out = tmp_path / "e.jsonl"
    _use_pool(monkeypatch, _FakePool([_record(1)], KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        run_tasks([_task(1), _task(2)], out, workers=2)

    assert out.read_text().endswith("\n")
    assert [record["scenario_seed"] for record in read_records(out)] == [1]


def test_a_finished_parallel_sweep_closes_the_pool_and_keeps_every_record(tmp_path, monkeypatch):
    out = tmp_path / "e.jsonl"
    pool = _FakePool([_record(1), _record(2)])
    _use_pool(monkeypatch, pool)

    records = run_tasks([_task(1), _task(2)], out, workers=2)

    assert pool.calls == ["close", "join"]
    assert [record["scenario_seed"] for record in records] == [1, 2]


def test_an_output_in_a_missing_directory_creates_it(tmp_path, monkeypatch):
    out = tmp_path / "sweeps" / "unit" / "e.jsonl"
    monkeypatch.setattr(runner, "run_episode_task", lambda task: _record(task.scenario_seed))

    run_tasks([_task(1)], out)

    assert [record["scenario_seed"] for record in read_records(out)] == [1]
