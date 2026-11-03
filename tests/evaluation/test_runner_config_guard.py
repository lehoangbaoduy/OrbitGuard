"""Resuming an episode log under a different configuration or checkpoint is refused (PRD §33, §40).

The resume key (experiment, model, training seed, condition, scenario seed) says nothing about the config or the
weights, so a rerun into the same OUT after a retrain or with a new overlay would silently skip every task and
report the old configuration's numbers. Each record therefore carries `config_hash` and the checkpoint's
`actor_weights_sha256`, and `run_tasks` refuses, before touching the file, when a logged record for any task it
was given disagrees with what that task would run.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.evaluation import runner
from orbit_guard.evaluation.runner import (
    EpisodeLogMismatchError,
    EpisodeTask,
    run_tasks,
)


def _task(seed: int, **overrides) -> EpisodeTask:
    fields = {"model": "rule_based", "condition": "normal", "scenario_seed": seed, "n_agents": 3,
              "experiment_id": "unit"}
    fields.update(overrides)
    return EpisodeTask(**fields)


def _record(seed: int, *, model: str = "rule_based", training_seed: int | None = None, status: str = "ok",
            **fields) -> dict:
    record = {
        "experiment_id": "unit", "model_name": model, "training_seed": training_seed, "shift_type": "normal",
        "scenario_seed": seed, "status": status, "config_hash": config_digest(DEFAULT_CONFIG_PATH),
        "actor_weights_sha256": None,
    }
    record.update(fields)
    return record


def _line(record: dict) -> str:
    return json.dumps(record) + "\n"


def _weights(tmp_path, content: bytes = b"weights-v1") -> str:
    path = tmp_path / "actor_weights.npz"
    path.write_bytes(content)
    return str(path)


def _sha(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@pytest.fixture
def fake_runner(monkeypatch):
    ran: list[int] = []

    def fake(task: EpisodeTask) -> dict:
        ran.append(task.scenario_seed)
        return _record(task.scenario_seed, marker="rerun")

    monkeypatch.setattr(runner, "run_episode_task", fake)
    return ran


# --- the record ---------------------------------------------------------------------------------------------


def test_the_record_carries_the_checkpoint_weights_sha256(tmp_path):
    weights = _weights(tmp_path)

    learned = runner._metadata(_task(1, model="adaptive_mappo", training_seed=0, actor_weights_path=weights))
    scripted = runner._metadata(_task(1))

    assert learned["actor_weights_sha256"] == _sha(weights)
    assert scripted["actor_weights_sha256"] is None


# --- resume under a changed configuration -------------------------------------------------------------------


def test_resuming_under_a_different_config_hash_is_refused_and_names_key_and_both_hashes(tmp_path, fake_runner):
    log = tmp_path / "e.jsonl"
    log.write_text(_line(_record(1, config_hash="0" * 64)))
    before = log.read_bytes()

    with pytest.raises(EpisodeLogMismatchError) as caught:
        run_tasks([_task(1), _task(2)], log, workers=1)

    message = str(caught.value)
    assert _task(1).key() in message
    assert "0" * 64 in message
    assert config_digest(DEFAULT_CONFIG_PATH) in message
    assert fake_runner == []
    assert log.read_bytes() == before


def test_resuming_under_a_new_overlay_is_refused(tmp_path, fake_runner):
    overlay = tmp_path / "tuned.yaml"
    overlay.write_text("reward:\n  w3: 1.5\n")
    log = tmp_path / "e.jsonl"
    log.write_text(_line(_record(1)))

    with pytest.raises(EpisodeLogMismatchError, match="config_hash"):
        run_tasks([_task(1, overlay_path=str(overlay))], log, workers=1)

    assert fake_runner == []


def test_resuming_after_a_retrain_is_refused(tmp_path, fake_runner):
    old = _weights(tmp_path, b"weights-v1")
    old_sha = _sha(old)
    _weights(tmp_path, b"retrained")  # same path, new bytes
    log = tmp_path / "e.jsonl"
    log.write_text(_line(_record(1, model="adaptive_mappo", training_seed=0, actor_weights_sha256=old_sha)))

    with pytest.raises(EpisodeLogMismatchError) as caught:
        run_tasks([_task(1, model="adaptive_mappo", training_seed=0, actor_weights_path=old)], log, workers=1)

    message = str(caught.value)
    assert "actor_weights_sha256" in message
    assert old_sha in message and _sha(old) in message
    assert fake_runner == []


def test_an_old_record_without_a_checkpoint_hash_mismatches_a_learned_task(tmp_path, fake_runner):
    weights = _weights(tmp_path)
    old = _record(1, model="adaptive_mappo", training_seed=0)
    del old["actor_weights_sha256"]
    log = tmp_path / "e.jsonl"
    log.write_text(_line(old))

    with pytest.raises(EpisodeLogMismatchError, match="predates"):
        run_tasks([_task(1, model="adaptive_mappo", training_seed=0, actor_weights_path=weights)], log, workers=1)


def test_an_old_record_without_a_checkpoint_hash_still_resumes_a_scripted_task(tmp_path, fake_runner):
    old = _record(1)
    del old["actor_weights_sha256"]
    log = tmp_path / "e.jsonl"
    log.write_text(_line(old))

    run_tasks([_task(1), _task(2)], log, workers=1)

    assert fake_runner == [2]


def test_matching_records_resume_as_before(tmp_path, fake_runner):
    weights = _weights(tmp_path)
    log = tmp_path / "e.jsonl"
    log.write_text(
        _line(_record(1)) + _line(_record(1, model="adaptive_mappo", training_seed=0, actor_weights_sha256=_sha(weights)))
    )

    run_tasks(
        [_task(1), _task(2), _task(1, model="adaptive_mappo", training_seed=0, actor_weights_path=weights)],
        log, workers=1,
    )

    assert fake_runner == [2]


def test_a_failed_record_to_be_retried_under_another_config_is_refused_too(tmp_path, fake_runner):
    log = tmp_path / "e.jsonl"
    log.write_text(_line(_record(1, status="failed", config_hash="f" * 64)))

    with pytest.raises(EpisodeLogMismatchError):
        run_tasks([_task(1)], log, workers=1, retry_failed=True)

    assert fake_runner == []


def test_only_the_latest_record_of_a_key_is_compared(tmp_path, fake_runner):
    log = tmp_path / "e.jsonl"
    log.write_text(_line(_record(1, status="failed", config_hash="f" * 64)) + _line(_record(1)))

    run_tasks([_task(1)], log, workers=1)

    assert fake_runner == []


def test_a_refused_resume_leaves_a_torn_tail_untouched(tmp_path, fake_runner, capsys):
    log = tmp_path / "e.jsonl"
    log.write_text(_line(_record(1, config_hash="0" * 64)) + '{"experiment_id": "unit", "mod')
    before = log.read_bytes()

    with pytest.raises(EpisodeLogMismatchError):
        run_tasks([_task(1)], log, workers=1)

    assert log.read_bytes() == before
