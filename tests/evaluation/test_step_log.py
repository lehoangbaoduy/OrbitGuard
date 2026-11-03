"""Tests for orbit_guard.evaluation.step_log: atomic, hash-verified gzipped JSON Lines of §40 step records."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math

import pytest

from orbit_guard.evaluation.step_log import (
    StepLogIntegrityError,
    read_step_log,
    step_log_filename,
    write_step_log,
)


def _records(n: int = 3) -> list[dict]:
    return [
        {"timestamp": 60.0 * i, "satellite_id": f"sat_{i}", "risk_before": 0.1 * i, "shield_reason": []}
        for i in range(n)
    ]


def _reject_constant(name: str):
    raise ValueError(f"non-strict JSON constant {name}")


def _strict_lines(path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line, parse_constant=_reject_constant) for line in handle if line.strip()]


def test_write_then_read_round_trips_the_records(tmp_path):
    records = _records()

    written = write_step_log(tmp_path, "exp|rule_based|None|normal|3", records)

    assert read_step_log(tmp_path / written.relative_path, written.sha256) == records


def test_the_returned_path_is_relative_and_the_hash_is_the_file_sha256(tmp_path):
    written = write_step_log(tmp_path, "exp|rule_based|None|normal|3", _records())

    path = tmp_path / written.relative_path
    assert not written.relative_path.startswith("/")
    assert path.parent == tmp_path
    assert written.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_file_is_gzipped_json_lines_one_line_per_record(tmp_path):
    records = _records(5)

    written = write_step_log(tmp_path, "k", records)

    assert _strict_lines(tmp_path / written.relative_path) == records


def test_the_filename_uses_a_safe_character_set_and_distinguishes_colliding_keys():
    unsafe = "exp|adaptive mappo|0|normal/../../etc|3"
    similar = "exp_adaptive_mappo_0_normal_.._.._etc_3"

    name = step_log_filename(unsafe)

    assert set(name) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    assert "/" not in name and not name.startswith(".")
    assert name != step_log_filename(similar)
    assert len(step_log_filename("x" * 5000)) <= 160


def test_identical_records_give_an_identical_hash(tmp_path):
    first = write_step_log(tmp_path / "a", "k", _records())
    second = write_step_log(tmp_path / "b", "k", _records())

    assert first.sha256 == second.sha256


def test_a_tampered_file_fails_hash_verification(tmp_path):
    written = write_step_log(tmp_path, "k", _records())
    path = tmp_path / written.relative_path
    lines = _strict_lines(path)
    lines[1]["risk_before"] = 0.0
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines).encode())
    path.write_bytes(buffer.getvalue())  # a valid gzip, so only the sha256 check can catch it

    with pytest.raises(StepLogIntegrityError, match="sha256"):
        read_step_log(path, written.sha256)


def test_a_writer_that_fails_midway_leaves_no_partial_file(tmp_path):
    records = [*_records(2), {"bad": object()}, *_records(2)]

    with pytest.raises(TypeError):
        write_step_log(tmp_path, "k", records)

    assert list(tmp_path.iterdir()) == []


def test_a_failed_overwrite_leaves_the_previous_file_byte_identical(tmp_path):
    written = write_step_log(tmp_path, "k", _records())
    before = (tmp_path / written.relative_path).read_bytes()

    with pytest.raises(TypeError):
        write_step_log(tmp_path, "k", [*_records(), {"bad": object()}])

    assert [p.name for p in tmp_path.iterdir()] == [written.relative_path]
    assert (tmp_path / written.relative_path).read_bytes() == before


def test_an_io_failure_after_the_temp_file_exists_removes_it_and_keeps_the_old_file(tmp_path, monkeypatch):
    from orbit_guard.evaluation import step_log

    written = write_step_log(tmp_path, "k", _records())
    before = (tmp_path / written.relative_path).read_bytes()

    def failing_fsync(fd):
        raise OSError("disk full")

    monkeypatch.setattr(step_log.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="disk full"):
        write_step_log(tmp_path, "k", _records(7))

    assert [p.name for p in tmp_path.iterdir()] == [written.relative_path]
    assert (tmp_path / written.relative_path).read_bytes() == before


def test_non_finite_floats_are_written_as_null_so_the_json_stays_strict(tmp_path):
    records = [{"risk_before": math.nan, "values": [1.0, math.inf, -math.inf], "nested": {"x": math.nan}}]

    written = write_step_log(tmp_path, "k", records)

    assert _strict_lines(tmp_path / written.relative_path) == [
        {"risk_before": None, "values": [1.0, None, None], "nested": {"x": None}}
    ]


def test_an_empty_episode_writes_an_empty_verifiable_log(tmp_path):
    written = write_step_log(tmp_path, "k", [])

    assert read_step_log(tmp_path / written.relative_path, written.sha256) == []


def test_the_writer_creates_the_directory(tmp_path):
    target = tmp_path / "out" / "steps"

    written = write_step_log(target, "k", _records())

    assert (target / written.relative_path).is_file()
