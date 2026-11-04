"""Per-control-step episode logs (PRD §40, FR13): one gzipped JSON Lines file per episode.

`write_step_log(directory, episode_key, records)` writes one line per §40 step record
(`OrbitGuardEnv.step_records()`) to `directory/step_log_filename(episode_key)` ATOMICALLY: the bytes go to a
hidden temp file in the same directory, are fsynced, and only then `os.replace`d onto the final name, so a
crash or an exception mid-write never leaves a partial file (and never damages an existing one). It returns
the file's path relative to `directory` and the sha256 of its bytes, which the episode record stores.
`read_step_log(path, sha256)` re-hashes the file before parsing and raises `StepLogIntegrityError` on a
mismatch.

Determinism: the gzip header carries no file name and mtime 0, and lines are written in record order with
fixed separators, so the same records always give the same bytes and the same sha256 (Acceptance Test F).

JSON is strict: non-finite floats (NaN, ±inf, e.g. from a numerically failed episode) are written as `null`,
never as the non-standard `NaN`/`Infinity` tokens, and the writer runs with `allow_nan=False`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM_CHARS = 100
_KEY_HASH_CHARS = 16
_PREFIX = "steps-"
_SUFFIX = ".jsonl.gz"
_COMPRESS_LEVEL = 6


class StepLogIntegrityError(ValueError):
    """A step log's bytes do not match the sha256 recorded for it."""


@dataclass(frozen=True)
class WrittenStepLog:
    relative_path: str  # relative to the directory passed to write_step_log
    sha256: str
    lines: int


def step_log_filename(episode_key: str) -> str:
    """A filesystem-safe, collision-resistant file name for an episode key.

    Characters outside [A-Za-z0-9._-] become '_', the result is capped in length, and a hash of the FULL
    original key is appended so two keys that sanitize to the same text still get different files. The fixed
    'steps-' prefix means a name never starts with '.', so it can never clash with the writer's temp files.
    """
    stem = _SAFE_CHARS.sub("_", episode_key)[:_MAX_STEM_CHARS]
    digest = hashlib.sha256(episode_key.encode("utf-8")).hexdigest()[:_KEY_HASH_CHARS]
    return f"{_PREFIX}{stem}.{digest}{_SUFFIX}"


def _strict(value: Any) -> Any:
    """`value` with every non-finite float replaced by None (recursively through dicts, lists and tuples)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_strict(item) for item in value]
    return value


def _encode(records: Sequence[Mapping]) -> bytes:
    lines = (json.dumps(_strict(record), allow_nan=False, separators=(",", ":")) + "\n" for record in records)
    raw = "".join(lines).encode("utf-8")
    return gzip.compress(raw, compresslevel=_COMPRESS_LEVEL, mtime=0)


def fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory entry after a rename or link (a no-op where unsupported)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # not supported on every platform/filesystem; the file itself is already fsynced
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_step_log(directory: str | Path, episode_key: str, records: Sequence[Mapping]) -> WrittenStepLog:
    """Atomically write `records` as gzipped JSON Lines; return the relative path and sha256 of the file."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = _encode(records)  # serialization errors surface here, before any file exists
    name = step_log_filename(episode_key)
    handle = tempfile.NamedTemporaryFile(dir=target_dir, prefix=".", suffix=".tmp", delete=False)  # noqa: SIM115
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_dir / name)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    fsync_directory(target_dir)
    return WrittenStepLog(name, hashlib.sha256(payload).hexdigest(), len(records))


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-strict JSON constant {name!r} in a step log")


def read_step_log(path: str | Path, sha256: str) -> list[dict]:
    """The records of a step log, after checking its bytes against `sha256` (StepLogIntegrityError if not)."""
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise StepLogIntegrityError(f"{path}: sha256 {actual} does not match the recorded {sha256}")
    text = gzip.decompress(data).decode("utf-8")
    return [json.loads(line, parse_constant=_reject_constant) for line in text.splitlines() if line.strip()]


def read_episode_step_log(episode_log_dir: str | Path, record: Mapping) -> list[dict]:
    """The verified step records of an episode record; its `step_log` is relative to the episode log's directory.

    `run_episode_task` stores `step_log` as '<step-log dir name>/<file name>', i.e. relative to the PARENT of
    the step-log directory; the experiment CLI puts that directory next to the episode log (<out dir>/steps).
    """
    if record.get("step_log") is None:
        raise ValueError("this episode record has no step log (step logging was off for its task)")
    return read_step_log(Path(episode_log_dir) / record["step_log"], record["step_log_sha256"])
