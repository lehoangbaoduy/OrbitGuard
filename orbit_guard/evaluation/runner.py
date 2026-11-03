"""Episode runner for the experiment matrix (PRD §33-35, §40).

One `EpisodeTask` = one (model, shift condition, scenario seed, training seed)
episode. The task names the base config and the optional tuned overlay (CORE-1); the record's
`config_hash` is `config.config_digest(config_path, overlay_path)`, the same digest training writes to
run_config.json, so an episode can be traced to the exact configuration it ran under.
`run_episode_task` is a module-level pure function of the task, so it
pickles into worker processes and is reproducible: the same task gives the same
record up to the timing fields (Acceptance Test F). Learned policies run from
exported NumPy actor weights (`rl.actor_policy`), so workers need neither Ray
nor torch.

`run_tasks` fans tasks out over processes, appends one JSON line per episode as
it completes, and RESUMES: tasks whose key is already in the output file are
skipped, so a long sweep survives interruption. A sweep killed mid-write leaves a
torn LAST line: readers skip it with a warning on stderr, and `run_tasks` cuts it
off before appending so it can never turn into a corrupt line mid-file. An
invalid line anywhere else is corruption and raises. When a key is logged more
than once (a retried episode) the LAST record wins; `retry_failed=True` reruns the
tasks whose latest record has status 'failed'.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from orbit_guard.config import config_digest
from orbit_guard.evaluation.baselines import build_env_factory, build_policy
from orbit_guard.evaluation.shifts import condition_env_kwargs
from orbit_guard.rl.episode_metrics import LABEL_SIMULATION_FAILURE

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class EpisodeTask:
    model: str
    condition: str
    scenario_seed: int
    n_agents: int = 15
    training_seed: int | None = None
    actor_weights_path: str | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    config_path: str = str(_DEFAULT_CONFIG)
    overlay_path: str | None = None
    experiment_id: str = "adhoc"
    git_commit: str = "unknown"

    def key(self) -> str:
        return _key(self.experiment_id, self.model, self.training_seed, self.condition, self.scenario_seed)


def _key(experiment_id: str, model: str, training_seed: int | None, condition: str, scenario_seed: int) -> str:
    return f"{experiment_id}|{model}|{training_seed}|{condition}|{scenario_seed}"


def record_key(record: dict) -> str:
    """The resume/dedup key of a logged record; equal to `EpisodeTask.key()` of the task that produced it."""
    return _key(
        record["experiment_id"], record["model_name"], record["training_seed"], record["shift_type"],
        record["scenario_seed"],
    )


def run_episode_task(task: EpisodeTask) -> dict:
    """Run one episode and return its flat, JSON-serializable record (PRD §40 metadata + summary)."""
    policy = build_policy(
        task.model, task.actor_weights_path, config_path=task.config_path, overlay_path=task.overlay_path
    )
    shift_kwargs = condition_env_kwargs(task.condition, task.config_path)
    factory = build_env_factory(
        task.model, task.n_agents, config_path=task.config_path, overlay_path=task.overlay_path,
        **{**shift_kwargs, **task.env_kwargs},
    )
    record: dict[str, Any] = {
        "experiment_id": task.experiment_id,
        "scenario_id": f"{task.condition}-{task.scenario_seed}",
        "scenario_seed": task.scenario_seed,
        "training_seed": task.training_seed,
        "model_name": task.model,
        "git_commit": task.git_commit,
        "config_hash": config_digest(task.config_path, task.overlay_path),
        "config_path": task.config_path,
        "overlay_path": task.overlay_path,
        "start_time": datetime.now(UTC).isoformat(),
        "n_satellites": task.n_agents,
        "shift_type": task.condition,
    }
    started = time.perf_counter()
    env = factory({})
    try:
        observations, _ = env.reset(seed=task.scenario_seed)
        episode_return = 0.0
        while env.agents:
            actions = {agent: int(policy(agent, observations[agent])) for agent in env.agents}
            observations, rewards, _, _, _ = env.step(actions)
            episode_return += sum(rewards.values()) / max(len(rewards), 1)
        summary = env.episode_summary()
        record.update(
            {"status": "ok", "return_per_agent": episode_return, **_summary_fields(summary), "error": None}
        )
    except Exception as error:  # noqa: BLE001 - a failed episode is data (PRD §36 #9), not a crash
        record.update(
            {
                "status": "failed",
                "return_per_agent": float("nan"),
                **_summary_fields(env.episode_summary()),
                "failure_labels": [LABEL_SIMULATION_FAILURE],
                "error": f"{type(error).__name__}: {error}",
            }
        )
    record["wall_time_s"] = time.perf_counter() - started
    return record


def _summary_fields(summary) -> dict:
    fields = dict(summary.__dict__)
    fields["failure_labels"] = list(fields["failure_labels"])
    return fields


@dataclass(frozen=True)
class _ParsedLog:
    records: tuple[dict, ...]  # every valid record, in file order (duplicates kept)
    valid_end_byte: int  # the file holds only valid lines before this offset
    torn_tail: bool  # the last line was invalid (a killed sweep) and was skipped


def _parse_log(path: Path) -> _ParsedLog:
    """Parse a JSONL episode log; an invalid LAST line is a torn write, an invalid earlier line raises."""
    data = path.read_bytes()
    lines = data.split(b"\n")
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    last_content = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    records: list[dict] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        record = _decode_record(line)
        if record is not None:
            records.append(record)
        elif index == last_content:
            return _ParsedLog(tuple(records), starts[index], torn_tail=True)
        else:
            raise ValueError(f"{path}: line {index + 1} is not a valid JSON record; the episode log is corrupt")
    return _ParsedLog(tuple(records), len(data), torn_tail=False)


def _decode_record(line: bytes) -> dict | None:
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _warn_torn(path: Path, action: str) -> None:
    print(f"warning: {path}: torn last line from an interrupted sweep, {action}", file=sys.stderr)


def _read_log(path: Path) -> _ParsedLog:
    parsed = _parse_log(path)
    if parsed.torn_tail:
        _warn_torn(path, "skipped")
    return parsed


def _latest_by_key(records: tuple[dict, ...]) -> dict[str, dict]:
    """Key -> latest record; iteration order is the order of each key's LAST occurrence."""
    latest: dict[str, dict] = {}
    for record in records:
        key = record_key(record)
        latest.pop(key, None)
        latest[key] = record
    return latest


def _completed_keys(path: Path, retry_failed: bool = False) -> set[str]:
    """Keys already logged; with `retry_failed`, keys whose latest record failed count as not done."""
    if not path.exists():
        return set()
    latest = _latest_by_key(_read_log(path).records)
    return {key for key, record in latest.items() if not (retry_failed and record.get("status") == "failed")}


def _repair_tail(path: Path) -> None:
    """Cut a torn last line off (or terminate a valid unterminated one) so appends start on a clean line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    parsed = _parse_log(path)
    with path.open("r+b") as handle:
        if parsed.torn_tail:
            _warn_torn(path, f"truncated to byte {parsed.valid_end_byte} before appending")
            handle.truncate(parsed.valid_end_byte)
        handle.seek(0, 2)
        size = handle.tell()
        if size:
            handle.seek(size - 1)
            if handle.read(1) != b"\n":
                handle.write(b"\n")


def run_tasks(
    tasks: list[EpisodeTask], out_path: str | Path, workers: int = 1, *, retry_failed: bool = False
) -> list[dict]:
    """Run `tasks` (skipping those already in `out_path`), appending each record as it finishes.

    With `retry_failed`, tasks whose latest logged record has status 'failed' run again; the new record is
    appended and, being the last for its key, supersedes the failed one in `read_records`.
    """
    out = Path(out_path)
    done = _completed_keys(out, retry_failed)
    pending = [task for task in tasks if task.key() not in done]
    _repair_tail(out)
    records: list[dict] = []
    with out.open("a") as handle:
        if workers <= 1:
            results = map(run_episode_task, pending)
        else:
            pool = get_context("spawn").Pool(workers)
            results = pool.imap_unordered(run_episode_task, pending)
        try:
            for record in results:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
        finally:
            if workers > 1:
                pool.close()
                pool.join()
    return records


def read_records(path: str | Path) -> list[dict]:
    """The latest record per key, in order of each key's last occurrence; a torn last line is skipped."""
    return list(_latest_by_key(_read_log(Path(path)).records).values())


def mean_of(records: list[dict], key: str) -> float:
    return float(np.mean([record[key] for record in records]))
