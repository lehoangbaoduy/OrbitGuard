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

Per-step logging (PRD §40, FR13) is opt-in per task: when `EpisodeTask.step_log_dir` is set (a plain string,
since tasks are pickled into spawn workers), the runner collects `env.step_records()` after every `env.step()`
and writes them with `evaluation.step_log` as one gzipped JSON Lines file per episode. The record then carries
`step_log` (the file's path relative to the PARENT of the step-log directory, so '<dir name>/<file>'; the
experiment CLI puts the directory next to the episode log) and `step_log_sha256`. A failed episode logs the
steps that completed and is still recorded as failed. With no directory both fields are None.

Besides the existing summary keys, every record carries the PRD §40 per-episode summary names that the
EpisodeSummary does not spell the same way: `total_delta_v` (= total_delta_v_mps), `return`
(= return_per_agent) and `mission_deviation_summary` = {"mean", "max"} of the post-step mission deviations.
The original keys stay, because the experiment reports and older logs read them.
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
from orbit_guard.evaluation.step_log import write_step_log
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
    step_log_dir: str | None = None  # None = no per-step log (the grid and validation passes stay cheap)

    def __post_init__(self) -> None:
        if self.step_log_dir is not None and not isinstance(self.step_log_dir, str):
            raise TypeError(f"step_log_dir must be a plain string or None, got {type(self.step_log_dir).__name__}")
        if self.step_log_dir == "":
            raise ValueError("step_log_dir must be a directory path or None, not an empty string")

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
    record = _metadata(task)
    started = time.perf_counter()
    env = factory({})
    step_records: list[dict] = []
    try:
        episode_return = _play(env, policy, task.scenario_seed, None if task.step_log_dir is None else step_records)
        record.update(
            {"status": "ok", **_summary_fields(env.episode_summary(), episode_return), "error": None}
        )
    except Exception as error:  # noqa: BLE001 - a failed episode is data (PRD §36 #9), not a crash
        record.update(
            {
                "status": "failed",
                **_summary_fields(env.episode_summary(), float("nan")),
                "failure_labels": [LABEL_SIMULATION_FAILURE],
                "error": f"{type(error).__name__}: {error}",
            }
        )
    record.update(_write_steps(task, step_records))
    record["wall_time_s"] = time.perf_counter() - started
    return record


def _metadata(task: EpisodeTask) -> dict[str, Any]:
    return {
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


def _play(env, policy, scenario_seed: int, step_sink: list[dict] | None) -> float:
    """Run the episode to the end; append each completed step's §40 records to `step_sink` if given."""
    observations, _ = env.reset(seed=scenario_seed)
    episode_return = 0.0
    while env.agents:
        actions = {agent: int(policy(agent, observations[agent])) for agent in env.agents}
        observations, rewards, _, _, _ = env.step(actions)
        if step_sink is not None:
            step_sink.extend(env.step_records())
        episode_return += sum(rewards.values()) / max(len(rewards), 1)
    return episode_return


def _write_steps(task: EpisodeTask, step_records: list[dict]) -> dict[str, str | None]:
    """Write the episode's step log (if enabled); an I/O error here propagates, it is not episode data."""
    if task.step_log_dir is None:
        return {"step_log": None, "step_log_sha256": None}
    directory = Path(task.step_log_dir).resolve()
    written = write_step_log(directory, task.key(), step_records)
    return {"step_log": f"{directory.name}/{written.relative_path}", "step_log_sha256": written.sha256}


def _summary_fields(summary, episode_return: float) -> dict:
    """The EpisodeSummary fields plus the PRD §40 per-episode summary names (aliases, nothing renamed)."""
    fields = dict(summary.__dict__)
    fields["failure_labels"] = list(fields["failure_labels"])
    return {
        "return_per_agent": episode_return,
        **fields,
        "total_delta_v": fields["total_delta_v_mps"],
        "mission_deviation_summary": {
            "mean": fields["mean_mission_deviation"], "max": fields["max_mission_deviation"],
        },
        "return": episode_return,
    }


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
