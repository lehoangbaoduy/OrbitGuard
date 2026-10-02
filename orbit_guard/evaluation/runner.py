"""Episode runner for the experiment matrix (PRD §33-35, §40).

One `EpisodeTask` = one (model, shift condition, scenario seed, training seed)
episode. `run_episode_task` is a module-level pure function of the task, so it
pickles into worker processes and is reproducible: the same task gives the same
record up to the timing fields (Acceptance Test F). Learned policies run from
exported NumPy actor weights (`rl.actor_policy`), so workers need neither Ray
nor torch.

`run_tasks` fans tasks out over processes, appends one JSON line per episode as
it completes, and RESUMES: tasks whose key is already in the output file are
skipped, so a long sweep survives interruption.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

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
    experiment_id: str = "adhoc"
    git_commit: str = "unknown"

    def key(self) -> str:
        return f"{self.experiment_id}|{self.model}|{self.training_seed}|{self.condition}|{self.scenario_seed}"


def config_hash(config_path: str | Path = _DEFAULT_CONFIG) -> str:
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def run_episode_task(task: EpisodeTask) -> dict:
    """Run one episode and return its flat, JSON-serializable record (PRD §40 metadata + summary)."""
    policy = build_policy(task.model, task.actor_weights_path, config_path=task.config_path)
    shift_kwargs = condition_env_kwargs(task.condition, task.config_path)
    factory = build_env_factory(
        task.model, task.n_agents, config_path=task.config_path, **{**shift_kwargs, **task.env_kwargs}
    )
    record: dict[str, Any] = {
        "experiment_id": task.experiment_id,
        "scenario_id": f"{task.condition}-{task.scenario_seed}",
        "scenario_seed": task.scenario_seed,
        "training_seed": task.training_seed,
        "model_name": task.model,
        "git_commit": task.git_commit,
        "config_hash": config_hash(task.config_path),
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


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        keys.add(
            f"{record['experiment_id']}|{record['model_name']}|{record['training_seed']}"
            f"|{record['shift_type']}|{record['scenario_seed']}"
        )
    return keys


def run_tasks(tasks: list[EpisodeTask], out_path: str | Path, workers: int = 1) -> list[dict]:
    """Run `tasks` (skipping those already in `out_path`), appending each record as it finishes."""
    out = Path(out_path)
    done = _completed_keys(out)
    pending = [task for task in tasks if task.key() not in done]
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
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def mean_of(records: list[dict], key: str) -> float:
    return float(np.mean([record[key] for record in records]))
