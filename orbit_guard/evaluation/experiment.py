"""Experiment orchestration: task matrix, aggregation, paired report (PRD §33-35).

  * `build_tasks`       model x condition x scenario seed (x training seed) -> `EpisodeTask`s. Every model
                        gets IDENTICAL scenario seeds per condition, which is what makes paired comparison valid.
  * `summarize`         one row per (model, condition): primary endpoint (collision-episode rate) and the
                        secondary endpoints, pooled across training seeds, with degradation vs `normal`.
  * `paired_by_scenario` paired difference between two models on matched scenario seeds.
  * `format_report`     the markdown table.
CLI subcommands (see `main`): `freeze`, `run`, `report`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from orbit_guard.evaluation.baselines import MODEL_NAMES, _model_spec
from orbit_guard.evaluation.runner import EpisodeTask, read_records, run_tasks
from orbit_guard.evaluation.statistics import (
    PairedComparison,
    Summary,
    degradation,
    paired_comparison,
    pooled_by_training_seed,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def build_tasks(
    models: Sequence[str],
    condition_seeds: Mapping[str, Sequence[int]],
    registry: Mapping[str, Mapping[int, str]],
    *,
    n_agents: int,
    experiment_id: str,
    git_commit: str = "unknown",
    config_path: str | Path = _DEFAULT_CONFIG,
    env_kwargs: Mapping | None = None,
) -> list[EpisodeTask]:
    tasks: list[EpisodeTask] = []
    for model in models:
        learned = _model_spec(model, config_path)["policy"] == "mappo"
        if learned and not registry.get(model):
            raise ValueError(f"{model} is learned: register its trained checkpoints (training seed -> actor weights)")
        variants = sorted(registry[model].items()) if learned else [(None, None)]
        for training_seed, weights in variants:
            for condition, seeds in condition_seeds.items():
                tasks.extend(
                    EpisodeTask(
                        model=model, condition=condition, scenario_seed=seed, n_agents=n_agents,
                        training_seed=training_seed, actor_weights_path=weights,
                        env_kwargs=dict(env_kwargs or {}), config_path=str(config_path),
                        experiment_id=experiment_id, git_commit=git_commit,
                    )
                    for seed in seeds
                )
    return tasks


def _derived(record: Mapping) -> dict[str, float]:
    steps = max(int(record["steps"]), 1)
    return {
        "collision_rate": float(record["collision_count"] > 0),
        "unsafe_rate": float(record["unsafe_conjunction_count"] > 0 or record["collision_count"] > 0),
        "total_delta_v_mps": float(record["total_delta_v_mps"]),
        "maneuver_count": float(record["maneuver_count"]),
        "mean_mission_deviation": float(record["mean_mission_deviation"]),
        "shield_intervention_rate": record["shield_intervention_count"] / steps,
        "mean_residual_risk": float(record["mean_residual_risk"]),
        "return_per_agent": float(record["return_per_agent"]),
    }


_METRICS = tuple(_derived(
    {"steps": 1, "collision_count": 0, "unsafe_conjunction_count": 0, "total_delta_v_mps": 0,
     "maneuver_count": 0, "mean_mission_deviation": 0, "shield_intervention_count": 0,
     "mean_residual_risk": 0, "return_per_agent": 0}
))


def summarize(records: Sequence[Mapping]) -> list[dict]:
    """One row per (model, condition); metrics are `Summary`s pooled across training seeds."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("status", "ok") != "ok":
            continue
        groups[(record["model_name"], record["shift_type"])].append(
            {"training_seed": record["training_seed"], **_derived(record)}
        )

    rows = []
    for (model, condition), episodes in groups.items():
        row: dict = {
            "model": model, "condition": condition, "episodes": len(episodes),
            "training_seeds": len({e["training_seed"] for e in episodes}),
        }
        for metric in _METRICS:
            row[metric] = pooled_by_training_seed(episodes, metric)
        rows.append(row)

    normal = {row["model"]: row for row in rows if row["condition"] == "normal"}
    for row in rows:
        for metric in ("collision_rate", "total_delta_v_mps"):
            base = normal.get(row["model"])
            row[f"{metric}_degradation"] = (
                degradation(row[metric].mean, base[metric].mean) if base is not None else float("nan")
            )
    return sorted(rows, key=lambda r: (r["condition"], MODEL_NAMES.index(r["model"])))


def _per_scenario(records: Sequence[Mapping], model: str, condition: str, metric: str) -> dict[int, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record["model_name"] == model and record["shift_type"] == condition and record.get("status", "ok") == "ok":
            by_seed[record["scenario_seed"]].append(float(record[metric]))
    return {seed: float(np.mean(values)) for seed, values in by_seed.items()}


def paired_by_scenario(
    records: Sequence[Mapping], method: str, baseline: str, condition: str, metric: str
) -> PairedComparison:
    """`method - baseline` on the scenario seeds both ran (training seeds averaged per scenario)."""
    method_values = _per_scenario(records, method, condition, metric)
    baseline_values = _per_scenario(records, baseline, condition, metric)
    shared = sorted(set(method_values) & set(baseline_values))
    if not shared:
        raise ValueError(f"{method} and {baseline} share no scenario seeds under {condition}")
    return paired_comparison([method_values[s] for s in shared], [baseline_values[s] for s in shared])


def _cell(summary: Summary, digits: int = 3) -> str:
    return f"{summary.mean:.{digits}f} ± {summary.std:.{digits}f}"


def format_report(records: Sequence[Mapping]) -> str:
    rows = summarize(records)
    lines = [
        (
            "| model | condition | episodes | training seeds | collision-episode rate | unsafe-conjunction rate "
            "| Δv (m/s) | maneuvers | mission dev. | shield interv./step | residual risk |"
        ),
        "|---|---|---:|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['episodes']} | {row['training_seeds']} "
            f"| {_cell(row['collision_rate'])} | {_cell(row['unsafe_rate'])} | {_cell(row['total_delta_v_mps'])} "
            f"| {_cell(row['maneuver_count'], 1)} | {_cell(row['mean_mission_deviation'])} "
            f"| {_cell(row['shield_intervention_rate'], 4)} | {_cell(row['mean_residual_risk'])} |"
        )
    lines.append("")
    lines.append("Values are mean ± std across training seeds (per-seed means first); non-learned models have one 'seed'.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: `run` a sweep from a registry of exported actors, then `report` on the episode log."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the model x condition x seed sweep")
    run.add_argument("--split", choices=("validation", "test"), required=True)
    run.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    run.add_argument("--registry", required=True, help='JSON: {"model": {"training_seed": "actor_weights.npz"}}')
    run.add_argument("--out", required=True)
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--n-agents", type=int, default=15)
    run.add_argument("--experiment-id", default="exp")
    run.add_argument("--max-seeds", type=int, default=None, help="validation split: use only this many seeds")
    run.add_argument("--freeze-snapshot", default=None, help="required for --split test")
    run.add_argument("--test-pool", default="configs/test_seeds.json")

    report = sub.add_parser("report", help="markdown table from an episode log")
    report.add_argument("--episodes", required=True)

    freeze = sub.add_parser("freeze", help="snapshot config + checkpoints before the final test")
    freeze.add_argument("--registry", required=True)
    freeze.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "report":
        print(format_report(read_records(args.episodes)))
        return

    registry_raw = json.loads(Path(args.registry).read_text())
    registry = {model: {int(seed): path for seed, path in seeds.items()} for model, seeds in registry_raw.items()}
    flat = {f"{model}/{seed}": path for model, seeds in registry.items() for seed, path in seeds.items()}

    if args.command == "freeze":
        from orbit_guard.evaluation.freeze import write_freeze_snapshot

        print(json.dumps(write_freeze_snapshot(args.out, flat), indent=2))
        return

    from orbit_guard.evaluation.freeze import _git_commit
    from orbit_guard.evaluation.shifts import CONDITION_NAMES

    if args.split == "validation":
        from orbit_guard.evaluation.seed_pools import load_seed_pool

        seeds = load_seed_pool("configs/validation_seeds.json", "validation")[: args.max_seeds]
        condition_seeds = {"normal": seeds}
    else:
        if args.freeze_snapshot is None:
            parser.error("--split test requires --freeze-snapshot (freeze configs and checkpoints first)")
        from orbit_guard.evaluation.test_seed_pool import load_test_pool

        condition_seeds = load_test_pool(args.test_pool, args.freeze_snapshot, flat)
        assert tuple(condition_seeds) == CONDITION_NAMES

    tasks = build_tasks(
        args.models, condition_seeds, registry, n_agents=args.n_agents,
        experiment_id=args.experiment_id, git_commit=_git_commit(),
    )
    print(f"{len(tasks)} episodes ({args.workers} workers)", flush=True)
    run_tasks(tasks, args.out, workers=args.workers)
    print(format_report(read_records(args.out)))


if __name__ == "__main__":
    main()
