"""Experiment orchestration: task matrix, aggregation, paired report (PRD §33-35).

  * `build_tasks`       model x condition x scenario seed (x training seed) -> `EpisodeTask`s. Every model
                        gets IDENTICAL scenario seeds per condition, which is what makes paired comparison valid.
  * `summarize`         one row per (model, condition): primary endpoint (collision-episode rate) and the
                        secondary endpoints, pooled across training seeds, with degradation vs `normal`.
  * `paired_by_scenario` paired difference between two models on matched scenario seeds.
  * `format_report`     the markdown table.
  * `check_checkpoint_provenance`  a checkpoint trained under one configuration is never evaluated under
                        another: the run_config.json sidecar that rl/training.py writes next to
                        actor_weights.npz records `config.config_digest(config, overlay)`, and a mismatch
                        with the digest the experiment is about to use raises `CheckpointProvenanceError`.
                        A model trained with a tuned overlay must therefore be evaluated with that overlay.
CLI subcommands (see `main`): `registry`, `run`, `freeze`, `report`; the first three take `--config` and
`--overlay` (default configs/default.yaml, no overlay). `run --split validation --conditions ...` runs the
validation seeds under any of the five shift conditions (or `all`), reusing the same seeds for each; the test
split always runs the frozen pool's five conditions and refuses a pool that lacks one.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from orbit_guard.config import DEFAULT_CONFIG_PATH, REPO_ROOT, config_digest
from orbit_guard.evaluation.baselines import MODEL_NAMES, _model_spec
from orbit_guard.evaluation.runner import EpisodeTask, read_records, run_tasks
from orbit_guard.evaluation.shifts import CONDITION_NAMES
from orbit_guard.evaluation.statistics import (
    PairedComparison,
    Summary,
    degradation,
    paired_comparison,
    pooled_by_training_seed,
)

_DEFAULT_CONFIG = DEFAULT_CONFIG_PATH
_DEFAULT_VALIDATION_POOL = REPO_ROOT / "configs" / "validation_seeds.json"
_DEFAULT_TEST_POOL = REPO_ROOT / "configs" / "test_seeds.json"
# rl/training.py writes this sidecar next to actor_weights.npz; the name is repeated here (and pinned equal by a
# test) because importing rl.training pulls in Ray, which evaluation workers deliberately never need.
RUN_CONFIG_FILENAME = "run_config.json"


class CheckpointProvenanceError(ValueError):
    """A checkpoint's training configuration differs from the configuration it is about to be evaluated with."""


def check_checkpoint_provenance(
    weights_path: str | Path, *, config_path: str | Path = _DEFAULT_CONFIG, overlay_path: str | Path | None = None
) -> None:
    """Refuse `weights_path` if its run_config.json sidecar records a different config_digest.

    Digests, not paths, are compared: an overlay copied elsewhere with identical bytes is the same
    configuration. A checkpoint with no sidecar (exported before sidecars existed) is accepted.
    """
    sidecar = Path(weights_path).parent / RUN_CONFIG_FILENAME
    if not sidecar.exists():
        return
    try:
        recorded = json.loads(sidecar.read_text())
        trained_digest = recorded["config_digest"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise CheckpointProvenanceError(f"{sidecar} is unreadable or has no config_digest ({error!r})") from error
    expected = config_digest(config_path, overlay_path)
    if trained_digest != expected:
        raise CheckpointProvenanceError(
            f"checkpoint {weights_path} was trained under config_digest {trained_digest} "
            f"(config {recorded.get('config_path')!r}, overlay {recorded.get('overlay_path')!r}) but would be "
            f"evaluated under {expected} (config {str(config_path)!r}, overlay "
            f"{None if overlay_path is None else str(overlay_path)!r}); pass the overlay it was trained with"
        )


def check_registry_provenance(
    registry: Mapping[str, Mapping[int, str]],
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
) -> None:
    """`check_checkpoint_provenance` for every registered checkpoint."""
    for seeds in registry.values():
        for weights in seeds.values():
            check_checkpoint_provenance(weights, config_path=config_path, overlay_path=overlay_path)


def build_tasks(
    models: Sequence[str],
    condition_seeds: Mapping[str, Sequence[int]],
    registry: Mapping[str, Mapping[int, str]],
    *,
    n_agents: int,
    experiment_id: str,
    git_commit: str = "unknown",
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
    env_kwargs: Mapping | None = None,
) -> list[EpisodeTask]:
    """Every (model, training seed, condition, scenario seed) task; learned checkpoints must match the config."""
    tasks: list[EpisodeTask] = []
    overlay = None if overlay_path is None else str(overlay_path)
    for model in models:
        learned = _model_spec(model, config_path)["policy"] == "mappo"
        if learned and not registry.get(model):
            raise ValueError(f"{model} is learned: register its trained checkpoints (training seed -> actor weights)")
        if learned:
            check_registry_provenance({model: registry[model]}, config_path=config_path, overlay_path=overlay_path)
        variants = sorted(registry[model].items()) if learned else [(None, None)]
        for training_seed, weights in variants:
            for condition, seeds in condition_seeds.items():
                tasks.extend(
                    EpisodeTask(
                        model=model, condition=condition, scenario_seed=seed, n_agents=n_agents,
                        training_seed=training_seed, actor_weights_path=weights,
                        env_kwargs=dict(env_kwargs or {}), config_path=str(config_path), overlay_path=overlay,
                        experiment_id=experiment_id, git_commit=git_commit,
                    )
                    for seed in seeds
                )
    return tasks


ALL_CONDITIONS = "all"


def validation_condition_seeds(seeds: Sequence[int], conditions: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """The SAME validation seeds under each requested condition (`all` = the five), in canonical order.

    Reusing seeds across conditions is fine on the validation split (it is validation, not test); the final
    test instead draws ten distinct seeds per condition from the frozen pool (PRD §33).
    """
    requested = set(conditions)
    if not requested:
        raise ValueError("choose at least one condition")
    unknown = sorted(requested - set(CONDITION_NAMES) - {ALL_CONDITIONS})
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; expected names from {CONDITION_NAMES} or {ALL_CONDITIONS!r}")
    chosen = CONDITION_NAMES if ALL_CONDITIONS in requested else [c for c in CONDITION_NAMES if c in requested]
    return {condition: tuple(seeds) for condition in chosen}


def require_all_conditions(condition_seeds: Mapping[str, Sequence[int]]) -> None:
    """Raise ValueError unless the final-test pool holds exactly the five conditions, in canonical order."""
    found = tuple(condition_seeds)
    if found != CONDITION_NAMES:
        missing = sorted(set(CONDITION_NAMES) - set(found))
        extra = sorted(set(found) - set(CONDITION_NAMES))
        raise ValueError(
            f"the final-test pool must hold the conditions {CONDITION_NAMES} in that order; got {found} "
            f"(missing {missing}, unexpected {extra})"
        )


_RUN_VARIANT_TO_MODEL = {
    "adaptive": "adaptive_mappo",
    "fixed": "fixed_margin_mappo",
    "noshield": "adaptive_mappo_no_shield",
}


def build_registry(runs_dir: str | Path) -> dict[str, dict[int, str]]:
    """model -> {training seed -> actor weights}, from `n15_<variant>_s<seed>/actor_weights.npz` run directories."""
    registry: dict[str, dict[int, str]] = {}
    for run in sorted(Path(runs_dir).glob("n15_*_s*")):
        weights = run / "actor_weights.npz"
        match = re.fullmatch(r"n15_(adaptive|fixed|noshield)_s(\d+)", run.name)
        if match and weights.exists():
            registry.setdefault(_RUN_VARIANT_TO_MODEL[match[1]], {})[int(match[2])] = str(weights)
    return registry


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


def _build_parser() -> argparse.ArgumentParser:
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
    run.add_argument(
        "--conditions", nargs="+", default=None, choices=(*CONDITION_NAMES, ALL_CONDITIONS),
        help="validation split only: shift conditions to run, each on the same seeds (default: normal)",
    )
    run.add_argument("--freeze-snapshot", default=None, help="required for --split test")
    run.add_argument("--test-pool", default=str(_DEFAULT_TEST_POOL))
    run.add_argument("--validation-pool", default=str(_DEFAULT_VALIDATION_POOL))

    report = sub.add_parser("report", help="markdown table from an episode log")
    report.add_argument("--episodes", required=True)

    registry_cmd = sub.add_parser("registry", help="write the model -> checkpoint registry from finished runs")
    registry_cmd.add_argument("--runs-dir", required=True)
    registry_cmd.add_argument("--out", required=True)

    freeze = sub.add_parser("freeze", help="snapshot config + checkpoints before the final test")
    freeze.add_argument("--registry", required=True)
    freeze.add_argument("--out", required=True)

    for command in (run, registry_cmd, freeze):
        command.add_argument("--config", default=str(_DEFAULT_CONFIG), help="base config (default: configs/default.yaml)")
        command.add_argument("--overlay", default=None, help="tuned overlay the checkpoints were trained with")
    return parser


def _read_registry(path: str | Path) -> dict[str, dict[int, str]]:
    raw = json.loads(Path(path).read_text())
    return {model: {int(seed): weights for seed, weights in seeds.items()} for model, seeds in raw.items()}


def _flat_checkpoints(registry: Mapping[str, Mapping[int, str]]) -> dict[str, str]:
    return {f"{model}/{seed}": path for model, seeds in registry.items() for seed, path in seeds.items()}


def _registry_command(args: argparse.Namespace) -> None:
    built = build_registry(args.runs_dir)
    check_registry_provenance(built, config_path=args.config, overlay_path=args.overlay)
    serialized = {model: {str(seed): path for seed, path in seeds.items()} for model, seeds in built.items()}
    Path(args.out).write_text(json.dumps(serialized, indent=2))
    print({model: sorted(seeds) for model, seeds in built.items()})


def _freeze_command(args: argparse.Namespace) -> None:
    from orbit_guard.evaluation.freeze import write_freeze_snapshot

    registry = _read_registry(args.registry)
    check_registry_provenance(registry, config_path=args.config, overlay_path=args.overlay)
    print(json.dumps(write_freeze_snapshot(args.out, _flat_checkpoints(registry), config_path=args.config), indent=2))


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from orbit_guard.evaluation.freeze import _git_commit

    if args.split == "test" and args.conditions is not None:
        parser.error("--conditions applies to --split validation only; the final test runs the frozen pool's five")
    registry = _read_registry(args.registry)
    # Before any seed pool is read: a mismatched checkpoint must not even open the final-test pool.
    check_registry_provenance(registry, config_path=args.config, overlay_path=args.overlay)
    if args.split == "validation":
        from orbit_guard.evaluation.seed_pools import load_seed_pool

        seeds = load_seed_pool(args.validation_pool, "validation")[: args.max_seeds]
        condition_seeds = validation_condition_seeds(seeds, args.conditions or ("normal",))
    else:
        if args.freeze_snapshot is None:
            parser.error("--split test requires --freeze-snapshot (freeze configs and checkpoints first)")
        from orbit_guard.evaluation.test_seed_pool import load_test_pool

        condition_seeds = load_test_pool(args.test_pool, args.freeze_snapshot, _flat_checkpoints(registry), args.config)
        require_all_conditions(condition_seeds)

    tasks = build_tasks(
        args.models, condition_seeds, registry, n_agents=args.n_agents, experiment_id=args.experiment_id,
        git_commit=_git_commit(), config_path=args.config, overlay_path=args.overlay,
    )
    print(f"{len(tasks)} episodes ({args.workers} workers)", flush=True)
    run_tasks(tasks, args.out, workers=args.workers)
    print(format_report(read_records(args.out)))


def main(argv: list[str] | None = None) -> None:
    """CLI: `registry` of exported actors, `run` a sweep, `freeze` before the final test, `report` on the log."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        print(format_report(read_records(args.episodes)))
    elif args.command == "registry":
        _registry_command(args)
    elif args.command == "freeze":
        _freeze_command(args)
    else:
        _run_command(args, parser)


if __name__ == "__main__":
    main()

