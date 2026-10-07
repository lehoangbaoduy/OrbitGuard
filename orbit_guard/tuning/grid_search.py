"""Coarse grid search of reward weights and adaptive-margin coefficients (PRD §25, §33; P4-M5).

VALIDATION SPLIT ONLY. This module loads the validation seed pool through
`evaluation.seed_pools.load_seed_pool(..., "validation")` (which refuses any
other split) and never imports, names or reads a final-test pool;
tests/tuning/test_grid_search.py enforces that structurally (no final-test
pool module in the import closure, no reference to one in any tuning source).

The ranking score is fixed BEFORE running any grid, and deliberately weight-free
(raw returns are not comparable across reward weightings): the §4 safety
hierarchy puts collisions first and fuel second,

    score = collision_episodes / episodes + 0.25 * mean_delta_v_used / budget      (lower is better)

so one avoided collision episode in ten (0.1) outweighs spending an extra 40%
of the Δv budget (0.1). The winner is frozen into an overlay file whose hash
covers `configs/default.yaml` plus the overrides, snapshotted before any
final-test evaluation (PRD §33, "freeze tuned params before final test").
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from orbit_guard.evaluation.seed_pools import load_seed_pool
from orbit_guard.rl.evaluation import EvaluationReport

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"
_VALIDATION_CONFIG = _CONFIG_DIR / "validation.yaml"
_FUEL_WEIGHT = 0.25


@dataclass(frozen=True)
class GridPoint:
    """One configuration: overrides on top of `configs/default.yaml`."""

    name: str
    reward_weights: dict[str, float] = field(default_factory=dict)
    margin: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GridResult:
    point: GridPoint
    report: EvaluationReport
    score: float


def default_grid() -> list[GridPoint]:
    """Eight coarse points: 3 margin shapes x 2 coordination weights, plus 2 Δv-cost variants."""
    narrow = {"alpha": 0.5, "beta": 0.25, "gamma": 0.5}
    wide = {"alpha": 2.0, "beta": 1.0, "gamma": 2.0}
    return [
        GridPoint("default"),
        GridPoint("margin_narrow", margin=narrow),
        GridPoint("margin_wide", margin=wide),
        GridPoint("w4_1.5", reward_weights={"w4": 1.5}),
        GridPoint("margin_narrow_w4_1.5", reward_weights={"w4": 1.5}, margin=narrow),
        GridPoint("margin_wide_w4_1.5", reward_weights={"w4": 1.5}, margin=wide),
        GridPoint("w3_2.5", reward_weights={"w3": 2.5}),
        GridPoint("w3_10", reward_weights={"w3": 10.0}),
    ]


def validation_score(report: EvaluationReport, delta_v_budget_mps: float) -> float:
    return report.collision_episodes / report.episodes + _FUEL_WEIGHT * report.mean_delta_v_used_mps / delta_v_budget_mps


def _logged_reports(log_path: Path, settings: dict | None, episodes: int) -> dict[str, EvaluationReport]:
    """Reports already in the grid log for these settings and this many validation episodes (latest per point).

    A line torn by a killed process, or one written under other settings, is ignored: that point simply runs again.
    """
    if not log_path.is_file():
        return {}
    fields = {f.name for f in dataclasses.fields(EvaluationReport)}
    found: dict[str, EvaluationReport] = {}
    for line in log_path.read_text().splitlines():
        try:
            record = json.loads(line)
            if record.get("settings") != settings or record.get("episodes") != episodes:
                continue
            found[record["point"]] = EvaluationReport(**{key: value for key, value in record.items() if key in fields})
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            continue
    return found


def run_grid(
    points: Sequence[GridPoint],
    run_point: Callable[[GridPoint, Sequence[int]], EvaluationReport],
    delta_v_budget_mps: float,
    *,
    validation_config: str | Path = _VALIDATION_CONFIG,
    max_seeds: int | None = None,
    log_path: str | Path | None = None,
    settings: dict | None = None,
    resume: bool = False,
) -> list[GridResult]:
    """Evaluate every point on the validation pool (or its first `max_seeds`); best score first.

    Each finished point is logged with the run `settings`. With `resume`, points already logged under identical
    settings (and the same episode count) are not run again but still take part in the ranking, so a killed grid
    continues where it stopped instead of starting over.
    """
    config = yaml.safe_load(Path(validation_config).read_text())
    pool_path = Path(config["seed_pool_file"])
    if not pool_path.is_absolute():
        pool_path = _CONFIG_DIR.parent / pool_path
    seeds = load_seed_pool(pool_path, "validation", config.get("seed_pool_sha256"))
    if max_seeds is not None:
        seeds = seeds[:max_seeds]

    logged = _logged_reports(Path(log_path), settings, len(seeds)) if resume and log_path is not None else {}
    results = []
    for point in points:
        if point.name in logged:
            results.append(GridResult(point, logged[point.name], validation_score(logged[point.name], delta_v_budget_mps)))
            continue
        report = run_point(point, seeds)
        result = GridResult(point, report, validation_score(report, delta_v_budget_mps))
        results.append(result)
        if log_path is not None:
            with Path(log_path).open("a") as handle:
                handle.write(json.dumps({"point": point.name, "score": result.score, "settings": settings, **report.__dict__}) + "\n")
    return sorted(results, key=lambda result: (result.score, result.point.name))


def freeze_best(results: Sequence[GridResult], path: str | Path) -> str:
    """Write the winner as a hashed config overlay; return the hash."""
    best = results[0]
    body = {
        "selected": best.point.name,
        "selected_on": "validation",
        "reward": dict(best.point.reward_weights),
        "adaptive_margin": dict(best.point.margin),
        "validation": {
            "episodes": best.report.episodes,
            "collision_episodes": best.report.collision_episodes,
            "mean_delta_v_used_mps": best.report.mean_delta_v_used_mps,
            "score": best.score,
        },
    }
    digest = hashlib.sha256(_DEFAULT_CONFIG.read_bytes() + json.dumps(body, sort_keys=True).encode()).hexdigest()
    Path(path).write_text(yaml.safe_dump({**body, "config_hash": digest}, sort_keys=True))
    return digest


def main(argv: list[str] | None = None) -> None:
    """CLI: train + validate every grid point (sequentially), log each, freeze the winner."""
    import argparse

    from orbit_guard.rl.evaluation import evaluate
    from orbit_guard.rl.training import (
        build_algo,
        load_margin_fn,
        load_reward_weights,
        load_training_settings,
        make_env_factory,
        train,
    )

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--agents", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--num-env-runners", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shield", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these grid point names")
    parser.add_argument("--out-dir", default="grid_runs")
    parser.add_argument("--resume", action="store_true", help="skip points already in the log under the same settings (a killed run continues)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = [p for p in default_grid() if args.only is None or p.name in args.only]

    def run_point(point: GridPoint, seeds: Sequence[int]) -> EvaluationReport:
        reward_weights = load_reward_weights(**point.reward_weights)
        factory = make_env_factory(args.agents, reward_weights, args.shield, load_margin_fn(**point.margin))
        settings = load_training_settings(num_env_runners=args.num_env_runners, seed=args.seed)
        print(f"[grid] {point.name}: training {args.iterations} iterations", flush=True)
        algo = build_algo(factory, settings)
        try:
            train(algo, args.iterations)
            report = evaluate(algo, factory, len(seeds), seeds=seeds)
        finally:
            algo.stop()
        print(f"[grid] {point.name}: {dataclasses.asdict(report)}", flush=True)
        return report

    probe = make_env_factory(args.agents, load_reward_weights(), args.shield)({})
    results = run_grid(
        points, run_point, probe.delta_v_budget_mps,
        max_seeds=args.validation_episodes, log_path=out_dir / "grid_log.jsonl", resume=args.resume,
        settings={"agents": args.agents, "iterations": args.iterations, "validation_episodes": args.validation_episodes, "seed": args.seed, "shield": args.shield},
    )
    for result in results:
        print(f"{result.score:8.4f}  {result.point.name}")
    print("frozen config hash:", freeze_best(results, out_dir / "tuned.yaml"))


if __name__ == "__main__":
    main()
