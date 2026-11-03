"""Coarse grid search of reward weights and adaptive-margin coefficients (PRD §25, §33; P4-M5).

VALIDATION SPLIT ONLY. This module loads the validation seed pool through
`evaluation.seed_pools.load_seed_pool(..., "validation")` (which refuses any
other split) and never imports, names or reads a final-test pool;
tests/tuning/test_grid_search.py enforces that structurally (no final-test
pool module in the import closure, no reference to one in any tuning source).

The ranking score is fixed BEFORE running any grid, and deliberately weight-free
(raw returns are not comparable across reward weightings): the §4 safety
hierarchy puts collisions first and fuel second. Two pre-registered rules exist
(lower is better in both; `fraction` is the mean fraction of each satellite's OWN
initial Δv budget used, since budgets differ per satellite -- CORE-3):

    q27       score = collision_episodes / episodes + 0.25 * fraction
    extended  score = collision_episodes / episodes + 0.5 * unsafe_episodes / episodes + 0.25 * fraction

`q27` (Q27, the default) means one avoided collision episode in ten (0.1)
outweighs spending an extra 40% of the budget (0.1). `extended` is the R3 rule
of the pre-registered compute plan, used when the No-Maneuver collision-episode
rate on validation is too low (< 0.10) for collisions alone to rank grid points;
an unsafe episode is one with a non-colliding pair inside the common adaptive
reference margin (PRD §36 #2, Q28). The rule is part of the logged run settings,
together with the base config's `config_digest` and the validation pool file's
sha256, so a resumed grid never mixes rules, configurations or seed pools.
The winner is frozen into an overlay file whose hash
covers `configs/default.yaml` plus the overrides, snapshotted before any
final-test evaluation (PRD §33, "freeze tuned params before final test").
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from orbit_guard.config import (
    DEFAULT_CONFIG_PATH,
    apply_overlay,
    config_digest,
    load_config,
    margin_from_config,
    overlay_digest,
)
from orbit_guard.evaluation.seed_pools import load_seed_pool
from orbit_guard.rl.ctde_module import CRITIC_INPUTS
from orbit_guard.rl.env_factory import EnvFactory, make_env_factory_from_config
from orbit_guard.rl.evaluation import EvaluationReport

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_DEFAULT_CONFIG = DEFAULT_CONFIG_PATH
_VALIDATION_CONFIG = _CONFIG_DIR / "validation.yaml"
_FUEL_WEIGHT = 0.25
_UNSAFE_WEIGHT = 0.5
SCORE_RULES = ("q27", "extended")
DEFAULT_SCORE_RULE = "q27"


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
    score_rule: str


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


def config_for_point(point: GridPoint, base_config: Mapping | None = None) -> dict:
    """The default configuration with this point's reward and margin overrides applied (a fresh dict)."""
    base = load_config() if base_config is None else base_config
    return apply_overlay(base, {"reward": point.reward_weights, "adaptive_margin": point.margin})


def factory_for_point(point: GridPoint, num_agents: int, *, shielded: bool, base_config: Mapping | None = None) -> EnvFactory:
    """Environment factory for a grid point, built through the same config path as every other run."""
    config = config_for_point(point, base_config)
    return make_env_factory_from_config(
        config, num_agents, shielded=shielded, margin_fn=margin_from_config(config, "adaptive")
    )


def _require_score_rule(rule: str) -> None:
    if rule not in SCORE_RULES:
        raise ValueError(f"unknown score rule {rule!r}; expected one of {list(SCORE_RULES)}")


def validation_score(report: EvaluationReport, rule: str) -> float:
    """The pre-registered grid score of `report` under `rule` ('q27' or 'extended'; see the module docstring)."""
    _require_score_rule(rule)
    fraction = report.mean_budget_fraction_used
    if not math.isfinite(fraction):
        raise ValueError(f"report has no finite mean_budget_fraction_used ({fraction}); it cannot be scored")
    score = report.collision_episodes / report.episodes + _FUEL_WEIGHT * fraction
    if rule == "extended":
        score += _UNSAFE_WEIGHT * report.unsafe_episodes / report.episodes
    return score


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
    score_rule: str,
    *,
    validation_config: str | Path = _VALIDATION_CONFIG,
    max_seeds: int | None = None,
    log_path: str | Path | None = None,
    settings: dict | None = None,
    resume: bool = False,
    base_config: str | Path = _DEFAULT_CONFIG,
) -> list[GridResult]:
    """Evaluate every point on the validation pool (or its first `max_seeds`); best score first.

    Each finished point is logged with the run `settings`, to which the `score_rule`, the `config_digest` of
    `base_config` (the config `run_point` trains under; `factory_for_point` defaults to the same file) and the
    sha256 of the validation seed-pool file are always added. With `resume`, points already logged under
    identical settings and the same episode count are not run again but still take part in the ranking, so a
    killed grid continues where it stopped instead of starting over, and never ranks together points scored
    under different rules, base configurations or seed pools.
    """
    _require_score_rule(score_rule)
    config = yaml.safe_load(Path(validation_config).read_text())
    pool_path = Path(config["seed_pool_file"])
    if not pool_path.is_absolute():
        pool_path = _CONFIG_DIR.parent / pool_path
    seeds = load_seed_pool(pool_path, "validation", config.get("seed_pool_sha256"))
    settings = {
        **(settings or {}),
        "score": score_rule,
        "base_config_digest": config_digest(base_config),
        "validation_pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
    }
    if max_seeds is not None:
        seeds = seeds[:max_seeds]

    logged = _logged_reports(Path(log_path), settings, len(seeds)) if resume and log_path is not None else {}
    results = []
    for point in points:
        if point.name in logged:
            report = logged[point.name]
            results.append(GridResult(point, report, validation_score(report, score_rule), score_rule))
            continue
        report = run_point(point, seeds)
        result = GridResult(point, report, validation_score(report, score_rule), score_rule)
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
            "mean_budget_fraction_used": best.report.mean_budget_fraction_used,
            "unsafe_episodes": best.report.unsafe_episodes,
            "score_rule": best.score_rule,
            "score": best.score,
        },
    }
    digest = overlay_digest(_DEFAULT_CONFIG.read_bytes(), body)
    Path(path).write_text(yaml.safe_dump({**body, "config_hash": digest}, sort_keys=True))
    return digest


def build_parser() -> argparse.ArgumentParser:
    """Arguments of the grid CLI (kept Ray-free so they can be tested without training)."""
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
    parser.add_argument(
        "--score", choices=SCORE_RULES, default=DEFAULT_SCORE_RULE,
        help="pre-registered ranking rule: q27 (collisions + fuel) or extended (adds unsafe episodes; prereg R3)",
    )
    parser.add_argument(
        "--critic-input", choices=CRITIC_INPUTS, default=None,
        help="critic input passed to training (prereg R4); default: rl.critic_input of the config",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI: train + validate every grid point (sequentially), log each, freeze the winner."""
    from orbit_guard.rl.evaluation import evaluate
    from orbit_guard.rl.training import build_algo, load_training_settings, train

    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = [p for p in default_grid() if args.only is None or p.name in args.only]

    overrides = {"num_env_runners": args.num_env_runners, "seed": args.seed}
    if args.critic_input is not None:
        overrides["critic_input"] = args.critic_input
    # Resolved once, so the logged settings name the critic actually trained and --resume never mixes variants.
    critic_input = load_training_settings(**overrides).critic_input

    def run_point(point: GridPoint, seeds: Sequence[int]) -> EvaluationReport:
        factory = factory_for_point(point, args.agents, shielded=args.shield)
        settings = load_training_settings(**overrides)
        print(f"[grid] {point.name}: training {args.iterations} iterations", flush=True)
        algo = build_algo(factory, settings)
        try:
            train(algo, args.iterations)
            report = evaluate(algo, factory, len(seeds), seeds=seeds)
        finally:
            algo.stop()
        print(f"[grid] {point.name}: {dataclasses.asdict(report)}", flush=True)
        return report

    results = run_grid(
        points, run_point, args.score,
        max_seeds=args.validation_episodes, log_path=out_dir / "grid_log.jsonl", resume=args.resume,
        settings={"agents": args.agents, "iterations": args.iterations, "validation_episodes": args.validation_episodes, "seed": args.seed, "shield": args.shield,
                  "critic_input": critic_input},
    )
    for result in results:
        print(f"{result.score:8.4f}  {result.point.name}")
    print("frozen config hash:", freeze_best(results, out_dir / "tuned.yaml"))


if __name__ == "__main__":
    main()
