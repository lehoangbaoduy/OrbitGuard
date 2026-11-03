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
When the winner changes the adaptive-margin coefficients, the overlay also
carries the Fixed-Margin constant re-measured under them by the Q30 protocol
(`tuning.fixed_margin`), so the Fixed-Margin arm keeps the adaptive arm's mean
margin size. The grid itself trains only the adaptive arm.

The CLI also logs the env-runner count and the code stamp (`tuning.code_state`: git HEAD, dirty flag, hash of
the uncommitted diff), so a resumed grid never ranks points trained with other runners or other code. It ranks
the union of this invocation's points and every other default_grid() point logged under the same settings,
and freezes only when all of them have a result, so a grid split over `--only` runs freezes the global winner.
Rates are over finished episodes; a point with failed episodes never wins while a clean point exists.
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
from orbit_guard.tuning.code_state import code_state
from orbit_guard.tuning.fixed_margin import measure_fixed_margin, round_fixed_margin

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
    """The pre-registered grid score of `report` under `rule` ('q27' or 'extended'; see the module docstring).

    Rates are over the episodes that finished (`ok_episodes`): a failed episode is neither a collision nor a safe
    one. A report none of whose episodes finished scores +inf (ranked last, never frozen) instead of raising, so the
    grid still logs it and `--resume` does not retry it forever.
    """
    _require_score_rule(rule)
    finished = report.ok_episodes
    if finished < 1:
        return math.inf
    fraction = report.mean_budget_fraction_used
    if not math.isfinite(fraction):
        raise ValueError(f"report has no finite mean_budget_fraction_used ({fraction}); it cannot be scored")
    score = report.collision_episodes / finished + _FUEL_WEIGHT * fraction
    if rule == "extended":
        score += _UNSAFE_WEIGHT * report.unsafe_episodes / finished
    return score


def rank_results(results: Sequence[GridResult]) -> list[GridResult]:
    """Best first: points with failed episodes go after every clean point, unless every point has failures.

    Within each group the order is (score, name). The first element is the winner `freeze_best` freezes.
    """
    every_point_failed = all(result.report.failed_episodes > 0 for result in results)

    def key(result: GridResult) -> tuple:
        demoted = result.report.failed_episodes > 0 and not every_point_failed
        return (demoted, result.score, result.point.name)

    return sorted(results, key=key)


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
    rank_logged: Sequence[GridPoint] = (),
) -> list[GridResult]:
    """Evaluate every point on the validation pool (or its first `max_seeds`); best score first.

    Each finished point is logged with the run `settings`, to which the `score_rule`, the `config_digest` of
    `base_config` (the config `run_point` trains under; `factory_for_point` defaults to the same file) and the
    sha256 of the validation seed-pool file are always added. With `resume`, points already logged under
    identical settings and the same episode count are not run again but still take part in the ranking, so a
    killed grid continues where it stopped instead of starting over, and never ranks together points scored
    under different rules, base configurations or seed pools.

    `rank_logged` points are never run: each one already in the log under the same settings and episode count
    (whether or not `resume` is set) joins the ranking, so a grid split over several `--only` invocations ranks
    the union. Ranking is `rank_results` (points with failed episodes after every clean point).
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

    logged = _logged_reports(Path(log_path), settings, len(seeds)) if log_path is not None else {}
    to_run = {point.name for point in points}
    results = [
        GridResult(point, logged[point.name], validation_score(logged[point.name], score_rule), score_rule)
        for point in rank_logged
        if point.name in logged and point.name not in to_run
    ]
    for point in points:
        if resume and point.name in logged:
            report = logged[point.name]
            results.append(GridResult(point, report, validation_score(report, score_rule), score_rule))
            continue
        report = run_point(point, seeds)
        result = GridResult(point, report, validation_score(report, score_rule), score_rule)
        results.append(result)
        if log_path is not None:
            with Path(log_path).open("a") as handle:
                handle.write(json.dumps({"point": point.name, "score": result.score, "settings": settings, **report.__dict__}) + "\n")
    return rank_results(results)


def freeze_best(
    results: Sequence[GridResult],
    path: str | Path,
    *,
    measure: Callable[[Mapping], float] = measure_fixed_margin,
) -> str:
    """Write the winner as a hashed config overlay; return the hash.

    If the winner's adaptive coefficients differ from the default config's, `fixed_margin.d_safe_m` is set to
    `measure(merged config)` (the Q30 mean adaptive d_safe, `tuning.fixed_margin`) rounded to the nearest whole
    metre like the default constant, before hashing. With the default coefficients the block is left out: the
    default constant stands (and equals what the protocol measures, tested).

    The winner is `rank_results(results)[0]`, so a point with failed episodes never wins while a clean one exists.
    Refuses (ValueError, nothing written) when no point finished a single validation episode.
    """
    best = rank_results(results)[0]
    if best.report.ok_episodes < 1:
        raise ValueError("no grid point finished a single validation episode; refusing to freeze a winner")
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
            "failed_episodes": best.report.failed_episodes,
            "score_rule": best.score_rule,
            "score": best.score,
        },
    }
    fixed = _rematched_fixed_margin(best.point, measure)
    if fixed is not None:
        body["fixed_margin"] = {"d_safe_m": fixed}
    digest = overlay_digest(_DEFAULT_CONFIG.read_bytes(), body)
    Path(path).write_text(yaml.safe_dump({**body, "config_hash": digest}, sort_keys=True))
    return digest


def _rematched_fixed_margin(point: GridPoint, measure: Callable[[Mapping], float]) -> float | None:
    """The Fixed-Margin constant matched to `point`'s adaptive coefficients; None when they are the defaults."""
    base = load_config(_DEFAULT_CONFIG)
    merged = config_for_point(point, base)
    if margin_from_config(merged, "adaptive") == margin_from_config(base, "adaptive"):
        return None
    return round_fixed_margin(measure(merged))


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

    code = code_state()
    if code["git_commit"] is None or code["git_dirty"]:
        print(f"[grid] warning: code stamp {code}; commit before a real grid so its log is tied to one commit", flush=True)
    results = run_grid(
        points, run_point, args.score,
        max_seeds=args.validation_episodes, log_path=out_dir / "grid_log.jsonl", resume=args.resume,
        settings={"agents": args.agents, "iterations": args.iterations, "validation_episodes": args.validation_episodes, "seed": args.seed, "shield": args.shield,
                  "critic_input": critic_input, "num_env_runners": args.num_env_runners, **code},
        rank_logged=default_grid(),
    )
    for result in results:
        failed = f"  ({result.report.failed_episodes} failed episodes)" if result.report.failed_episodes else ""
        print(f"{result.score:8.4f}  {result.point.name}{failed}")
    _freeze_if_complete(results, out_dir / "tuned.yaml")


def _freeze_if_complete(results: Sequence[GridResult], path: Path) -> None:
    """Freeze the winner only when every default_grid() point has a result under these settings; else say why not.

    Exits normally either way (see DECISION gap/gridrobust): a grid split over several --only invocations is
    incomplete after all but the last, and a non-zero code would stop a `set -e` driver between the halves.
    """
    have = {result.point.name for result in results}
    missing = [point.name for point in default_grid() if point.name not in have]
    if missing:
        print(
            f"[grid] not freezing: {len(missing)} of {len(default_grid())} grid points have no result under these "
            f"settings: {', '.join(missing)}. {path} is left as it was; run them (e.g. --only ... --resume).",
            flush=True,
        )
        return
    print("frozen config hash:", freeze_best(results, path))


if __name__ == "__main__":
    main()
