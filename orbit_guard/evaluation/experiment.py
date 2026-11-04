"""Experiment orchestration: task matrix, aggregation, paired report (PRD §33-35).

  * `build_tasks`       model x condition x scenario seed (x training seed) -> `EpisodeTask`s. Every model
                        gets IDENTICAL scenario seeds per condition, which is what makes paired comparison valid.
  * `summarize`         one row per (model, condition): primary endpoint (collision-episode rate) and the
                        secondary endpoints, pooled across training seeds, with degradation vs `normal`.
  * `paired_by_scenario` paired difference between two models on matched scenario seeds.
  * `format_report`     the markdown table: every endpoint, the 95% CIs of the primary endpoint
                        (collision-episode rate) and the unsafe rate, and a failed-episode count per row.
  * `format_paired_report`      method − baseline per condition and metric: n, mean difference, CI, Wilcoxon p.
  * `format_degradation_report` per model and metric, (shift − normal) / (|normal| + ε) for each shift.
  * `check_checkpoint_provenance`  a checkpoint trained under one configuration is never evaluated under
                        another: the run_config.json sidecar that rl/training.py writes next to
                        actor_weights.npz records `config.config_digest(config, overlay)`, and a mismatch
                        with the digest the experiment is about to use raises `CheckpointProvenanceError`.
                        A model trained with a tuned overlay must therefore be evaluated with that overlay.
                        The sidecar's recorded margin and reward weights must also equal the ones the
                        evaluation environment for that model builds from config + overlay
                        (`trained_values`), so training CLI overrides that bypass the digest are refused.
`registry` and `freeze` refuse a registry in which a learned model lacks one of its rl.num_training_seeds training
seeds (`registry_checks`) unless `--allow-partial` is passed; the freeze snapshot records that flag.
CLI subcommands (see `main`): `registry`, `run`, `freeze`, `report`; the first three take `--config` and
`--overlay` (default configs/default.yaml, no overlay). `run --split validation --conditions ...` runs the
validation seeds under any of the five shift conditions (or `all`), reusing the same seeds for each; the test
split always runs the frozen pool's five conditions and refuses a pool that lacks one. `run --step-logs` /
`--no-step-logs` writes the PRD §40 per-control-step log of every episode to <out dir>/steps (`step_log_dir_for`);
the default is on for the test split and off for validation, which stays cheap.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from orbit_guard.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    config_digest,
    load_config,
)
from orbit_guard.evaluation.baselines import MODEL_NAMES, _model_spec, _spec_from_config
from orbit_guard.evaluation.registry_checks import (
    ALLOW_PARTIAL_FLAG,
    check_registry_complete,
)
from orbit_guard.evaluation.runner import EpisodeTask, read_records, run_tasks
from orbit_guard.evaluation.shifts import CONDITION_NAMES
from orbit_guard.evaluation.statistics import (
    PairedComparison,
    Summary,
    degradation,
    holm_adjust,
    paired_comparison,
    pooled_by_training_seed,
)
from orbit_guard.evaluation.trained_values import (
    OVERLAY_ADVICE,
    RUN_SHAPE_ADVICE,
    evaluation_values,
    recorded_margin_kind,
    run_shape_mismatches,
    trained_value_mismatches,
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
    weights_path: str | Path,
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
    model: str | None = None,
    n_agents: int | None = None,
) -> None:
    """Refuse `weights_path` if its run_config.json sidecar records another config_digest, margin, reward weights,
    shield setting or constellation size.

    Digests, not paths, are compared: an overlay copied elsewhere with identical bytes is the same
    configuration. The recorded `margin` and `reward_weights` must equal what the evaluation environment for
    `model` builds from config + overlay (its `models:` margin kind; without a model, the kind the sidecar
    records); a sidecar missing either is refused. The recorded `shield` must equal `model`'s spec and the recorded
    `num_agents` must equal `n_agents`; both keys must be present (and a bool / an int) even when `model` or
    `n_agents` is not given. A checkpoint with no sidecar (exported before sidecars existed) is accepted.
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
    config = load_config(config_path, overlay_path)
    _check_run_shape(weights_path, recorded, config, model, n_agents)
    _check_trained_values(weights_path, recorded, config, model)


def _check_run_shape(
    weights_path: str | Path, recorded: Mapping, config: Mapping, model: str | None, n_agents: int | None
) -> None:
    """Refuse a checkpoint trained with another shield setting than `model`'s or at another N than `n_agents`."""
    shield = None if model is None else bool(_spec_from_config(model, config)["shield"])
    problems = run_shape_mismatches(recorded, shield=shield, num_agents=n_agents, model=model)
    if problems:
        raise CheckpointProvenanceError(
            f"checkpoint {weights_path} does not match the model or constellation it would be evaluated as: "
            f"{'; '.join(problems)}. {RUN_SHAPE_ADVICE}"
        )


def _check_trained_values(weights_path: str | Path, recorded: Mapping, config: Mapping, model: str | None) -> None:
    """Refuse a checkpoint trained with a margin or reward weights its evaluation environment would not use."""
    kind = recorded_margin_kind(recorded) if model is None else _spec_from_config(model, config)["margin"]
    problems = trained_value_mismatches(recorded, evaluation_values(config, kind))
    if problems:
        target = f"model {model!r}" if model is not None else f"its recorded {kind} margin"
        raise CheckpointProvenanceError(
            f"checkpoint {weights_path} was not trained with the margin and reward weights the evaluation "
            f"environment for {target} builds from config + overlay: {'; '.join(problems)}. {OVERLAY_ADVICE}"
        )


def check_registry_provenance(
    registry: Mapping[str, Mapping[int, str]],
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    overlay_path: str | Path | None = None,
    n_agents: int | None = None,
) -> None:
    """`check_checkpoint_provenance` for every registered checkpoint, against the model it is registered as and the
    constellation size `n_agents` it will run at."""
    for model, seeds in registry.items():
        for weights in seeds.values():
            check_checkpoint_provenance(
                weights, config_path=config_path, overlay_path=overlay_path, model=model, n_agents=n_agents
            )


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
    step_log_dir: str | None = None,
) -> list[EpisodeTask]:
    """Every (model, training seed, condition, scenario seed) task; learned checkpoints must match the config."""
    tasks: list[EpisodeTask] = []
    overlay = None if overlay_path is None else str(overlay_path)
    for model in models:
        learned = _model_spec(model, config_path)["policy"] == "mappo"
        if learned and not registry.get(model):
            raise ValueError(f"{model} is learned: register its trained checkpoints (training seed -> actor weights)")
        if learned:
            check_registry_provenance(
                {model: registry[model]}, config_path=config_path, overlay_path=overlay_path, n_agents=n_agents
            )
        variants = sorted(registry[model].items()) if learned else [(None, None)]
        for training_seed, weights in variants:
            for condition, seeds in condition_seeds.items():
                tasks.extend(
                    EpisodeTask(
                        model=model, condition=condition, scenario_seed=seed, n_agents=n_agents,
                        training_seed=training_seed, actor_weights_path=weights,
                        env_kwargs=dict(env_kwargs or {}), config_path=str(config_path), overlay_path=overlay,
                        experiment_id=experiment_id, git_commit=git_commit, step_log_dir=step_log_dir,
                    )
                    for seed in seeds
                )
    return tasks


ALL_CONDITIONS = "all"
STEP_LOG_DIRNAME = "steps"


def step_log_dir_for(split: str, episode_log_path: str | Path, enabled: bool | None) -> str | None:
    """Where `run` writes per-step logs: <episode log dir>/steps, by default only for the test split.

    `enabled` is the --step-logs/--no-step-logs flag (None = not given). Returned as a plain string because
    tasks travel to spawn workers.
    """
    if not (split == "test" if enabled is None else enabled):
        return None
    return str(Path(episode_log_path).parent / STEP_LOG_DIRNAME)


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


DEGRADATION_METRICS = ("collision_rate", "unsafe_rate", "total_delta_v_mps")
RATE_METRICS = ("collision_rate", "unsafe_rate")  # per-episode 0/1: CIs are clipped to the rate's range
PAIRED_BASELINES = ("fixed_margin_mappo", "adaptive_mappo_no_shield", "rule_based", "no_maneuver")
PAIRED_METRICS = ("collision_rate", "unsafe_rate", "total_delta_v_mps", "mean_mission_deviation")


def _is_ok(record: Mapping) -> bool:
    return record.get("status", "ok") == "ok"


def _model_order(model: str) -> int:
    return MODEL_NAMES.index(model) if model in MODEL_NAMES else len(MODEL_NAMES)


def _condition_order(condition: str) -> int:
    return CONDITION_NAMES.index(condition) if condition in CONDITION_NAMES else len(CONDITION_NAMES)


def summarize(records: Sequence[Mapping]) -> list[dict]:
    """One row per (model, condition); metrics are `Summary`s pooled across training seeds.

    Failed episodes are excluded from the metrics but counted in `failed`; a (model, condition) whose every
    episode failed still gets a row, with `episodes == 0` and every metric None, so it is never silently lost.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    failed: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        key = (record["model_name"], record["shift_type"])
        if _is_ok(record):
            groups[key].append({"training_seed": record["training_seed"], **_derived(record)})
        else:
            failed[key] += 1

    rows = []
    for model, condition in set(groups) | set(failed):
        episodes = groups.get((model, condition), [])
        row: dict = {
            "model": model, "condition": condition, "episodes": len(episodes),
            "failed": failed.get((model, condition), 0),
            "training_seeds": len({e["training_seed"] for e in episodes}),
        }
        for metric in _METRICS:
            row[metric] = pooled_by_training_seed(episodes, metric) if episodes else None
        rows.append(row)

    normal = {row["model"]: row for row in rows if row["condition"] == "normal"}
    for row in rows:
        base = normal.get(row["model"])
        for metric in DEGRADATION_METRICS:
            usable = base is not None and base[metric] is not None and row[metric] is not None
            row[f"{metric}_degradation"] = (
                degradation(row[metric].mean, base[metric].mean) if usable else float("nan")
            )
    return sorted(rows, key=lambda r: (_condition_order(r["condition"]), r["condition"], _model_order(r["model"])))


def _metric_value(record: Mapping, metric: str) -> float:
    """A logged field (e.g. `collision_count`) or a derived per-episode value (e.g. `collision_rate`, 0/1)."""
    derived = _derived(record)
    return float(derived[metric] if metric in derived else record[metric])


def _per_scenario(records: Sequence[Mapping], model: str, condition: str, metric: str) -> dict[int, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record["model_name"] == model and record["shift_type"] == condition and _is_ok(record):
            by_seed[record["scenario_seed"]].append(_metric_value(record, metric))
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


_MISSING = "—"


def _cell(summary: Summary | None, digits: int = 3) -> str:
    return _MISSING if summary is None else f"{summary.mean:.{digits}f} ± {summary.std:.{digits}f}"


_RATE_RANGE = (0.0, 1.0)
_NO_SPREAD = 1.0e-12


def _ci_cell(summary: Summary | None, digits: int = 3, bounds: tuple[float, float] | None = None) -> str:
    """The t-interval across training seeds; `bounds` clips it to the metric's range (a rate lies in [0, 1])."""
    if summary is None:
        return _MISSING
    if summary.n < 2:
        return f"n/a ({summary.n} seed)"
    if summary.std <= _NO_SPREAD * max(1.0, abs(summary.mean)):  # equal seed means can carry ~1e-16 of float std
        return f"no spread ({summary.n} seeds, all {summary.mean:.{digits}f})"
    low, high = summary.ci_low, summary.ci_high
    if bounds is not None:
        low, high = max(low, bounds[0]), min(high, bounds[1])
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def format_report(records: Sequence[Mapping]) -> str:
    """Per (model, condition): episodes, failed episodes, every endpoint, and 95% CIs of the primary endpoint."""
    rows = summarize(records)
    lines = [
        (
            "| model | condition | episodes | failed | training seeds | collision-episode rate | collision 95% CI "
            "| collision-or-unsafe rate | collision-or-unsafe 95% CI | fleet Δv (m/s) | maneuvers | mission dev. "
            "| shield agent-changes/step | residual risk |"
        ),
        "|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['episodes']} | {row['failed']} | {row['training_seeds']} "
            f"| {_cell(row['collision_rate'])} | {_ci_cell(row['collision_rate'], bounds=_RATE_RANGE)} "
            f"| {_cell(row['unsafe_rate'])} | {_ci_cell(row['unsafe_rate'], bounds=_RATE_RANGE)} "
            f"| {_cell(row['total_delta_v_mps'])} "
            f"| {_cell(row['maneuver_count'], 1)} | {_cell(row['mean_mission_deviation'])} "
            f"| {_cell(row['shield_intervention_rate'], 4)} | {_cell(row['mean_residual_risk'])} |"
        )
    lines += [
        "",
        (
            "Values are mean ± std across training seeds (per-seed means first); non-learned models have one 'seed'. "
        "95% CIs are t-intervals across training seeds, clipped to [0, 1] for the rates; a single seed has none "
        "(n/a), and seeds with identical rates show `no spread` instead of a zero-width interval. An episode counts "
        "as collision-or-unsafe if it had a collision or any pair's closest approach fell inside the "
        "default-config adaptive margin (one yardstick for every model, so a tuned overlay cannot move it; Q33). "
        "Fleet Δv is the episode total over all satellites; shield agent-changes/step is the number of agents "
        "whose action the shield changed, per control step. `episodes` counts ok "
        "episodes; `failed` episodes (PRD §36 #9) are excluded from every metric but always listed here, and a "
        f"row whose episodes all failed shows {_MISSING}."
        ),
    ]
    return "\n".join(lines)


def _paired_result(
    records: Sequence[Mapping], method: str, baseline: str, condition: str, metric: str
) -> PairedComparison | str:
    """The comparison, or the reason it was skipped."""
    try:
        return paired_by_scenario(records, method, baseline, condition, metric)
    except ValueError:
        method_seeds = len(_per_scenario(records, method, condition, metric))
        baseline_seeds = len(_per_scenario(records, baseline, condition, metric))
        return (
            f"skipped: no shared scenario seeds ({method} {method_seeds}, {baseline} {baseline_seeds} "
            "seeds with an ok episode)"
        )


def _paired_row(condition: str, baseline: str, metric: str, result: PairedComparison | str, holm: float | None) -> str:
    if isinstance(result, str):
        return f"| {condition} | {baseline} | {metric} | 0 | {_MISSING} | {_MISSING} | {result} | {_MISSING} |"
    low, high = result.ci_low, result.ci_high
    if metric in RATE_METRICS:  # a difference of two rates lies in [-1, 1]
        low, high = max(low, -1.0), min(high, 1.0)
    interval = f"n/a (n={result.n})" if result.n < 2 else f"[{low:.3f}, {high:.3f}]"
    return (
        f"| {condition} | {baseline} | {metric} | {result.n} | {result.mean_difference:.3f} | {interval} "
        f"| {_p_cell(result.p_value)} | {_MISSING if holm is None or np.isnan(holm) else _p_cell(holm)} |"
    )


def _p_cell(p_value: float) -> str:
    return f"{p_value:.3f}" if p_value >= 0.01 else f"{p_value:.2e}"


def format_paired_report(
    records: Sequence[Mapping],
    method: str = "adaptive_mappo",
    baselines: Sequence[str] = PAIRED_BASELINES,
    metrics: Sequence[str] = PAIRED_METRICS,
) -> str:
    """Paired `method − baseline` comparisons on matched scenario seeds, per condition (PRD §34 headline).

    Per-scenario values are means over training seeds (a learned model's collision_rate per scenario is
    therefore k/3, not 0/1). Pairs that share no scenario seed are listed as skipped, never dropped.
    """
    conditions = sorted(
        {r["shift_type"] for r in records if r["model_name"] == method}, key=lambda c: (_condition_order(c), c)
    )
    lines = [
        f"Paired comparison: {method} − baseline on matched scenario seeds",
        "",
        "| condition | baseline | metric | n | mean difference | 95% CI | Wilcoxon p | Holm p (post hoc) |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    cells = [
        (condition, baseline, metric, _paired_result(records, method, baseline, condition, metric))
        for condition in conditions
        for baseline in (b for b in baselines if b != method)
        for metric in metrics
    ]
    # The family: every row with a test. A skipped row has none, and n = 1 gets p = 1.0 by convention, not a test.
    family = [i for i, (*_, result) in enumerate(cells) if not isinstance(result, str) and result.n >= 2]
    holm = dict(zip(family, holm_adjust([cells[i][3].p_value for i in family]), strict=True))
    for index, (condition, baseline, metric, result) in enumerate(cells):
        lines.append(_paired_row(condition, baseline, metric, result, holm.get(index)))
    if not conditions:
        lines.append(
            f"| {_MISSING} | {_MISSING} | {_MISSING} | 0 | {_MISSING} | {_MISSING} | no {method} episodes | {_MISSING} |"
        )
    lines += [
        "",
        (
            f"Sign: {method} − baseline; for every metric here lower is better, so a negative difference favours "
        f"{method}. n = scenario seeds with an ok episode for both models (failed episodes are excluded); "
        "per-scenario values of learned models are means over their training seeds. 95% CI: t-interval of the "
        "paired differences. Wilcoxon signed-rank (two-sided, zero_method='wilcox'): zero differences are "
        "dropped before ranking and tied |differences| share average ranks; when every difference is zero "
        "there is no evidence either way and p = 1.000. With few non-zero differences p cannot be small "
        "(n = 3 non-zero gives p >= 0.25). Rate-difference CIs are clipped to [-1, 1]. Holm p: Holm step-down "
        "over every row with a test (n >= 2, finite p) in this table, across conditions, baselines and metrics; "
        f"{_MISSING} marks a row outside the family. The family was declared after the final test sweep: the "
        "pre-registered compute plan (R1–R6) named no significance test or multiplicity correction for the "
        "headline comparison, so this column is a robustness check, not a confirmatory test."
        ),
    ]
    return "\n".join(lines)


_DEGRADATION_ZERO = 1.0e-12


def _degradation_cell(row: Mapping | None, base: Mapping | None, metric: str) -> str:
    if row is None or base is None or row[metric] is None or base[metric] is None:
        return _MISSING
    normal_mean, shift_mean = base[metric].mean, row[metric].mean
    if abs(normal_mean) < _DEGRADATION_ZERO:
        return f"n/a (normal = 0; shift {shift_mean:.3f}; change {shift_mean - normal_mean:+.3f})"
    return f"{row[f'{metric}_degradation']:+.3f} ({normal_mean:.3f} → {shift_mean:.3f})"


def format_degradation_report(records: Sequence[Mapping], metrics: Sequence[str] = DEGRADATION_METRICS) -> str:
    """Per model and metric, (shift − normal) / (|normal| + ε) for each shift condition (PRD §34)."""
    rows = {(row["model"], row["condition"]): row for row in summarize(records)}
    models = sorted({model for model, _ in rows}, key=lambda m: (_model_order(m), m))
    shifts = [c for c in CONDITION_NAMES if c != "normal"]
    lines = [
        "Degradation vs normal: (shift − normal) / (|normal| + ε), with (normal → shift) means",
        "",
        "| model | metric | " + " | ".join(shifts) + " |",
        "|---|---|" + "---|" * len(shifts),
    ]
    for model in models:
        base = rows.get((model, "normal"))
        for metric in metrics:
            cells = [_degradation_cell(rows.get((model, shift)), base, metric) for shift in shifts]
            lines.append(f"| {model} | {metric} | " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "Higher is worse for every metric here, so a positive degradation is a deterioration under the shift. "
        "Means are pooled across training seeds. When the normal mean is 0 the ratio is undefined (ε = 1e-9 "
        f"would print ~1e8), so the cell shows n/a with the shift mean and the absolute change; {_MISSING} means a "
        "condition was not run "
        "or all its episodes failed."
        ),
    ]
    return "\n".join(lines)


def format_all_reports(records: Sequence[Mapping]) -> str:
    """The results table, the paired comparisons and the degradation table, in that order."""
    return "\n\n".join((format_report(records), format_paired_report(records), format_degradation_report(records)))


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
    run.add_argument(
        "--step-logs", action=argparse.BooleanOptionalAction, default=None,
        help="write each episode's PRD §40 per-step log to <out dir>/steps (default: on for test, off for validation)",
    )
    run.add_argument(
        "--retry-failed", action="store_true", help="rerun episodes whose latest logged record has status 'failed'"
    )

    report = sub.add_parser("report", help="results, paired-comparison and degradation tables from an episode log")
    report.add_argument("--episodes", required=True)

    registry_cmd = sub.add_parser("registry", help="write the model -> checkpoint registry from finished runs")
    registry_cmd.add_argument("--runs-dir", required=True)
    registry_cmd.add_argument("--out", required=True)

    freeze = sub.add_parser("freeze", help="snapshot config + checkpoints before the final test")
    freeze.add_argument("--registry", required=True)
    freeze.add_argument("--out", required=True)

    for command in (registry_cmd, freeze):
        command.add_argument(
            "--n-agents", type=int, default=15, help="constellation size the checkpoints must have been trained at",
        )
        command.add_argument(
            ALLOW_PARTIAL_FLAG, action="store_true",
            help="accept learned models missing some of their rl.num_training_seeds seeds (recorded in the freeze)",
        )

    for command in (run, registry_cmd, freeze):
        command.add_argument("--config", default=str(_DEFAULT_CONFIG), help="base config (default: configs/default.yaml)")
        command.add_argument("--overlay", default=None, help="tuned overlay the checkpoints were trained with")
    for command in (run, freeze):
        command.add_argument("--test-pool", default=str(_DEFAULT_TEST_POOL))
        command.add_argument("--validation-pool", default=str(_DEFAULT_VALIDATION_POOL))
    return parser


def _read_registry(path: str | Path) -> dict[str, dict[int, str]]:
    raw = json.loads(Path(path).read_text())
    return {model: {int(seed): weights for seed, weights in seeds.items()} for model, seeds in raw.items()}


def _flat_checkpoints(registry: Mapping[str, Mapping[int, str]]) -> dict[str, str]:
    return {f"{model}/{seed}": path for model, seeds in registry.items() for seed, path in seeds.items()}


def _check_complete(registry: Mapping[str, Mapping[int, str]], args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overlay)
    check_registry_complete(registry, config, allow_partial=args.allow_partial)


def _registry_command(args: argparse.Namespace) -> None:
    built = build_registry(args.runs_dir)
    _check_complete(built, args)
    check_registry_provenance(built, config_path=args.config, overlay_path=args.overlay, n_agents=args.n_agents)
    serialized = {model: {str(seed): path for seed, path in seeds.items()} for model, seeds in built.items()}
    Path(args.out).write_text(json.dumps(serialized, indent=2))
    print({model: sorted(seeds) for model, seeds in built.items()})


def _freeze_command(args: argparse.Namespace) -> None:
    from orbit_guard.evaluation.freeze import write_freeze_snapshot

    registry = _read_registry(args.registry)
    _check_complete(registry, args)
    check_registry_provenance(registry, config_path=args.config, overlay_path=args.overlay, n_agents=args.n_agents)
    snapshot = write_freeze_snapshot(
        args.out, _flat_checkpoints(registry), config_path=args.config, overlay_path=args.overlay,
        test_pool_path=args.test_pool, validation_pool_path=args.validation_pool, allow_partial=args.allow_partial,
    )
    print(json.dumps(snapshot, indent=2))


def _test_condition_seeds(args: argparse.Namespace, registry: Mapping[str, Mapping[int, str]]) -> dict:
    """The final-test pool, opened only after the freeze verifies against THIS run's config and overlay."""
    from orbit_guard.evaluation.freeze import verify_freeze
    from orbit_guard.evaluation.test_seed_pool import load_test_pool

    checkpoints = _flat_checkpoints(registry)
    verify_freeze(
        args.freeze_snapshot, checkpoints, args.config, args.overlay,
        test_pool_path=args.test_pool, validation_pool_path=args.validation_pool,
    )
    condition_seeds = load_test_pool(args.test_pool, args.freeze_snapshot, checkpoints, args.config)
    require_all_conditions(condition_seeds)
    return condition_seeds


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from orbit_guard.evaluation.freeze import _git_commit

    if args.split == "test" and args.conditions is not None:
        parser.error("--conditions applies to --split validation only; the final test runs the frozen pool's five")
    registry = _read_registry(args.registry)
    # Before any seed pool is read: a mismatched checkpoint must not even open the final-test pool.
    check_registry_provenance(registry, config_path=args.config, overlay_path=args.overlay, n_agents=args.n_agents)
    if args.split == "validation":
        from orbit_guard.evaluation.seed_pools import load_seed_pool

        seeds = load_seed_pool(args.validation_pool, "validation")[: args.max_seeds]
        condition_seeds = validation_condition_seeds(seeds, args.conditions or ("normal",))
    else:
        if args.freeze_snapshot is None:
            parser.error("--split test requires --freeze-snapshot (freeze configs and checkpoints first)")
        condition_seeds = _test_condition_seeds(args, registry)

    tasks = build_tasks(
        args.models, condition_seeds, registry, n_agents=args.n_agents, experiment_id=args.experiment_id,
        git_commit=_git_commit(), config_path=args.config, overlay_path=args.overlay,
        step_log_dir=step_log_dir_for(args.split, args.out, args.step_logs),
    )
    print(f"{len(tasks)} episodes ({args.workers} workers)", flush=True)
    run_tasks(tasks, args.out, workers=args.workers, retry_failed=args.retry_failed)
    print(format_report(read_records(args.out)))


def main(argv: list[str] | None = None) -> None:
    """CLI: `registry` of exported actors, `run` a sweep, `freeze` before the final test, `report` on the log."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "report":
        print(format_all_reports(read_records(args.episodes)))
    elif args.command == "registry":
        _registry_command(args)
    elif args.command == "freeze":
        _freeze_command(args)
    else:
        _run_command(args, parser)


if __name__ == "__main__":
    main()

