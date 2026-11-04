"""Summary statistics for the experiment matrix (PRD §34).

  * `mean_std_ci`          mean, sample standard deviation, t-based confidence interval
  * `degradation`          (shift - normal) / (|normal| + eps)
  * `normal_mean_is_zero`  the zero-normal-mean test the report uses, with the same eps as `degradation`
  * `paired_comparison`    paired difference on matched scenario seeds: mean, CI, Wilcoxon p
  * `holm_adjust`          Holm step-down adjustment of a family of p-values
  * `cluster_bootstrap_ci` two-stage cluster bootstrap of the mean paired difference: scenario seeds are
                           resampled, and within each resampled scenario the training seeds (learned models)
                           are resampled. Post hoc sensitivity check (confirmatory.md), not a headline test.
  * `pooled_by_training_seed`  average episodes within each training seed first, then summarize
                           ACROSS training seeds -- a single training run is never the headline
                           (PRD §34), and episodes from one seed are not independent samples of
                           the training procedure.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

DEGRADATION_EPSILON = 1.0e-9
BOOTSTRAP_RESAMPLES = 10_000  # cluster_bootstrap_ci default; confirmatory.md states B
BOOTSTRAP_SEED = 20261104  # numpy default_rng seed for the bootstrap; declared with the confirmatory families


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairedComparison:
    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float


def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> Summary:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("mean_std_ci needs at least one value")
    mean = float(data.mean())
    if data.size == 1:
        return Summary(1, mean, 0.0, mean, mean)
    std = float(data.std(ddof=1))
    if std == 0.0:
        return Summary(int(data.size), mean, 0.0, mean, mean)
    half_width = float(stats.t.ppf(0.5 + confidence / 2.0, df=data.size - 1) * std / np.sqrt(data.size))
    return Summary(int(data.size), mean, std, mean - half_width, mean + half_width)


def degradation(metric_shift: float, metric_normal: float, epsilon: float = DEGRADATION_EPSILON) -> float:
    """(Metric_shift - Metric_normal) / (|Metric_normal| + eps); sign convention is the caller's per metric."""
    return (metric_shift - metric_normal) / (abs(metric_normal) + epsilon)


def normal_mean_is_zero(normal_mean: float) -> bool:
    """True when the normal condition's mean is zero for reporting: the ratio is then undefined (~1/eps), so the
    report shows the absolute change instead. Shares `DEGRADATION_EPSILON` so the report and the ratio never disagree."""
    return abs(normal_mean) < DEGRADATION_EPSILON


def paired_comparison(method: Sequence[float], baseline: Sequence[float], confidence: float = 0.95) -> PairedComparison:
    """`method - baseline` on matched scenario seeds."""
    if len(method) != len(baseline):
        raise ValueError(f"paired samples need equal length, got {len(method)} and {len(baseline)}")
    differences = np.asarray(method, dtype=float) - np.asarray(baseline, dtype=float)
    summary = mean_std_ci(differences, confidence)
    if not np.any(differences):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(differences, zero_method="wilcox").pvalue)
    return PairedComparison(summary.n, summary.mean, summary.ci_low, summary.ci_high, p_value)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order: max over k <= rank of (m - k) * p_(k), capped at 1.

    A non-finite p (NaN from a NaN metric) is no test: it stays NaN and is not counted in the family size m.
    """
    finite = [i for i, p in enumerate(p_values) if np.isfinite(p)]
    order = sorted(finite, key=lambda i: p_values[i])
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def _resampled_scenario_means(
    values_by_scenario: Sequence[Sequence[float]], drawn: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Stage 2: for each drawn scenario position, the mean of a fresh with-replacement resample of its training seeds.

    `drawn` is (resamples, positions) of indices into `values_by_scenario`; the result has the same shape. Every
    scenario is resampled for all positions and the drawn ones are selected, so the draws stay vectorized.
    """
    resamples, positions = drawn.shape
    within = np.empty((len(values_by_scenario), resamples, positions))
    for index, values in enumerate(values_by_scenario):
        data = np.asarray(values, dtype=float)
        picks = rng.integers(0, data.size, size=(resamples, positions, data.size))
        within[index] = data[picks].mean(axis=2)
    return np.take_along_axis(within, drawn[np.newaxis], axis=0)[0]


def cluster_bootstrap_ci(
    method_by_scenario: Mapping[int, Sequence[float]],
    baseline_by_scenario: Mapping[int, Sequence[float]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI of the mean paired difference (method - baseline) under a two-stage cluster bootstrap.

    Values are grouped by scenario seed (lists over training seeds; a non-learned model has one value each). Each
    resample draws the shared scenario seeds with replacement, and within every drawn scenario draws each model's
    training seeds with replacement and averages them; the statistic is the mean over the drawn scenarios of the
    two models' difference of means. Only scenario seeds both models ran enter. Deterministic for a fixed `seed`.
    """
    scenarios = sorted(set(method_by_scenario) & set(baseline_by_scenario))
    if not scenarios:
        raise ValueError("cluster bootstrap needs at least one scenario seed that both models ran")
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(scenarios), size=(n_resamples, len(scenarios)))
    method_means = _resampled_scenario_means([method_by_scenario[s] for s in scenarios], drawn, rng)
    baseline_means = _resampled_scenario_means([baseline_by_scenario[s] for s in scenarios], drawn, rng)
    statistic = (method_means - baseline_means).mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(statistic, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def pooled_by_training_seed(records: Sequence[Mapping], key: str, seed_key: str = "training_seed") -> Summary:
    """Per-training-seed means of `key`, summarized across training seeds."""
    by_seed: dict = defaultdict(list)
    for record in records:
        by_seed[record[seed_key]].append(float(record[key]))
    return mean_std_ci([float(np.mean(values)) for values in by_seed.values()])
