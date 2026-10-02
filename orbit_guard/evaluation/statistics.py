"""Summary statistics for the experiment matrix (PRD §34).

  * `mean_std_ci`          mean, sample standard deviation, t-based confidence interval
  * `degradation`          (shift - normal) / (|normal| + eps)
  * `paired_comparison`    paired difference on matched scenario seeds: mean, CI, Wilcoxon p
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

_DEGRADATION_EPSILON = 1.0e-9


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


def degradation(metric_shift: float, metric_normal: float, epsilon: float = _DEGRADATION_EPSILON) -> float:
    """(Metric_shift - Metric_normal) / (|Metric_normal| + eps); sign convention is the caller's per metric."""
    return (metric_shift - metric_normal) / (abs(metric_normal) + epsilon)


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


def pooled_by_training_seed(records: Sequence[Mapping], key: str, seed_key: str = "training_seed") -> Summary:
    """Per-training-seed means of `key`, summarized across training seeds."""
    by_seed: dict = defaultdict(list)
    for record in records:
        by_seed[record[seed_key]].append(float(record[key]))
    return mean_std_ci([float(np.mean(values)) for values in by_seed.values()])
