"""Tests for orbit_guard.evaluation.statistics (PRD §34): spread, CIs, paired comparison, degradation."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.evaluation.statistics import (
    cluster_bootstrap_ci,
    degradation,
    mean_std_ci,
    normal_mean_is_zero,
    paired_comparison,
    pooled_by_training_seed,
)


def test_mean_std_and_a_t_based_confidence_interval():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    summary = mean_std_ci(values)

    assert summary.n == 5
    assert summary.mean == pytest.approx(3.0)
    assert summary.std == pytest.approx(np.std(values, ddof=1))
    half_width = 2.7764451 * summary.std / np.sqrt(5)  # t(0.975, df=4)
    assert summary.ci_low == pytest.approx(3.0 - half_width, rel=1e-4)
    assert summary.ci_high == pytest.approx(3.0 + half_width, rel=1e-4)


def test_a_single_value_has_no_spread_and_a_degenerate_interval():
    summary = mean_std_ci([2.5])

    assert summary.n == 1
    assert summary.std == 0.0
    assert summary.ci_low == summary.ci_high == 2.5


def test_constant_values_have_a_zero_width_interval():
    summary = mean_std_ci([1.0, 1.0, 1.0])

    assert summary.std == 0.0
    assert summary.ci_low == summary.ci_high == 1.0


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        mean_std_ci([])


def test_a_normal_mean_counts_as_zero_below_the_shared_threshold():
    # Between the report's former 1e-12 cut-off and the 1e-9 epsilon: the one threshold must decide.
    assert normal_mean_is_zero(1.0e-10)
    assert normal_mean_is_zero(-1.0e-10)
    assert not normal_mean_is_zero(1.0e-8)


def test_degradation_follows_the_prd_formula_and_guards_a_zero_baseline():
    assert degradation(metric_shift=0.3, metric_normal=0.1) == pytest.approx(0.2 / 0.1, rel=1e-6)
    assert np.isfinite(degradation(metric_shift=0.5, metric_normal=0.0))
    assert degradation(metric_shift=-1.0, metric_normal=-2.0) == pytest.approx(1.0 / 2.0, rel=1e-6)


def test_paired_comparison_reports_the_mean_difference_and_its_sign():
    method = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    baseline = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    result = paired_comparison(method, baseline)

    assert result.n == 10
    assert result.mean_difference == pytest.approx(np.mean(np.array(method) - np.array(baseline)))
    assert result.mean_difference < 0
    assert result.ci_high < 0  # method is reliably lower
    assert result.p_value < 0.05


def test_paired_comparison_with_identical_samples_is_a_clean_null():
    values = [0.0, 1.0, 0.0, 1.0]

    result = paired_comparison(values, values)

    assert result.mean_difference == 0.0
    assert result.p_value == 1.0


def test_paired_comparison_requires_matched_lengths():
    with pytest.raises(ValueError):
        paired_comparison([1.0, 2.0], [1.0])


def test_training_seed_pooling_averages_within_a_seed_before_spreading_across_seeds():
    records = [
        {"training_seed": 0, "x": 1.0}, {"training_seed": 0, "x": 3.0},
        {"training_seed": 1, "x": 5.0}, {"training_seed": 1, "x": 7.0},
    ]

    summary = pooled_by_training_seed(records, "x")

    assert summary.n == 2  # two training seeds, not four episodes
    assert summary.mean == pytest.approx(4.0)  # mean of the per-seed means 2 and 6
    assert summary.std == pytest.approx(np.std([2.0, 6.0], ddof=1))


def test_cluster_bootstrap_is_deterministic_for_a_fixed_seed():
    method = {s: [float(s % 2), float((s + 1) % 2)] for s in range(10)}
    baseline = {s: [0.0] for s in range(10)}

    first = cluster_bootstrap_ci(method, baseline, n_resamples=2000, seed=20261104)
    second = cluster_bootstrap_ci(method, baseline, n_resamples=2000, seed=20261104)

    assert first == second


def test_cluster_bootstrap_interval_contains_the_point_estimate():
    # Mean paired difference over scenarios: 6 scenarios with +1, 4 with 0 -> 0.6.
    method = {s: [1.0 if s < 6 else 0.0] for s in range(10)}
    baseline = {s: [0.0] for s in range(10)}

    low, high = cluster_bootstrap_ci(method, baseline, n_resamples=4000, seed=20261104)

    assert low <= 0.6 <= high
    assert low < high


def test_cluster_bootstrap_collapses_to_a_point_when_every_difference_is_equal():
    method = {s: [1.0, 1.0, 1.0] for s in range(6)}
    baseline = {s: [0.0, 0.0] for s in range(6)}

    assert cluster_bootstrap_ci(method, baseline, n_resamples=500, seed=20261104) == (1.0, 1.0)


def test_cluster_bootstrap_resamples_training_seeds_within_each_scenario():
    # Every scenario's learned values are 0 or 1 across two training seeds, the baseline is 0.5. Averaging each
    # scenario's training seeds as they are (ignoring the within-scenario resample) gives difference 0 everywhere,
    # a zero-width interval; resampling the training seeds spreads the per-scenario means and the interval.
    method = {s: [0.0, 1.0] for s in range(10)}
    baseline = {s: [0.5] for s in range(10)}

    low, high = cluster_bootstrap_ci(method, baseline, n_resamples=4000, seed=20261104)

    assert low < 0.0 < high


def test_cluster_bootstrap_uses_only_scenarios_both_models_ran():
    method = {0: [1.0], 1: [1.0], 2: [1.0]}
    baseline = {1: [0.0], 2: [0.0], 3: [0.0]}

    assert cluster_bootstrap_ci(method, baseline, n_resamples=200, seed=20261104) == (1.0, 1.0)


def test_cluster_bootstrap_needs_a_shared_scenario():
    with pytest.raises(ValueError):
        cluster_bootstrap_ci({0: [1.0]}, {1: [0.0]}, n_resamples=10, seed=1)
