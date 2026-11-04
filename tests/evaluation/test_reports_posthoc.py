"""Reporting fixes found by the R6 results audit (2026-11-04), applied after the final test sweep.

They change how logged episodes are rendered, never which episodes exist: rate CIs stay t-intervals across
training seeds (PRD §34) but are clipped to the rate's range, a cell with no spread says so instead of printing a
zero-width interval, column names state what is counted (collision OR unsafe; fleet Δv; agents changed by the
shield), paired comparisons carry a Holm-adjusted p declared post hoc, degradation from a zero normal mean shows
the absolute change, and the failure taxonomy shows its denominators.
"""

from __future__ import annotations

import pytest

from orbit_guard.evaluation.experiment import (
    format_degradation_report,
    format_paired_report,
    format_report,
)
from orbit_guard.evaluation.statistics import holm_adjust, mean_std_ci


def _record(model, condition, seed, training_seed=None, collisions=0, unsafe=0, dv=0.1, status="ok", labels=()):
    return {
        "model_name": model, "shift_type": condition, "scenario_seed": seed, "training_seed": training_seed,
        "collision_count": collisions, "unsafe_conjunction_count": unsafe, "total_delta_v_mps": dv,
        "maneuver_count": 2, "mean_mission_deviation": 0.05, "shield_intervention_count": 0,
        "mean_residual_risk": 0.1, "return_per_agent": 10.0, "steps": 100, "status": status,
        "failure_labels": list(labels), "experiment_id": "x",
    }


def _cells(report: str, *first: str) -> list[str]:
    for line in report.splitlines():
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts[: len(first)] == list(first):
            return parts
    raise AssertionError(f"no row starting {first} in\n{report}")


def _header(report: str) -> list[str]:
    return [part.strip() for part in report.splitlines()[0].strip("|").split("|")]


def _learned(condition: str, collisions_by_seed: dict[int, list[int]]) -> list[dict]:
    return [
        _record("adaptive_mappo_no_shield", condition, s, training_seed=ts, collisions=c)
        for ts, per_scenario in collisions_by_seed.items() for s, c in enumerate(per_scenario)
    ]


def test_a_rate_ci_never_leaves_zero_to_one():
    # Seed rates 0, 0, 0.2: the unclipped t-interval starts below zero.
    records = _learned("normal", {0: [0] * 10, 1: [0] * 10, 2: [1, 1] + [0] * 8})
    raw = mean_std_ci([0.0, 0.0, 0.2])
    assert raw.ci_low < 0.0

    cells = _cells(format_report(records), "adaptive_mappo_no_shield", "normal")
    column = _header(format_report(records)).index("collision 95% CI")

    assert cells[column] == f"[0.000, {raw.ci_high:.3f}]"


def test_a_rate_with_no_spread_across_seeds_says_so_rather_than_a_zero_width_interval():
    records = _learned("normal", {0: [0] * 10, 1: [0] * 10, 2: [0] * 10})

    report = format_report(records)
    cells = _cells(report, "adaptive_mappo_no_shield", "normal")
    column = _header(report).index("collision 95% CI")

    assert cells[column] == "no spread (3 seeds, all 0.000)"


def test_equal_seed_rates_with_float_rounding_still_count_as_no_spread():
    # 7 of 10 in every seed: the per-seed means are equal, but their float std is ~1e-16, not 0.
    per_seed = [1] * 7 + [0] * 3
    records = _learned("capability_reduction", {0: per_seed, 1: per_seed, 2: per_seed})

    report = format_report(records)
    cells = _cells(report, "adaptive_mappo_no_shield", "capability_reduction")

    assert cells[_header(report).index("collision 95% CI")] == "no spread (3 seeds, all 0.700)"


def test_the_results_table_names_what_each_column_counts():
    report = format_report(_learned("normal", {0: [0, 1], 1: [0, 0]}))
    header = _header(report)

    assert "collision-or-unsafe rate" in header and "collision-or-unsafe 95% CI" in header
    assert "fleet Δv (m/s)" in header
    assert "shield agent-changes/step" in header
    assert "default-config adaptive margin" in report  # the yardstick unsafe is scored against (Q33)
    assert "clipped to [0, 1]" in report


def _paired_records() -> list[dict]:
    records = []
    for seed in range(8):
        records += [
            _record("adaptive_mappo", "normal", seed, training_seed=ts, unsafe=int(seed == 0 and ts == 0), dv=0.2)
            for ts in (0, 1, 2)
        ]
        records.append(_record("no_maneuver", "normal", seed, collisions=1, unsafe=1, dv=0.0))
        records.append(_record("fixed_margin_mappo", "normal", seed, training_seed=0, unsafe=int(seed < 2)))
    return records


def test_a_paired_rate_difference_ci_stays_within_minus_one_to_one():
    report = format_paired_report(_paired_records(), baselines=("no_maneuver",))

    cells = _cells(report, "normal", "no_maneuver", "unsafe_rate")
    low, high = (float(x) for x in cells[5].strip("[]").split(","))

    assert -1.0 <= low <= high <= 1.0


def _rows(report: str) -> list[list[str]]:
    return [
        [part.strip() for part in line.strip("|").split("|")]
        for line in report.splitlines()
        if line.startswith("| ") and not line.startswith(("| condition", "|---"))
    ]


def test_the_paired_report_adds_a_holm_adjusted_p_over_the_whole_table_declared_post_hoc():
    # Two conditions: a per-condition family would give different numbers than the table-wide one.
    density = [{**record, "shift_type": "density_increase", "unsafe_conjunction_count": 1 - record["unsafe_conjunction_count"]}
               for record in _paired_records()]
    report = format_paired_report(_paired_records() + density, baselines=("fixed_margin_mappo", "no_maneuver"))
    header = [part.strip() for part in report.splitlines()[2].strip("|").split("|")]
    rows = _rows(report)
    raw = [float(row[6]) for row in rows]

    assert {row[0] for row in rows} == {"normal", "density_increase"}
    assert header[-1] == "Holm p (post hoc)"
    assert [float(row[7]) for row in rows] == pytest.approx(holm_adjust(raw), abs=5e-4)
    assert "after the final test sweep" in report
    assert "every row with a test (n >= 2, finite p)" in report


def test_a_row_without_a_test_stays_out_of_the_holm_family():
    # rule_based shares one scenario seed with adaptive_mappo: n = 1, p = 1.0 by convention, not a test.
    records = _paired_records() + [_record("rule_based", "normal", 0, dv=0.0)]
    report = format_paired_report(records, baselines=("rule_based", "no_maneuver"))
    rows = _rows(report)
    tested = [row for row in rows if row[1] == "no_maneuver"]

    assert all(row[7] == "—" for row in rows if row[1] == "rule_based")
    assert [float(row[7]) for row in tested] == pytest.approx(holm_adjust([float(r[6]) for r in tested]), abs=5e-4)


def test_holm_adjustment_matches_the_step_down_rule_and_keeps_input_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 1.0]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []


def test_holm_adjustment_leaves_a_nan_p_out_of_the_family():
    adjusted = holm_adjust([0.04, float("nan"), 0.01])

    assert adjusted[0] == pytest.approx(0.04) and adjusted[2] == pytest.approx(0.02)
    assert adjusted[1] != adjusted[1]  # NaN stays NaN


def test_degradation_from_a_zero_normal_mean_shows_the_absolute_change():
    records = [
        _record("rule_based", "normal", 1, collisions=0),
        _record("rule_based", "density_increase", 1, collisions=1),
    ]

    line = " | ".join(_cells(format_degradation_report(records), "rule_based", "collision_rate"))

    assert "n/a (normal = 0; shift 1.000; change +1.000)" in line


def test_the_failure_taxonomy_shows_how_many_episodes_each_column_counts():
    pytest.importorskip("plotly")
    from orbit_guard.evaluation.results import (
        episode_counts,
        failure_taxonomy,
        taxonomy_markdown,
    )

    records = [
        _record("rule_based", "normal", 1, labels=["physical_collision"]),
        _record("rule_based", "normal", 2),
        _record("rule_based", "density_increase", 1),
        _record("rule_based", "normal", 3, status="failed", labels=["numerical_simulation_failure"]),
    ]

    markdown = taxonomy_markdown(failure_taxonomy(records), episode_counts(records))

    assert _cells(markdown, "episodes (any status)") == ["episodes (any status)", "4"]
    assert "total over all conditions, scenario seeds and training seeds, failed episodes included" in markdown
