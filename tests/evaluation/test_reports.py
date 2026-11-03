"""PRD §34 statistics rendered from the episode log: CIs, failed counts, paired comparisons, degradation.

The statistics were computed (statistics.mean_std_ci, paired_comparison, degradation) but never
rendered. These tests pin what each report shows, and that failed episodes are counted, never dropped.
"""

from __future__ import annotations

import re

import pytest

from orbit_guard.evaluation import experiment
from orbit_guard.evaluation.experiment import (
    format_degradation_report,
    format_paired_report,
    format_report,
    paired_by_scenario,
    summarize,
)
from orbit_guard.evaluation.statistics import mean_std_ci


def _record(model, condition, seed, training_seed=None, collisions=0, unsafe=0, dv=0.1, status="ok"):
    return {
        "model_name": model, "shift_type": condition, "scenario_seed": seed, "training_seed": training_seed,
        "collision_count": collisions, "unsafe_conjunction_count": unsafe, "total_delta_v_mps": dv,
        "maneuver_count": 2, "mean_mission_deviation": 0.05, "shield_intervention_count": 0,
        "mean_residual_risk": 0.1, "return_per_agent": 10.0, "steps": 100, "status": status,
        "failure_labels": [], "experiment_id": "x",
    }


def _row_line(report: str, *cells: str) -> str:
    for line in report.splitlines():
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts[: len(cells)] == list(cells):
            return line
    raise AssertionError(f"no row starting {cells} in\n{report}")


def test_the_report_renders_the_95_percent_ci_of_the_collision_and_unsafe_rates():
    records = [
        _record("adaptive_mappo", "normal", s, training_seed=ts, collisions=int(s < ts + 1), unsafe=int(s < 2))
        for ts in (0, 1, 2) for s in range(4)
    ]
    collision = mean_std_ci([0.25, 0.5, 0.75])
    unsafe = mean_std_ci([0.5, 0.5, 0.75])

    report = format_report(records)

    assert "collision 95% CI" in report and "unsafe 95% CI" in report
    line = _row_line(report, "adaptive_mappo", "normal")
    assert f"[{collision.ci_low:.3f}, {collision.ci_high:.3f}]" in line
    assert f"[{unsafe.ci_low:.3f}, {unsafe.ci_high:.3f}]" in line


def test_a_single_training_seed_has_no_ci_rather_than_a_zero_width_one():
    report = format_report([_record("rule_based", "normal", s, collisions=s % 2) for s in range(4)])

    line = _row_line(report, "rule_based", "normal")
    assert "n/a (1 seed)" in line


def test_each_row_counts_its_failed_episodes():
    records = [_record("rule_based", "normal", 1), _record("rule_based", "normal", 2, status="failed")]

    (row,) = summarize(records)
    report = format_report(records)

    assert row["episodes"] == 1 and row["failed"] == 1
    assert "| failed |" in report
    assert _row_line(report, "rule_based", "normal", "1", "1")


def test_a_condition_where_every_episode_failed_still_gets_a_row():
    records = [
        _record("rule_based", "normal", 1),
        _record("rule_based", "noise_increase", 1, status="failed"),
        _record("rule_based", "noise_increase", 2, status="failed"),
    ]

    rows = {row["condition"]: row for row in summarize(records)}
    report = format_report(records)

    assert rows["noise_increase"]["episodes"] == 0 and rows["noise_increase"]["failed"] == 2
    assert rows["noise_increase"]["collision_rate"] is None
    assert _row_line(report, "rule_based", "noise_increase", "0", "2")


def test_summary_reports_unsafe_rate_degradation_against_normal():
    records = [
        _record("rule_based", "normal", 1, unsafe=1),
        _record("rule_based", "normal", 2, unsafe=0),
        _record("rule_based", "noise_increase", 1, unsafe=1),
        _record("rule_based", "noise_increase", 2, unsafe=1),
    ]

    rows = {row["condition"]: row for row in summarize(records)}

    assert rows["noise_increase"]["unsafe_rate_degradation"] == pytest.approx(1.0, rel=1e-6)


def test_paired_by_scenario_accepts_the_derived_episode_rates():
    records = [_record("adaptive_mappo", "normal", s, training_seed=0) for s in range(4)] + [
        _record("no_maneuver", "normal", s, collisions=1) for s in range(4)
    ]

    result = paired_by_scenario(records, "adaptive_mappo", "no_maneuver", "normal", "collision_rate")

    assert result.n == 4 and result.mean_difference == pytest.approx(-1.0)


def _paired_records() -> list[dict]:
    records = []
    for seed in range(8):
        records += [_record("adaptive_mappo", "normal", seed, training_seed=ts, dv=0.2) for ts in (0, 1)]
        records.append(_record("no_maneuver", "normal", seed, collisions=1, unsafe=1, dv=0.0))
        records.append(_record("rule_based", "normal", seed, dv=0.2))
    records.append(_record("fixed_margin_mappo", "normal", 99, training_seed=0))  # no shared seeds
    return records


def test_the_paired_report_shows_n_mean_difference_ci_and_wilcoxon_p():
    report = format_paired_report(_paired_records())

    line = _row_line(report, "normal", "no_maneuver", "collision_rate")
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    assert cells[3] == "8"
    assert cells[4] == "-1.000"
    assert float(cells[6]) == pytest.approx(0.0078125, abs=1e-4)  # exact two-sided signed-rank, n=8


def test_identical_outcomes_give_a_zero_difference_with_p_equal_one():
    report = format_paired_report(_paired_records())

    line = _row_line(report, "normal", "rule_based", "total_delta_v_mps")
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    assert cells[4] == "0.000" and cells[6] == "1.000"


def test_a_baseline_without_shared_scenario_seeds_is_skipped_and_says_so():
    report = format_paired_report(_paired_records())

    line = _row_line(report, "normal", "fixed_margin_mappo")
    assert "skipped" in line and "no shared scenario seeds" in line


def test_the_paired_report_states_its_sign_convention_and_tie_handling():
    report = format_paired_report(_paired_records())

    assert "adaptive_mappo − baseline" in report
    assert re.search(r"zero differences", report)


def test_the_degradation_report_lists_each_model_metric_and_shift():
    records = [
        _record("rule_based", "normal", 1, collisions=1, unsafe=1, dv=0.1),
        _record("rule_based", "normal", 2, collisions=0, unsafe=1, dv=0.1),
        _record("rule_based", "noise_increase", 1, collisions=1, unsafe=1, dv=0.2),
        _record("rule_based", "noise_increase", 2, collisions=1, unsafe=1, dv=0.2),
    ]

    report = format_degradation_report(records)

    collision = _row_line(report, "rule_based", "collision_rate")
    assert "+1.000 (0.500 → 1.000)" in collision
    assert "+1.000 (0.100 → 0.200)" in _row_line(report, "rule_based", "total_delta_v_mps")
    assert "+0.000 (1.000 → 1.000)" in _row_line(report, "rule_based", "unsafe_rate")
    assert "noise_increase" in report


def test_degradation_from_a_zero_normal_rate_is_not_a_huge_number():
    records = [
        _record("rule_based", "normal", 1, collisions=0),
        _record("rule_based", "density_increase", 1, collisions=1),
    ]

    report = format_degradation_report(records)

    line = _row_line(report, "rule_based", "collision_rate")
    assert "n/a (normal = 0; shift 1.000)" in line
    assert "e+" not in line


def test_the_report_command_prints_all_three_tables(tmp_path, capsys):
    import json

    log = tmp_path / "episodes.jsonl"
    log.write_text("".join(json.dumps(record) + "\n" for record in _paired_records()))

    experiment.main(["report", "--episodes", str(log)])

    out = capsys.readouterr().out
    assert "collision 95% CI" in out
    assert "Wilcoxon p" in out
    assert "Degradation" in out
