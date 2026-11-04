"""Confirmatory comparisons (declared 2026-11-04, after the final test sweep; post hoc).

Two families of paired comparisons on the collision-or-unsafe rate, each family with its own Holm correction over
its five conditions, plus a two-stage cluster bootstrap as the sensitivity column. The pre-registration fixed none
of these choices, so the report says so, and the descriptive tables stay as they were.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("plotly")

from orbit_guard.evaluation import experiment
from orbit_guard.evaluation.experiment import format_confirmatory_report
from orbit_guard.evaluation.results import write_results
from orbit_guard.evaluation.shifts import CONDITION_NAMES
from orbit_guard.evaluation.statistics import holm_adjust

# Per scenario seed 0..9, the 0/1 episode outcome (collision-or-unsafe) of each model under each condition.
ADAPTIVE = {
    "normal": [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    "noise_increase": [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    "density_increase": [0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
    "simultaneous_conjunctions": [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    "capability_reduction": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
}
FIXED_MARGIN = {
    "normal": [1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    "noise_increase": [1] * 10,
    "density_increase": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "simultaneous_conjunctions": [1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    "capability_reduction": [0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
}
NO_SHIELD = {
    "normal": [1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    "noise_increase": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
    "density_increase": [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    "simultaneous_conjunctions": [0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    "capability_reduction": [1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
}
# Family 1 (adaptive - fixed) Holm values for this fixture are 1.000, 0.156, 0.250, 1.000, 1.000 within the family;
# Holm over all ten rows would give 1.000, 0.312, 0.563, 1.000, 1.000, so the tests can tell the two apart.
FAMILY_1 = "Family 1 (primary, PRD §2 headline)"
FAMILY_2 = "Family 2 (secondary)"


def _record(model, condition, seed, training_seed=None, unsafe=0, collisions=0, status="ok"):
    return {
        "model_name": model, "shift_type": condition, "scenario_seed": seed, "training_seed": training_seed,
        "collision_count": collisions, "unsafe_conjunction_count": unsafe, "total_delta_v_mps": 0.1,
        "maneuver_count": 1, "mean_mission_deviation": 0.05, "shield_intervention_count": 0,
        "mean_residual_risk": 0.1, "return_per_agent": 10.0, "steps": 100, "status": status,
        "failure_labels": [], "experiment_id": "x",
    }


def _model_records(model, condition, outcomes):
    """One learned checkpoint (training seed 0) per scenario seed; `outcomes[seed]` is 0 or 1."""
    return [_record(model, condition, seed, training_seed=0, unsafe=value) for seed, value in enumerate(outcomes)]


def _fixture(adaptive=ADAPTIVE, fixed=FIXED_MARGIN, no_shield=NO_SHIELD):
    records = []
    for condition in CONDITION_NAMES:
        records += _model_records("adaptive_mappo", condition, adaptive[condition])
        records += _model_records("fixed_margin_mappo", condition, fixed[condition])
        records += _model_records("adaptive_mappo_no_shield", condition, no_shield[condition])
    return records


def _section(report: str, heading: str) -> list[list[str]]:
    """The table rows under one `### <heading>` section, cells stripped."""
    (body,) = [part for part in report.split("### ")[1:] if part.startswith(heading)]
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in body.splitlines()
        if line.startswith("| ") and not line.startswith(("| condition", "|---"))
    ]


def _row(report: str, heading: str, condition: str) -> list[str]:
    (row,) = [cells for cells in _section(report, heading) if cells[0] == condition]
    return row


def test_family_one_has_one_row_per_condition_in_canonical_order():
    report = format_confirmatory_report(_fixture())

    assert [cells[0] for cells in _section(report, FAMILY_1)] == list(CONDITION_NAMES)
    assert [cells[0] for cells in _section(report, FAMILY_2)] == list(CONDITION_NAMES)


def test_holm_is_adjusted_within_family_one_not_across_both_families():
    report = format_confirmatory_report(_fixture())

    # Over all ten rows the noise and density rows would be 0.312 and 0.563; within family 1 they are 0.156 and 0.250.
    assert _row(report, FAMILY_1, "noise_increase")[5] == "0.156"
    assert _row(report, FAMILY_1, "density_increase")[5] == "0.250"
    assert _row(report, FAMILY_1, "normal")[5] == "1.000"


def test_holm_within_family_two_matches_the_step_down_rule_over_its_own_rows():
    report = format_confirmatory_report(_fixture())
    rows = _section(report, FAMILY_2)
    raw = [float(cells[4]) for cells in rows]

    assert [float(cells[5]) for cells in rows] == pytest.approx(holm_adjust(raw), abs=5e-4)


def test_a_row_without_a_test_stays_out_of_its_family_and_is_shown_as_a_dash():
    # Family 2 has no no-shield episodes under simultaneous_conjunctions (skipped) and one scenario under
    # capability_reduction (n = 1, no test): neither enters the Holm family of its five.
    records = [
        record for record in _fixture()
        if not (record["model_name"] == "adaptive_mappo_no_shield" and record["shift_type"] == "simultaneous_conjunctions")
        and not (record["model_name"] == "adaptive_mappo_no_shield" and record["shift_type"] == "capability_reduction"
                 and record["scenario_seed"] > 0)
    ]
    report = format_confirmatory_report(records)

    skipped = _row(report, FAMILY_2, "simultaneous_conjunctions")
    single = _row(report, FAMILY_2, "capability_reduction")
    assert skipped[1] == "0" and skipped[5] == "—" and skipped[6] == "—"
    assert single[1] == "1" and single[5] == "—" and single[6] == "—"
    tested = [_row(report, FAMILY_2, c) for c in ("normal", "noise_increase", "density_increase")]
    assert [float(cells[5]) for cells in tested] == pytest.approx(
        holm_adjust([float(cells[4]) for cells in tested]), abs=5e-4
    )


def test_the_paired_t_interval_is_clipped_to_minus_one_and_one():
    # Nine scenarios with +1 and one with 0: the mean is 0.9 and the unclipped t-interval runs above 1.
    records = _model_records("adaptive_mappo", "normal", [1] * 9 + [0])
    records += _model_records("fixed_margin_mappo", "normal", [0] * 10)
    records += _model_records("adaptive_mappo_no_shield", "normal", [0] * 10)

    row = _row(format_confirmatory_report(records), FAMILY_1, "normal")
    low, high = (float(x) for x in row[3].strip("[]").split(","))

    assert high == 1.0
    assert -1.0 <= low <= high <= 1.0


def test_the_bootstrap_column_is_deterministic_and_contains_the_mean_difference():
    records = _fixture()

    first = format_confirmatory_report(records)
    second = format_confirmatory_report(records)
    row = _row(first, FAMILY_1, "normal")
    low, high = (float(x) for x in row[6].strip("[]").split(","))

    assert first == second
    assert low <= float(row[2]) <= high


def test_the_footnote_declares_the_families_post_hoc_and_the_bootstrap():
    report = format_confirmatory_report(_fixture())

    assert "declared on 2026-11-04" in report
    assert "AFTER the final test sweep" in report
    assert "post hoc" in report
    assert "descriptive" in report
    assert "collision-or-unsafe rate" in report
    assert "Q33" in report
    assert "resamples scenario seeds" in report and "within them, training seeds" in report
    assert "B = 10000" in report and "20261104" in report
    assert "over the rows of its family that carry a test" in report
    assert "five conditions" not in report


def test_a_non_finite_p_value_is_shown_as_a_dash_not_as_nan(monkeypatch):
    real = experiment.paired_by_scenario

    def with_nan_p(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), p_value=float("nan"))

    monkeypatch.setattr(experiment, "paired_by_scenario", with_nan_p)
    report = format_confirmatory_report(_fixture())
    row = _row(report, FAMILY_1, "normal")

    assert row[4] == "—" and row[5] == "—"
    assert "nan" not in row


def test_write_results_writes_and_returns_confirmatory_md(tmp_path):
    written = write_results(_fixture(), tmp_path)

    confirmatory = tmp_path / "confirmatory.md"
    assert confirmatory in written
    assert confirmatory.read_text().startswith("# Confirmatory comparisons")
    assert FAMILY_1 in confirmatory.read_text()
