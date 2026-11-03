"""Failure taxonomy and result figures generated from logged episodes (PRD §36, §49; P6-M3/M4)."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from orbit_guard.evaluation.results import (
    ALL_FAILURE_LABELS,
    collision_rate_figure,
    failure_taxonomy,
    taxonomy_markdown,
    write_results,
)
from orbit_guard.rl import episode_metrics as em


def _record(model, condition, seed, labels=(), training_seed=None, collisions=0):
    return {
        "model_name": model, "shift_type": condition, "scenario_seed": seed, "training_seed": training_seed,
        "collision_count": collisions, "unsafe_conjunction_count": 0, "total_delta_v_mps": 0.1, "maneuver_count": 1,
        "mean_mission_deviation": 0.0, "shield_intervention_count": 0, "mean_residual_risk": 0.0,
        "return_per_agent": 1.0, "steps": 10, "status": "ok", "failure_labels": list(labels), "experiment_id": "x",
    }


def test_the_taxonomy_lists_all_nine_prd_categories_machine_readably():
    assert len(ALL_FAILURE_LABELS) == 9
    assert em.LABEL_COLLISION in ALL_FAILURE_LABELS and em.LABEL_SIMULATION_FAILURE in ALL_FAILURE_LABELS


def test_taxonomy_counts_each_label_per_model_without_collapsing_them():
    records = [
        _record("rule_based", "normal", 1, [em.LABEL_COLLISION, em.LABEL_UNSAFE_REMAINED]),
        _record("rule_based", "normal", 2, [em.LABEL_UNSAFE_REMAINED]),
        _record("adaptive_mappo", "normal", 1, [em.LABEL_SHIELD_REJECTED], training_seed=0),
    ]

    table = failure_taxonomy(records)

    assert table["rule_based"][em.LABEL_COLLISION] == 1
    assert table["rule_based"][em.LABEL_UNSAFE_REMAINED] == 2
    assert table["adaptive_mappo"][em.LABEL_SHIELD_REJECTED] == 1
    assert set(table["rule_based"]) == set(ALL_FAILURE_LABELS)  # zero-filled, every category present


def test_one_episode_can_carry_several_labels():
    table = failure_taxonomy([_record("m", "normal", 1, [em.LABEL_COLLISION, em.LABEL_BUDGET_EXHAUSTED])])

    assert table["m"][em.LABEL_COLLISION] == table["m"][em.LABEL_BUDGET_EXHAUSTED] == 1


def test_taxonomy_markdown_has_a_row_per_category():
    md = taxonomy_markdown(failure_taxonomy([_record("m", "normal", 1, [em.LABEL_COLLISION])]))

    for label in ALL_FAILURE_LABELS:
        assert label in md


def test_collision_rate_figure_has_a_bar_per_model_with_error_bars():
    records = [
        _record("no_maneuver", "normal", s, collisions=1 if s < 3 else 0) for s in range(6)
    ] + [
        _record("adaptive_mappo", "normal", s, training_seed=ts, collisions=0) for s in range(6) for ts in (0, 1)
    ]

    figure = collision_rate_figure(records)

    assert {trace.name for trace in figure.data} == {"no_maneuver", "adaptive_mappo"}
    assert all(trace.error_y is not None for trace in figure.data)


def test_write_results_generates_files_from_the_log_alone(tmp_path):
    records = [_record("no_maneuver", c, s) for c in ("normal", "noise_increase") for s in range(3)]

    written = write_results(records, tmp_path)

    names = {path.name for path in written}
    assert {"results_table.md", "failure_taxonomy.md", "collision_rate.html"} <= names
    assert all(path.stat().st_size > 0 for path in written)
    assert "no_maneuver" in (tmp_path / "results_table.md").read_text()


def test_write_results_also_writes_the_paired_and_degradation_tables(tmp_path):
    records = [_record("no_maneuver", c, s, collisions=s % 2) for c in ("normal", "noise_increase") for s in range(3)]
    records += [_record("adaptive_mappo", c, s, training_seed=0) for c in ("normal", "noise_increase") for s in range(3)]

    written = write_results(records, tmp_path)

    names = {path.name for path in written}
    assert {"paired_comparisons.md", "degradation.md"} <= names
    assert "no_maneuver" in (tmp_path / "paired_comparisons.md").read_text()
    assert "noise_increase" in (tmp_path / "degradation.md").read_text()


def test_the_figure_tolerates_a_condition_where_every_episode_failed():
    records = [_record("no_maneuver", "normal", s) for s in range(3)]
    records += [{**_record("no_maneuver", "noise_increase", s), "status": "failed"} for s in range(3)]

    figure = collision_rate_figure(records)

    (trace,) = figure.data
    assert list(trace.x) == ["normal", "noise_increase"]
    assert trace.y[1] is None
