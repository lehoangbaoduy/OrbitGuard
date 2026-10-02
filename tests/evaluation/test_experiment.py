"""Tests for orbit_guard.evaluation.experiment: task matrix, aggregation, paired report (PRD §34-35)."""

from __future__ import annotations

import pytest

from orbit_guard.evaluation.experiment import (
    build_tasks,
    format_report,
    paired_by_scenario,
    summarize,
)
from orbit_guard.evaluation.shifts import CONDITION_NAMES


def _record(model, condition, seed, training_seed=None, collisions=0, unsafe=0, dv=0.1, steps=100, shield=0):
    return {
        "model_name": model, "shift_type": condition, "scenario_seed": seed, "training_seed": training_seed,
        "collision_count": collisions, "unsafe_conjunction_count": unsafe, "total_delta_v_mps": dv,
        "maneuver_count": 2, "mean_mission_deviation": 0.05, "shield_intervention_count": shield,
        "mean_residual_risk": 0.1, "return_per_agent": 10.0, "steps": steps, "status": "ok",
        "failure_labels": [], "experiment_id": "x",
    }


def test_task_matrix_covers_every_model_condition_scenario_and_training_seed():
    seeds = {name: (10 + i, 20 + i) for i, name in enumerate(CONDITION_NAMES)}
    registry = {"adaptive_mappo": {0: "a0.npz", 1: "a1.npz", 2: "a2.npz"}}

    tasks = build_tasks(("no_maneuver", "adaptive_mappo"), seeds, registry, n_agents=15, experiment_id="e")

    # non-learned: 5 conditions x 2 seeds; learned: x 3 training seeds
    assert len(tasks) == 10 + 30
    learned = [t for t in tasks if t.model == "adaptive_mappo"]
    assert {t.training_seed for t in learned} == {0, 1, 2}
    assert all(t.actor_weights_path for t in learned)
    assert all(t.training_seed is None and t.actor_weights_path is None for t in tasks if t.model == "no_maneuver")


def test_every_model_sees_identical_scenario_seeds_per_condition():
    seeds = {name: (1, 2, 3) for name in CONDITION_NAMES}
    registry = {"adaptive_mappo": {0: "w.npz"}}

    tasks = build_tasks(("rule_based", "adaptive_mappo"), seeds, registry, n_agents=15, experiment_id="e")

    for condition in CONDITION_NAMES:
        per_model = {
            model: sorted(t.scenario_seed for t in tasks if t.model == model and t.condition == condition)
            for model in ("rule_based", "adaptive_mappo")
        }
        assert per_model["rule_based"] == per_model["adaptive_mappo"] == [1, 2, 3]


def test_a_learned_model_without_registered_checkpoints_is_rejected():
    with pytest.raises(ValueError, match="checkpoints"):
        build_tasks(("adaptive_mappo",), {"normal": (1,)}, {}, n_agents=15, experiment_id="e")


def test_summary_pools_training_seeds_and_reports_the_primary_endpoint():
    records = [
        _record("adaptive_mappo", "normal", 1, training_seed=0, collisions=0),
        _record("adaptive_mappo", "normal", 2, training_seed=0, collisions=1),
        _record("adaptive_mappo", "normal", 1, training_seed=1, collisions=0),
        _record("adaptive_mappo", "normal", 2, training_seed=1, collisions=0),
    ]

    rows = summarize(records)

    (row,) = rows
    assert row["model"] == "adaptive_mappo" and row["condition"] == "normal"
    assert row["training_seeds"] == 2 and row["episodes"] == 4
    assert row["collision_rate"].mean == pytest.approx((0.5 + 0.0) / 2)  # mean of per-seed rates


def test_summary_computes_degradation_relative_to_the_normal_condition():
    records = [
        _record("rule_based", "normal", 1, collisions=1),
        _record("rule_based", "normal", 2, collisions=0),
        _record("rule_based", "noise_increase", 1, collisions=1),
        _record("rule_based", "noise_increase", 2, collisions=1),
    ]

    rows = {row["condition"]: row for row in summarize(records)}

    assert rows["normal"]["collision_rate_degradation"] == 0.0
    assert rows["noise_increase"]["collision_rate_degradation"] == pytest.approx((1.0 - 0.5) / 0.5, rel=1e-6)


def test_shield_intervention_rate_is_per_step():
    records = [_record("adaptive_mappo", "normal", 1, training_seed=0, shield=10, steps=100)]

    (row,) = summarize(records)

    assert row["shield_intervention_rate"].mean == pytest.approx(0.1)


def test_paired_comparison_matches_scenario_seeds_and_averages_training_seeds():
    records = []
    for seed in range(1, 9):
        records.append(_record("fixed_margin_mappo", "normal", seed, training_seed=0, collisions=1))
        records.append(_record("fixed_margin_mappo", "normal", seed, training_seed=1, collisions=1))
        records.append(_record("adaptive_mappo", "normal", seed, training_seed=0, collisions=0))
        records.append(_record("adaptive_mappo", "normal", seed, training_seed=1, collisions=1 if seed == 1 else 0))

    result = paired_by_scenario(records, "adaptive_mappo", "fixed_margin_mappo", "normal", "collision_count")

    assert result.n == 8
    assert result.mean_difference < 0  # adaptive has fewer collisions on the same scenarios


def test_report_is_a_markdown_table_naming_models_conditions_and_seed_counts():
    records = [
        _record("rule_based", "normal", 1),
        _record("adaptive_mappo", "normal", 1, training_seed=0),
        _record("adaptive_mappo", "normal", 1, training_seed=1),
    ]

    report = format_report(records)

    assert "| model |" in report
    assert "rule_based" in report and "adaptive_mappo" in report
    assert "training seeds" in report.lower()
