"""Acceptance Test G (PRD §38): all four core shifts run from one experiment-config interface."""

from __future__ import annotations

from orbit_guard.evaluation.experiment import build_tasks
from orbit_guard.evaluation.runner import run_tasks
from orbit_guard.evaluation.shifts import CONDITION_NAMES

_FAST = {"episode_horizon_s": 240.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 300.0}


def test_the_runner_executes_normal_plus_all_four_shifts_for_every_non_learned_model(tmp_path):
    condition_seeds = {name: (4, 5) for name in CONDITION_NAMES}
    tasks = build_tasks(
        ("no_maneuver", "rule_based"), condition_seeds, {}, n_agents=10, experiment_id="g", env_kwargs=_FAST
    )
    out = tmp_path / "episodes.jsonl"

    records = run_tasks(tasks, out, workers=2)

    assert len(records) == 2 * 5 * 2
    assert all(record["status"] == "ok" for record in records)
    for model in ("no_maneuver", "rule_based"):
        seen = {record["shift_type"] for record in records if record["model_name"] == model}
        assert seen == set(CONDITION_NAMES)


def test_the_shifts_actually_change_the_world_they_describe(tmp_path):
    tasks = build_tasks(
        ("no_maneuver",), {name: (6,) for name in CONDITION_NAMES}, {}, n_agents=10, experiment_id="g2", env_kwargs=_FAST
    )

    records = {r["shift_type"]: r for r in run_tasks(tasks, tmp_path / "e.jsonl", workers=1)}

    assert records["noise_increase"]["mean_residual_risk"] != records["normal"]["mean_residual_risk"]
    assert records["density_increase"]["mean_residual_risk"] != records["normal"]["mean_residual_risk"]
