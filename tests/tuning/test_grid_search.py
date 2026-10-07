"""Tests for orbit_guard.tuning.grid_search (P4-M5): validation-only coarse tuning."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from orbit_guard.evaluation.seed_pools import load_seed_pool, write_seed_pool
from orbit_guard.rl.evaluation import EvaluationReport
from orbit_guard.tuning.grid_search import (
    GridPoint,
    default_grid,
    freeze_best,
    run_grid,
    validation_score,
)

_REPO = Path(__file__).resolve().parents[2]
_BUDGET_MPS = 2.0


def _report(collisions: int, delta_v: float, episodes: int = 10) -> EvaluationReport:
    return EvaluationReport(
        episodes=episodes, mean_return_per_agent=0.0, collision_episodes=collisions, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=delta_v, mean_episode_steps=360.0,
    )


def test_default_grid_has_unique_named_points_including_the_unmodified_defaults():
    grid = default_grid()

    names = [point.name for point in grid]
    assert len(names) == len(set(names)) >= 6
    assert any(not point.reward_weights and not point.margin for point in grid)


def test_default_grid_only_varies_known_weights_and_coefficients():
    from orbit_guard.rl.training import load_margin_fn, load_reward_weights

    for point in default_grid():
        load_reward_weights(**point.reward_weights)  # raises on unknown names
        load_margin_fn(**point.margin)


def test_score_ranks_safety_first_then_fuel():
    safe_costly = validation_score(_report(0, 0.6), _BUDGET_MPS)
    unsafe_cheap = validation_score(_report(1, 0.0), _BUDGET_MPS)
    safe_cheap = validation_score(_report(0, 0.2), _BUDGET_MPS)

    assert safe_cheap < safe_costly < unsafe_cheap


def test_run_grid_evaluates_every_point_on_the_validation_pool_and_ranks_best_first():
    seen = []
    reports = {"a": _report(2, 0.1), "b": _report(0, 0.5), "c": _report(0, 0.1)}

    def run_point(point, seeds):
        seen.append((point.name, tuple(seeds)))
        return reports[point.name]

    results = run_grid([GridPoint(name) for name in reports], run_point, _BUDGET_MPS)

    pool = load_seed_pool(_REPO / "configs" / "validation_seeds.json", "validation")
    assert [name for name, _ in seen] == ["a", "b", "c"]
    assert all(seeds == pool for _, seeds in seen)
    assert [result.point.name for result in results] == ["c", "b", "a"]


def test_run_grid_can_use_a_prefix_of_the_pool_for_a_cheaper_coarse_pass():
    captured = {}

    def run_point(point, seeds):
        captured["n"] = len(seeds)
        return _report(0, 0.0)

    run_grid([GridPoint("a")], run_point, _BUDGET_MPS, max_seeds=7)

    assert captured["n"] == 7


def test_run_grid_refuses_a_non_validation_pool(tmp_path):
    bogus = tmp_path / "pool.json"
    write_seed_pool(bogus, "test", (1, 2, 3))
    config = tmp_path / "validation.yaml"
    config.write_text(yaml.safe_dump({"split": "validation", "seed_pool_file": str(bogus)}))

    with pytest.raises(ValueError, match="expected 'validation'"):
        run_grid([GridPoint("a")], lambda p, s: _report(0, 0.0), _BUDGET_MPS, validation_config=config)


def test_freeze_best_writes_a_hashed_overlay_of_the_winning_point(tmp_path):
    results = run_grid(
        [GridPoint("weak", reward_weights={"w4": 1.5}), GridPoint("strong", margin={"alpha": 2.0})],
        lambda point, seeds: _report(0, 0.1) if point.name == "strong" else _report(3, 0.1),
        _BUDGET_MPS,
    )
    target = tmp_path / "tuned.yaml"

    digest = freeze_best(results, target)

    frozen = yaml.safe_load(target.read_text())
    assert frozen["selected"] == "strong"
    assert frozen["adaptive_margin"] == {"alpha": 2.0}
    assert frozen["selected_on"] == "validation"
    assert frozen["config_hash"] == digest
    body = {key: value for key, value in frozen.items() if key != "config_hash"}
    default_bytes = (_REPO / "configs" / "default.yaml").read_bytes()
    expected = hashlib.sha256(default_bytes + json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert digest == expected


def test_grid_search_code_path_cannot_reach_the_test_seed_pool():
    code = (
        "import sys, orbit_guard.tuning.grid_search;"
        "leaked = [m for m in sys.modules if 'test_seed' in m];"
        "assert not leaked, leaked"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, cwd=_REPO)

    assert completed.returncode == 0, completed.stderr


def test_no_tuning_module_references_the_test_seed_pool():
    for source in (_REPO / "orbit_guard" / "tuning").glob("*.py"):
        text = source.read_text()
        assert "test_seed" not in text, source
        assert "TEST_SEED" not in text, source


_SETTINGS = {"agents": 15, "iterations": 25, "validation_episodes": 20, "seed": 0, "shield": True}


def test_a_resumed_grid_skips_points_already_logged_with_the_same_settings(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    reports = {"a": _report(2, 0.1), "b": _report(0, 0.5), "c": _report(0, 0.1)}
    first = []
    run_grid([GridPoint("a"), GridPoint("b")], lambda p, s: first.append(p.name) or reports[p.name], _BUDGET_MPS, log_path=log, settings=_SETTINGS, max_seeds=10)
    second = []

    results = run_grid(
        [GridPoint(n) for n in reports], lambda p, s: second.append(p.name) or reports[p.name], _BUDGET_MPS,
        log_path=log, settings=_SETTINGS, max_seeds=10, resume=True,
    )

    assert first == ["a", "b"] and second == ["c"], "only the unfinished point runs again"
    assert [r.point.name for r in results] == ["c", "b", "a"], "the logged points still take part in the ranking"
    assert results[1].report == reports["b"]


def test_a_resumed_grid_reruns_points_logged_under_different_settings(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), _BUDGET_MPS, log_path=log, settings=_SETTINGS, max_seeds=10)
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), _BUDGET_MPS,
             log_path=log, settings={**_SETTINGS, "iterations": 50}, max_seeds=10, resume=True)

    assert reran == ["a"]


def test_a_resumed_grid_ignores_a_logged_point_whose_episode_count_does_not_match_the_pool(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1, episodes=3), _BUDGET_MPS, log_path=log, settings=_SETTINGS, max_seeds=3)
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1, episodes=7), _BUDGET_MPS,
             log_path=log, settings=_SETTINGS, max_seeds=7, resume=True)

    assert reran == ["a"]


def test_a_torn_last_log_line_from_a_killed_run_is_ignored_on_resume(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), _BUDGET_MPS, log_path=log, settings=_SETTINGS, max_seeds=10)
    with log.open("a") as handle:
        handle.write('{"point": "b", "score": 0.1, "episo')  # the process died mid-write
    reran = []

    run_grid([GridPoint("a"), GridPoint("b")], lambda p, s: reran.append(p.name) or _report(0, 0.1), _BUDGET_MPS,
             log_path=log, settings=_SETTINGS, max_seeds=10, resume=True)

    assert reran == ["b"]


def test_without_resume_every_point_runs_even_if_logged(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), _BUDGET_MPS, log_path=log, settings=_SETTINGS, max_seeds=10)
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), _BUDGET_MPS, log_path=log, settings=_SETTINGS, max_seeds=10)

    assert reran == ["a"]
