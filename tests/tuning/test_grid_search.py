"""Tests for orbit_guard.tuning.grid_search (P4-M5): validation-only coarse tuning."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.evaluation.seed_pools import (
    generate_validation_pool,
    load_seed_pool,
    write_seed_pool,
)
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


def _report(collisions: int, delta_v: float, episodes: int = 10, *, unsafe: int = 0) -> EvaluationReport:
    return EvaluationReport(
        episodes=episodes, mean_return_per_agent=0.0, collision_episodes=collisions, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=delta_v, mean_episode_steps=360.0,
        mean_budget_fraction_used=delta_v / _BUDGET_MPS, unsafe_episodes=unsafe,
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
    safe_costly = validation_score(_report(0, 0.6), "q27")
    unsafe_cheap = validation_score(_report(1, 0.0), "q27")
    safe_cheap = validation_score(_report(0, 0.2), "q27")

    assert safe_cheap < safe_costly < unsafe_cheap


def test_run_grid_evaluates_every_point_on_the_validation_pool_and_ranks_best_first():
    seen = []
    reports = {"a": _report(2, 0.1), "b": _report(0, 0.5), "c": _report(0, 0.1)}

    def run_point(point, seeds):
        seen.append((point.name, tuple(seeds)))
        return reports[point.name]

    results = run_grid([GridPoint(name) for name in reports], run_point, "q27")

    pool = load_seed_pool(_REPO / "configs" / "validation_seeds.json", "validation")
    assert [name for name, _ in seen] == ["a", "b", "c"]
    assert all(seeds == pool for _, seeds in seen)
    assert [result.point.name for result in results] == ["c", "b", "a"]


def test_run_grid_can_use_a_prefix_of_the_pool_for_a_cheaper_coarse_pass():
    captured = {}

    def run_point(point, seeds):
        captured["n"] = len(seeds)
        return _report(0, 0.0)

    run_grid([GridPoint("a")], run_point, "q27", max_seeds=7)

    assert captured["n"] == 7


def test_run_grid_refuses_a_non_validation_pool(tmp_path):
    bogus = tmp_path / "pool.json"
    write_seed_pool(bogus, "test", (1, 2, 3))
    config = tmp_path / "validation.yaml"
    config.write_text(yaml.safe_dump({"split": "validation", "seed_pool_file": str(bogus)}))

    with pytest.raises(ValueError, match="expected 'validation'"):
        run_grid([GridPoint("a")], lambda p, s: _report(0, 0.0), "q27", validation_config=config)


def test_freeze_best_writes_a_hashed_overlay_of_the_winning_point(tmp_path):
    results = run_grid(
        [GridPoint("weak", reward_weights={"w4": 1.5}), GridPoint("strong", margin={"alpha": 2.0})],
        lambda point, seeds: _report(0, 0.1) if point.name == "strong" else _report(3, 0.1),
        "q27",
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
    run_grid([GridPoint("a"), GridPoint("b")], lambda p, s: first.append(p.name) or reports[p.name], "q27", log_path=log, settings=_SETTINGS, max_seeds=10)
    second = []

    results = run_grid(
        [GridPoint(n) for n in reports], lambda p, s: second.append(p.name) or reports[p.name], "q27",
        log_path=log, settings=_SETTINGS, max_seeds=10, resume=True,
    )

    assert first == ["a", "b"] and second == ["c"], "only the unfinished point runs again"
    assert [r.point.name for r in results] == ["c", "b", "a"], "the logged points still take part in the ranking"
    assert results[1].report == reports["b"]


def test_a_resumed_grid_reruns_points_logged_under_different_settings(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10)
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "q27",
             log_path=log, settings={**_SETTINGS, "iterations": 50}, max_seeds=10, resume=True)

    assert reran == ["a"]


def test_a_resumed_grid_ignores_a_logged_point_whose_episode_count_does_not_match_the_pool(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1, episodes=3), "q27", log_path=log, settings=_SETTINGS, max_seeds=3)
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1, episodes=7), "q27",
             log_path=log, settings=_SETTINGS, max_seeds=7, resume=True)

    assert reran == ["a"]


def test_a_torn_last_log_line_from_a_killed_run_is_ignored_on_resume(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10)
    with log.open("a") as handle:
        handle.write('{"point": "b", "score": 0.1, "episo')  # the process died mid-write
    reran = []

    run_grid([GridPoint("a"), GridPoint("b")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "q27",
             log_path=log, settings=_SETTINGS, max_seeds=10, resume=True)

    assert reran == ["b"]


def test_without_resume_every_point_runs_even_if_logged(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10)
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10)

    assert reran == ["a"]


def test_q27_score_is_the_collision_rate_plus_a_quarter_of_the_budget_fraction():
    report = _report(1, 0.4, unsafe=3)  # 1/10 collisions, 0.2 of the budget

    assert validation_score(report, "q27") == pytest.approx(0.1 + 0.25 * 0.2)


def test_extended_score_adds_half_the_unsafe_episode_rate():
    report = _report(1, 0.4, unsafe=3)

    assert validation_score(report, "extended") == pytest.approx(0.1 + 0.5 * 0.3 + 0.25 * 0.2)


def test_extended_score_ranks_unsafe_episodes_that_q27_cannot_see():
    unsafe, clean = _report(0, 0.2, unsafe=4), _report(0, 0.2, unsafe=0)

    assert validation_score(unsafe, "q27") == validation_score(clean, "q27")
    assert validation_score(clean, "extended") < validation_score(unsafe, "extended")


def test_an_unknown_score_rule_is_rejected():
    with pytest.raises(ValueError, match="score rule"):
        validation_score(_report(0, 0.1), "weighted")


def test_a_report_without_a_finite_budget_fraction_cannot_be_scored():
    report = dataclasses.replace(_report(0, 0.1), mean_budget_fraction_used=float("nan"))

    with pytest.raises(ValueError, match="mean_budget_fraction_used"):
        validation_score(report, "q27")


def test_run_grid_records_the_score_rule_in_the_logged_settings(tmp_path):
    log = tmp_path / "grid_log.jsonl"

    results = run_grid([GridPoint("a")], lambda p, s: _report(1, 0.4, unsafe=3), "extended", log_path=log, settings=_SETTINGS, max_seeds=10)

    record = json.loads(log.read_text().splitlines()[0])
    assert record["settings"] == {
        **_SETTINGS, "score": "extended", "base_config_digest": config_digest(DEFAULT_CONFIG_PATH),
        "validation_pool_sha256": hashlib.sha256((_REPO / "configs" / "validation_seeds.json").read_bytes()).hexdigest(),
    }
    assert record["score"] == pytest.approx(results[0].score)
    assert results[0].score_rule == "extended"


def test_a_resumed_grid_never_mixes_score_rules(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a"), GridPoint("b")], lambda p, s: _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10)
    reran = []

    run_grid([GridPoint("a"), GridPoint("b")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "extended",
             log_path=log, settings=_SETTINGS, max_seeds=10, resume=True)

    assert reran == ["a", "b"]


def test_freeze_best_records_the_score_rule_and_the_budget_fraction(tmp_path):
    results = run_grid([GridPoint("a")], lambda p, s: _report(0, 0.4, unsafe=2), "extended")
    target = tmp_path / "tuned.yaml"

    freeze_best(results, target)

    validation = yaml.safe_load(target.read_text())["validation"]
    assert validation["score_rule"] == "extended"
    assert validation["mean_budget_fraction_used"] == pytest.approx(0.2)
    assert validation["unsafe_episodes"] == 2


def test_grid_cli_score_rule_defaults_to_q27_and_accepts_extended():
    from orbit_guard.tuning.grid_search import build_parser

    parser = build_parser()

    assert parser.parse_args([]).score == "q27"
    assert parser.parse_args(["--score", "extended"]).score == "extended"
    with pytest.raises(SystemExit):
        parser.parse_args(["--score", "weighted"])


def test_grid_cli_critic_input_defaults_to_the_config_and_accepts_both_variants():
    from orbit_guard.tuning.grid_search import build_parser

    parser = build_parser()

    assert parser.parse_args([]).critic_input is None
    assert parser.parse_args(["--critic-input", "agent_global"]).critic_input == "agent_global"
    with pytest.raises(SystemExit):
        parser.parse_args(["--critic-input", "local"])


@pytest.mark.parametrize(("argv", "expected"), [([], "global"), (["--critic-input", "agent_global"], "agent_global")])
def test_grid_cli_passes_the_critic_input_to_training_and_logs_it(argv, expected, tmp_path, monkeypatch):
    from orbit_guard.rl import evaluation, training
    from orbit_guard.tuning import grid_search

    built, logged = [], {}

    class FakeAlgo:
        def stop(self) -> None:
            return None

    def fake_build_algo(factory, settings):
        built.append(settings)
        return FakeAlgo()

    def fake_run_grid(points, run_point, score_rule, **kwargs):
        logged.update(kwargs["settings"])
        run_point(points[0], [1, 2])
        return []

    monkeypatch.setattr(training, "build_algo", fake_build_algo)
    monkeypatch.setattr(training, "train", lambda algo, iterations: None)
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: _report(0, 0.0))
    monkeypatch.setattr(grid_search, "factory_for_point", lambda *a, **k: None)
    monkeypatch.setattr(grid_search, "run_grid", fake_run_grid)
    monkeypatch.setattr(grid_search, "freeze_best", lambda results, path: "digest")

    grid_search.main(["--out-dir", str(tmp_path), "--only", "default", *argv])

    assert [settings.critic_input for settings in built] == [expected]
    assert logged["critic_input"] == expected


# --- resume never reuses a point scored under another base config or seed pool (HARDEN-2) ---------------------


def _validation_config(tmp_path) -> Path:
    pool = tmp_path / "validation_seeds.json"
    write_seed_pool(pool, "validation", generate_validation_pool())
    config = tmp_path / "validation.yaml"
    config.write_text(yaml.safe_dump({"split": "validation", "seed_pool_file": str(pool)}))
    return config


def test_a_resumed_grid_reruns_points_logged_under_another_base_config(tmp_path):
    base = tmp_path / "default.yaml"
    base.write_bytes(DEFAULT_CONFIG_PATH.read_bytes())
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10,
             base_config=base)
    base.write_bytes(base.read_bytes() + b"# edited after the first grid\n")
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "q27",
             log_path=log, settings=_SETTINGS, max_seeds=10, base_config=base, resume=True)

    assert reran == ["a"]


def test_a_resumed_grid_reruns_points_logged_under_another_validation_pool_file(tmp_path):
    validation = _validation_config(tmp_path)
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10,
             validation_config=validation)
    pool = tmp_path / "validation_seeds.json"
    pool.write_text(pool.read_text() + " ")  # same seeds, different file bytes
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "q27",
             log_path=log, settings=_SETTINGS, max_seeds=10, validation_config=validation, resume=True)

    assert reran == ["a"]


def test_a_resumed_grid_still_skips_points_when_base_config_and_pool_are_unchanged(tmp_path):
    validation = _validation_config(tmp_path)
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1), "q27", log_path=log, settings=_SETTINGS, max_seeds=10,
             validation_config=validation)
    reran = []

    run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "q27",
             log_path=log, settings=_SETTINGS, max_seeds=10, validation_config=validation, resume=True)

    assert reran == []
