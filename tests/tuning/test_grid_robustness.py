"""Grid robustness: split (--only) runs, code/runner stamps, failed episodes and the abort-then-resume path."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import yaml

from orbit_guard.rl.evaluation import EvaluationReport
from orbit_guard.tuning import grid_search
from orbit_guard.tuning.grid_search import (
    GridPoint,
    default_grid,
    freeze_best,
    run_grid,
    validation_score,
)

_BUDGET_MPS = 2.0
_SETTINGS = {"agents": 15, "iterations": 25, "validation_episodes": 10, "seed": 0, "shield": True}
_CODE = {"git_commit": "c0ffee", "git_dirty": False, "git_diff_sha256": None}


def _report(collisions: int, delta_v: float, episodes: int = 10, *, failed: int = 0) -> EvaluationReport:
    ok = episodes - failed
    return EvaluationReport(
        episodes=episodes, mean_return_per_agent=0.0 if ok else math.nan, collision_episodes=collisions,
        maneuver_rate=0.0, threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=delta_v if ok else math.nan, mean_episode_steps=360.0 if ok else math.nan,
        mean_budget_fraction_used=delta_v / _BUDGET_MPS if ok else math.nan, unsafe_episodes=0, failed_episodes=failed,
    )


# --- failed episodes: scored over the finished ones, never the winner while a clean point exists -----------


def test_the_score_of_a_report_with_failed_episodes_uses_only_the_finished_episodes():
    report = _report(1, 0.4, episodes=10, failed=5)  # 1 collision in 5 finished episodes

    assert validation_score(report, "q27") == pytest.approx(1 / 5 + 0.25 * 0.2)


def test_a_point_whose_every_episode_failed_scores_infinity_instead_of_raising():
    assert validation_score(_report(0, 0.0, episodes=10, failed=10), "q27") == math.inf


def test_a_point_with_failed_episodes_is_logged_with_its_failed_count(tmp_path):
    log = tmp_path / "grid_log.jsonl"

    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1, failed=2), "q27", log_path=log, settings=_SETTINGS, max_seeds=10)

    assert json.loads(log.read_text().splitlines()[0])["failed_episodes"] == 2


def test_a_logged_point_with_failed_episodes_is_reused_on_resume_not_rerun(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    run_grid([GridPoint("a")], lambda p, s: _report(0, 0.1, failed=2), "q27", log_path=log, settings=_SETTINGS, max_seeds=10)
    reran = []

    results = run_grid([GridPoint("a")], lambda p, s: reran.append(p.name) or _report(0, 0.1), "q27",
                       log_path=log, settings=_SETTINGS, max_seeds=10, resume=True)

    assert reran == [], "a point with failures is a finished point; rerunning it would repeat the failure forever"
    assert results[0].report.failed_episodes == 2


def test_a_point_with_failed_episodes_never_wins_while_a_clean_point_exists(tmp_path):
    reports = {"flaky_best": _report(0, 0.0, failed=1), "clean_worse": _report(3, 0.5)}

    results = run_grid([GridPoint(n) for n in reports], lambda p, s: reports[p.name], "q27")
    target = tmp_path / "tuned.yaml"
    freeze_best(results, target)

    assert results[0].score > results[1].score, "the flaky point has the better score ..."
    assert results[0].point.name == "clean_worse", "... but is ranked after every clean point"
    assert yaml.safe_load(target.read_text())["selected"] == "clean_worse"


def test_freeze_best_applies_the_failure_rule_even_to_an_unsorted_result_list(tmp_path):
    flaky = grid_search.GridResult(GridPoint("flaky"), _report(0, 0.0, failed=1), 0.0, "q27")
    clean = grid_search.GridResult(GridPoint("clean"), _report(3, 0.5), 0.425, "q27")
    target = tmp_path / "tuned.yaml"

    freeze_best([flaky, clean], target)

    assert yaml.safe_load(target.read_text())["selected"] == "clean"


def test_when_every_point_has_failed_episodes_the_best_score_wins_and_the_count_is_frozen(tmp_path):
    reports = {"a": _report(0, 0.1, failed=1), "b": _report(2, 0.1, failed=3)}

    results = run_grid([GridPoint(n) for n in reports], lambda p, s: reports[p.name], "q27")
    target = tmp_path / "tuned.yaml"
    freeze_best(results, target)

    frozen = yaml.safe_load(target.read_text())
    assert frozen["selected"] == "a"
    assert frozen["validation"]["failed_episodes"] == 1


def test_freeze_best_refuses_when_no_point_finished_a_single_episode(tmp_path):
    results = run_grid([GridPoint("a"), GridPoint("b")], lambda p, s: _report(0, 0.0, failed=10), "q27", max_seeds=10)
    target = tmp_path / "tuned.yaml"

    with pytest.raises(ValueError, match="no grid point finished"):
        freeze_best(results, target)
    assert not target.exists()


# --- --only: rank the union of logged points, freeze only a complete grid ---------------------------------------


def _fake_training(monkeypatch, reports, *, trained, fail_on=None, stopped=None):
    """Stub out Ray: build_algo/train/evaluate return `reports[point]`; `train` raises for the point `fail_on`."""
    from orbit_guard.rl import evaluation, training

    current = {}

    class FakeAlgo:
        def stop(self) -> None:
            if stopped is not None:
                stopped.append(current["point"])

    def fake_factory(point, *args, **kwargs):
        current["point"] = point.name
        return point.name

    def fake_train(algo, iterations):
        trained.append(current["point"])
        if current["point"] == fail_on:
            raise RuntimeError("iteration 3 made no learner update (no samples returned)")

    monkeypatch.setattr(training, "build_algo", lambda factory, settings: FakeAlgo())
    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr(evaluation, "evaluate", lambda algo, factory, n, seeds: reports[factory])
    monkeypatch.setattr(grid_search, "factory_for_point", fake_factory)
    monkeypatch.setattr(grid_search, "code_state", lambda: dict(_CODE))


def _reports(best: str = "default") -> dict[str, EvaluationReport]:
    """One report per default_grid() point; `best` is the global minimum (default margins: no Q30 measurement)."""
    reports = {point.name: _report(index + 1, 0.1) for index, point in enumerate(default_grid())}
    return {**reports, best: _report(0, 0.0)}


def _main(tmp_path, *argv: str) -> None:
    grid_search.main(["--out-dir", str(tmp_path), "--validation-episodes", "10", *argv])  # = the fake reports' episodes


def test_an_only_subset_does_not_freeze_while_grid_points_are_missing(tmp_path, monkeypatch, capsys):
    _fake_training(monkeypatch, _reports(), trained=[])
    previous = tmp_path / "tuned.yaml"
    previous.write_text("selected: from_an_earlier_complete_grid\n")

    _main(tmp_path, "--only", "margin_narrow", "w3_10")

    assert previous.read_text() == "selected: from_an_earlier_complete_grid\n", "tuned.yaml left untouched"
    out = capsys.readouterr().out
    assert "not freezing" in out
    assert "default" in out and "w4_1.5" in out, "the missing points are named"


def test_a_grid_split_over_only_invocations_freezes_the_global_winner(tmp_path, monkeypatch):
    names = [point.name for point in default_grid()]
    reports = _reports(best="default")
    trained = []
    _fake_training(monkeypatch, reports, trained=trained)

    _main(tmp_path, "--only", *names[:4])
    assert not (tmp_path / "tuned.yaml").exists()
    _main(tmp_path, "--only", *names[4:], "--resume")

    assert trained == names, "each point trained exactly once"
    assert yaml.safe_load((tmp_path / "tuned.yaml").read_text())["selected"] == "default", (
        "the winner of the WHOLE grid, not of the last --only subset"
    )


def test_a_debug_only_run_refreezes_the_global_winner_not_its_own_point(tmp_path, monkeypatch):
    _fake_training(monkeypatch, _reports(best="default"), trained=[])
    _main(tmp_path)
    frozen = (tmp_path / "tuned.yaml").read_text()
    trained = []
    _fake_training(monkeypatch, _reports(best="default"), trained=trained)

    _main(tmp_path, "--only", "w3_10")  # no --resume: w3_10 retrains and is relogged; the rest come from the log

    assert trained == ["w3_10"]
    assert (tmp_path / "tuned.yaml").read_text() == frozen, "still the whole grid's winner, byte for byte"


def test_only_ranking_ignores_logged_points_from_other_settings(tmp_path, monkeypatch, capsys):
    _fake_training(monkeypatch, _reports(), trained=[])
    _main(tmp_path, "--iterations", "50")  # a full grid under OTHER settings

    _main(tmp_path, "--only", "default")

    out = capsys.readouterr().out
    assert out.count("frozen config hash") == 1 and "not freezing" in out


def test_run_grid_ranks_logged_points_it_was_not_asked_to_run(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    reports = {"a": _report(0, 0.0), "b": _report(1, 0.1), "c": _report(2, 0.1)}
    run_grid([GridPoint("a")], lambda p, s: reports[p.name], "q27", log_path=log, settings=_SETTINGS, max_seeds=10)
    ran = []

    results = run_grid([GridPoint("b")], lambda p, s: ran.append(p.name) or reports[p.name], "q27", log_path=log,
                       settings=_SETTINGS, max_seeds=10, rank_logged=[GridPoint("a"), GridPoint("c")])

    assert ran == ["b"], "logged points are never run, and unlogged ones (c) are not run either"
    assert [r.point.name for r in results] == ["a", "b"]


# --- the logged settings carry the runner count and the code stamp ----------------------------------------------


def test_the_grid_cli_logs_the_runner_count_and_the_code_stamp(tmp_path, monkeypatch):
    _fake_training(monkeypatch, _reports(), trained=[])

    _main(tmp_path, "--only", "default", "--num-env-runners", "4")

    settings = json.loads((tmp_path / "grid_log.jsonl").read_text().splitlines()[0])["settings"]
    assert settings["num_env_runners"] == 4
    assert {key: settings[key] for key in _CODE} == _CODE


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        (["--num-env-runners", "4"], _CODE),
        ([], {**_CODE, "git_commit": "badc0de"}),
        ([], {**_CODE, "git_dirty": True, "git_diff_sha256": "f" * 64}),
    ],
    ids=["runner_count", "commit", "dirty_diff"],
)
def test_a_resumed_grid_reruns_points_trained_with_other_runners_or_code(tmp_path, monkeypatch, argv, code):
    trained = []
    _fake_training(monkeypatch, _reports(), trained=trained)
    _main(tmp_path, "--only", "default")
    monkeypatch.setattr(grid_search, "code_state", lambda: dict(code))

    _main(tmp_path, "--only", "default", "--resume", *argv)

    assert trained == ["default", "default"]


# --- the per-iteration abort (train() raises on an iteration with no learner update) leaves the grid resumable ---


def test_an_aborted_point_is_not_logged_and_a_resumed_grid_continues_from_it(tmp_path, monkeypatch):
    names = [point.name for point in default_grid()]
    failing = names[2]
    trained, stopped = [], []
    _fake_training(monkeypatch, _reports(), trained=trained, fail_on=failing, stopped=stopped)

    with pytest.raises(RuntimeError, match="no learner update"):
        _main(tmp_path)

    assert trained == names[:3] and stopped == names[:3], "the failed point's algo is still stopped"
    logged = [json.loads(line)["point"] for line in (tmp_path / "grid_log.jsonl").read_text().splitlines()]
    assert logged == names[:2], "the aborted point is not logged"
    assert not (tmp_path / "tuned.yaml").exists()

    trained.clear()
    _fake_training(monkeypatch, _reports(), trained=trained)
    _main(tmp_path, "--resume")

    assert trained == names[2:], "the finished points are reused; the aborted one and the rest run"
    assert yaml.safe_load((tmp_path / "tuned.yaml").read_text())["selected"] == "default"


def test_report_failed_episodes_round_trips_through_the_grid_log(tmp_path):
    log = tmp_path / "grid_log.jsonl"
    report = _report(1, 0.2, failed=3)
    run_grid([GridPoint("a")], lambda p, s: report, "q27", log_path=log, settings=_SETTINGS, max_seeds=10)

    results = run_grid([GridPoint("a")], lambda p, s: pytest.fail("must not rerun"), "q27", log_path=log,
                       settings=_SETTINGS, max_seeds=10, resume=True)

    assert dataclasses.asdict(results[0].report) == dataclasses.asdict(report)
