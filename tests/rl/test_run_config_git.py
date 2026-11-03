"""The training sidecar records the git state the run STARTED from, and flags code that moved while it ran.

`git_commit` is HEAD when training starts (Ray workers import the code then, not when the run ends), and
`git_dirty` is the porcelain status of orbit_guard/, configs/ and data/ at that moment plus a line for any change
under those paths while the run was in progress. The freeze refuses a checkpoint whose `git_dirty` is not empty.
Git runs in a throwaway repository.
"""

from __future__ import annotations

import json
from pathlib import Path

from orbit_guard.evaluation.freeze import GitTree
from orbit_guard.rl import training
from orbit_guard.rl.evaluation import EvaluationReport
from orbit_guard.rl.training import (
    RUN_CONFIG_FILENAME,
    TrainingReport,
    git_drift,
    git_state,
)
from tests.evaluation.git_repo_fixture import commit_all, git, make_repo


def test_a_clean_tree_records_head_and_no_dirty_paths(tmp_path):
    repo = make_repo(tmp_path / "repo")

    state = git_state(GitTree(repo))

    assert state == {"git_commit": git(repo, "rev-parse", "HEAD"), "git_dirty": []}


def test_dirty_code_configs_or_data_are_recorded_but_docs_and_scripts_are_not(tmp_path):
    repo = make_repo(tmp_path / "repo")
    for relative in ("orbit_guard/module.py", "configs/default.yaml", "docs/notes.md", "scripts/run.sh"):
        (repo / relative).write_text("edited\n")
    (repo / "data" / "new.csv").write_text("untracked\n")

    dirty = git_state(GitTree(repo))["git_dirty"]

    assert any("orbit_guard/module.py" in line for line in dirty)
    assert any("configs/default.yaml" in line for line in dirty)
    assert any("data/new.csv" in line for line in dirty)
    assert not any("docs/" in line or "scripts/" in line for line in dirty)


def test_without_git_the_state_is_unknown_not_clean(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    assert git_state(GitTree(plain)) == {"git_commit": None, "git_dirty": None}


def test_no_drift_when_nothing_under_the_training_paths_changed(tmp_path):
    repo = make_repo(tmp_path / "repo")
    start = git_state(GitTree(repo))
    (repo / "docs" / "notes.md").write_text("docs commit while training\n")
    commit_all(repo, "docs")

    assert git_drift(GitTree(repo), start) == []


def test_code_committed_while_training_ran_is_drift(tmp_path):
    repo = make_repo(tmp_path / "repo")
    start = git_state(GitTree(repo))
    (repo / "orbit_guard" / "module.py").write_text("VALUE = 2\n")
    commit_all(repo, "fix committed mid-run")

    drift = git_drift(GitTree(repo), start)

    assert drift and start["git_commit"] in drift[0]


def test_a_new_untracked_code_file_while_training_ran_is_drift(tmp_path):
    repo = make_repo(tmp_path / "repo")
    start = git_state(GitTree(repo))
    (repo / "orbit_guard" / "new_module.py").write_text("X = 1\n")

    assert any("orbit_guard/new_module.py" in line for line in git_drift(GitTree(repo), start))


def _report() -> EvaluationReport:
    return EvaluationReport(
        episodes=1, mean_return_per_agent=0.0, collision_episodes=0, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=0.0, mean_budget_fraction_used=0.0, unsafe_episodes=0, mean_episode_steps=1.0,
    )


class _FakeAlgo:
    def stop(self) -> None:
        return None


def test_the_cli_records_the_commit_it_started_from_and_flags_a_mid_run_commit(tmp_path, monkeypatch):
    repo = make_repo(tmp_path / "repo")
    started_at = git(repo, "rev-parse", "HEAD")
    checkpoint = tmp_path / "run"

    def fake_train(algo, num_iterations, checkpoint_dir=None, on_iteration=None):
        (repo / "orbit_guard" / "module.py").write_text("VALUE = 3\n")
        commit_all(repo, "committed while the run trained")  # the review's af24fa2-during-R4 scenario
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return TrainingReport(iterations=num_iterations, episode_return_means=(), env_steps_sampled=0,
                              checkpoint_path=str(checkpoint_dir))

    monkeypatch.setattr(training, "_repository", lambda: GitTree(repo))
    monkeypatch.setattr(training, "build_algo", lambda env_factory, settings: _FakeAlgo())
    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate_policy", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.gate_verdict", lambda *a, **k: "PASS")

    training.main([
        "--agents", "2", "--iterations", "1", "--eval-episodes", "1", "--num-env-runners", "0",
        "--checkpoint-dir", str(checkpoint),
    ])

    sidecar = json.loads((checkpoint / RUN_CONFIG_FILENAME).read_text())
    assert sidecar["git_commit"] == started_at
    assert sidecar["git_dirty"] and started_at in sidecar["git_dirty"][0]
