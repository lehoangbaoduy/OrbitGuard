"""orbit_guard.tuning.code_state: the code stamp a grid point is logged under (commit + exact uncommitted diff)."""

from __future__ import annotations

from pathlib import Path

from orbit_guard.tuning.code_state import code_state
from tests.evaluation.git_repo_fixture import commit_all, git, make_repo


def test_a_clean_tree_is_stamped_with_its_commit_and_no_diff(tmp_path):
    root = make_repo(tmp_path / "repo")

    state = code_state(root)

    assert state == {"git_commit": git(root, "rev-parse", "HEAD"), "git_dirty": False, "git_diff_sha256": None}


def test_an_edit_to_tracked_code_marks_the_tree_dirty_and_hashes_the_diff(tmp_path):
    root = make_repo(tmp_path / "repo")
    (root / "orbit_guard" / "module.py").write_text("VALUE = 2\n")

    state = code_state(root)

    assert state["git_dirty"] is True
    assert isinstance(state["git_diff_sha256"], str) and len(state["git_diff_sha256"]) == 64


def test_two_different_dirty_trees_on_the_same_commit_get_different_stamps(tmp_path):
    root = make_repo(tmp_path / "repo")
    module = root / "orbit_guard" / "module.py"
    module.write_text("VALUE = 2\n")
    first = code_state(root)
    module.write_text("VALUE = 3\n")

    second = code_state(root)

    assert first["git_commit"] == second["git_commit"]
    assert first != second, "a bare dirty flag would let these two code versions be ranked together"


def test_the_same_dirty_tree_gets_the_same_stamp_twice(tmp_path):
    root = make_repo(tmp_path / "repo")
    (root / "configs" / "default.yaml").write_text("a: 2\n")

    assert code_state(root) == code_state(root)


def test_an_untracked_file_under_a_code_path_counts_and_its_content_is_hashed(tmp_path):
    root = make_repo(tmp_path / "repo")
    extra = root / "orbit_guard" / "new_module.py"
    extra.write_text("X = 1\n")
    first = code_state(root)
    extra.write_text("X = 2\n")

    second = code_state(root)

    assert first["git_dirty"] is True
    assert first["git_diff_sha256"] != second["git_diff_sha256"]


def test_edits_outside_the_code_paths_leave_the_stamp_clean(tmp_path):
    root = make_repo(tmp_path / "repo")
    (root / "docs" / "notes.md").write_text("another agent is editing the docs\n")
    (root / "grid_runs").mkdir()
    (root / "grid_runs" / "grid_log.jsonl").write_text("{}\n")  # the grid's own default out-dir, untracked

    assert code_state(root)["git_dirty"] is False


def test_a_new_commit_changes_the_stamp(tmp_path):
    root = make_repo(tmp_path / "repo")
    before = code_state(root)
    (root / "orbit_guard" / "module.py").write_text("VALUE = 2\n")
    commit_all(root, "change")

    after = code_state(root)

    assert after["git_commit"] != before["git_commit"] and after["git_dirty"] is False


def test_a_directory_outside_any_repository_is_stamped_unknown(tmp_path):
    outside = tmp_path / "plain"
    outside.mkdir()

    assert code_state(Path(outside)) == {"git_commit": None, "git_dirty": None, "git_diff_sha256": None}
