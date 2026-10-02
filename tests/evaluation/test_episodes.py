"""Episode discovery from trace.meta (redesign real-data rule 3/4; EC-02, EC-03, EC-26, EC-28).

Traces are indexed by what they SAY they are. File names are deliberately scrambled in these tests to prove they
are never consulted."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from orbit_guard.visualization.episodes import (
    NoEpisodesError,
    condition_label,
    discover_episodes,
    index_from_trace,
)
from orbit_guard.visualization.replay import EpisodeTrace, save_trace
from orbit_guard.visualization.replay_meta import ReplayMetaError
from tests.evaluation.replay_fixtures import (
    capability_trace,
    collision_trace,
    shielded_trace,
    unshielded_trace,
)


def _with_meta(trace: EpisodeTrace, **changes) -> EpisodeTrace:
    return dataclasses.replace(trace, meta={**trace.meta, **changes})


def _spy_on_npz_member_reads(monkeypatch) -> list[str]:
    """Record which archive members `np.load` hands out, to prove discovery reads headers only."""
    real_load = np.load
    touched: list[str] = []

    class Spy:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getitem__(self, key):
            touched.append(key)
            return self.handle[key]

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(np, "load", lambda *a, **k: Spy(real_load(*a, **k)))
    return touched


@pytest.fixture
def directory(tmp_path):
    # shielded = (normal:0, trained, shield on); unshielded = (normal:0, trained, shield off)
    save_trace(shielded_trace(), tmp_path / "zzz.npz")
    save_trace(unshielded_trace(), tmp_path / "aaa.npz")
    save_trace(collision_trace(), tmp_path / "middle.npz")
    return tmp_path


def test_index_is_built_from_meta_not_filenames(directory):
    index = discover_episodes(directory)
    assert len(index.entries) == 3
    on = index.find("normal:0", "trained", True)
    off = index.find("normal:0", "trained", False)
    assert on.path.name == "zzz.npz" and off.path.name == "aaa.npz"
    assert on.meta.shield is True and off.meta.shield is False


def test_missing_combination_is_none(directory):
    index = discover_episodes(directory)
    assert index.find("normal:0", "untrained", True) is None
    assert index.find("nope:1", "trained", True) is None


def test_scenarios_are_unique_and_sorted(directory):
    scenarios = discover_episodes(directory).scenarios()
    assert [s.scenario_id for s in scenarios] == ["normal:0", "simultaneous_conjunctions:3"]
    assert scenarios[0].condition == "normal" and scenarios[0].seed == 0
    assert scenarios[0].label == "Normal"


def test_availability_marks_missing_combinations_as_false(directory):
    availability = discover_episodes(directory).availability("normal:0")
    assert availability[("trained", True)] is True
    assert availability[("trained", False)] is True
    assert availability[("no_maneuver", False)] is False  # exists for another scenario, not this one


def test_before_and_after_are_always_offered_and_baselines_come_from_the_data(directory):
    index = discover_episodes(directory)

    assert index.policies() == ("untrained", "trained", "no_maneuver")  # EC-03: a missing BEFORE is disabled, not absent
    assert index.availability("normal:0")[("untrained", True)] is False


def test_empty_directory_is_an_empty_index_not_an_error(tmp_path):
    index = discover_episodes(tmp_path)
    assert index.entries == () and index.scenarios() == () and index.policies() == ("untrained", "trained")


def test_missing_directory_raises(tmp_path):
    with pytest.raises(NoEpisodesError, match="not a directory"):
        discover_episodes(tmp_path / "absent")


def test_old_trace_without_meta_fails_loudly_with_the_path(tmp_path):
    old = dataclasses.replace(shielded_trace(), meta={"model": "demo", "condition": "normal", "seed": 1})
    save_trace(old, tmp_path / "old.npz")
    with pytest.raises(ReplayMetaError, match="old.npz"):
        discover_episodes(tmp_path)


def test_duplicate_combinations_are_ambiguous(tmp_path):
    save_trace(shielded_trace(), tmp_path / "a.npz")
    save_trace(shielded_trace(), tmp_path / "b.npz")
    with pytest.raises(ReplayMetaError, match="duplicate"):
        discover_episodes(tmp_path)


def test_non_trace_files_are_ignored(directory):
    (directory / "notes.txt").write_text("hello")
    assert len(discover_episodes(directory).entries) == 3


def test_discovery_reads_headers_only(directory, monkeypatch):
    """Cheap enough to index a directory of full episodes: array members are never touched."""
    touched = _spy_on_npz_member_reads(monkeypatch)
    discover_episodes(directory)
    assert set(touched) == {"header"}


def test_index_from_a_single_trace_requires_meta(tmp_path):
    save_trace(capability_trace(), tmp_path / "one.npz")
    index = index_from_trace(tmp_path / "one.npz")
    assert [s.scenario_id for s in index.scenarios()] == ["capability_reduction:1"]
    save_trace(dataclasses.replace(capability_trace(), meta={"model": "demo"}), tmp_path / "old.npz")
    with pytest.raises(ReplayMetaError, match="record_replay_traces"):
        index_from_trace(tmp_path / "old.npz")


def test_condition_labels_are_humanised_with_known_overrides():
    assert condition_label("normal") == "Normal"
    assert condition_label("capability_reduction") == "Capability reduction"
    assert condition_label("simultaneous_conjunctions") == "Simultaneous conjunctions"
    assert condition_label("some_new_shift") == "Some new shift"


def test_condition_order_mirrors_the_config_shift_names():
    import yaml

    from orbit_guard.visualization.episodes import CONDITION_ORDER

    shifts = yaml.safe_load(Path("configs/default.yaml").read_text())["shifts"]
    assert CONDITION_ORDER == tuple(shifts)


def test_episode_summary_is_read_from_the_header_only(directory, monkeypatch):
    from orbit_guard.visualization.episodes import read_episode_summary

    touched = _spy_on_npz_member_reads(monkeypatch)
    on = read_episode_summary(directory / "zzz.npz")
    off = read_episode_summary(directory / "aaa.npz")
    assert set(touched) == {"header"}
    assert on.shield == shielded_trace().summary["shield_intervention_count"]
    assert off.shield is None  # shield off: "not applicable"


def test_a_non_trace_npz_is_a_clear_error_naming_the_file(tmp_path):
    np.savez(tmp_path / "weights.npz", w0=np.zeros(3))
    with pytest.raises(ReplayMetaError, match=r"weights\.npz.*not a replay trace"):
        discover_episodes(tmp_path)
