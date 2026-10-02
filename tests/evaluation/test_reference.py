"""Reference collision note from an evaluation log (real-data rule 6, EC-24): hidden unless a log produced by our
evaluation runner is supplied, and it only ever mentions models the log actually contains."""

from __future__ import annotations

import json

from orbit_guard.visualization.reference import reference_note


def _write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _rec(model, condition, collisions, status="ok", n_satellites=15):
    return {"model_name": model, "shift_type": condition, "status": status, "collision_count": collisions, "n_satellites": n_satellites}


def test_no_log_means_no_note():
    assert reference_note(None, "normal", ["adaptive_mappo"]) is None


def test_note_lists_each_present_model_with_its_collision_episode_fraction(tmp_path):
    log = _write(tmp_path / "e.jsonl", [_rec("no_maneuver", "normal", 1)] + [_rec("no_maneuver", "normal", 0)] * 9
                 + [_rec("adaptive_mappo", "normal", 0)] * 10)
    assert reference_note(log, "normal", ["adaptive_mappo", "no_maneuver"]) == "adaptive_mappo 0/10 · no_maneuver 1/10 episodes with a collision"


def test_other_conditions_and_failed_episodes_are_excluded(tmp_path):
    log = _write(tmp_path / "e.jsonl", [_rec("no_maneuver", "normal", 1), _rec("no_maneuver", "noise_increase", 1),
                                        _rec("no_maneuver", "normal", 1, status="failed")])
    assert reference_note(log, "normal", ["no_maneuver"]) == "no_maneuver 1/1 episodes with a collision"


def test_models_absent_from_the_log_are_not_invented(tmp_path):
    log = _write(tmp_path / "e.jsonl", [_rec("no_maneuver", "normal", 0)])
    note = reference_note(log, "normal", ["untrained_policy", "no_maneuver"])
    assert note == "no_maneuver 0/1 episodes with a collision" and "untrained" not in note


def test_nothing_matching_means_no_note(tmp_path):
    log = _write(tmp_path / "e.jsonl", [_rec("no_maneuver", "normal", 0)])
    assert reference_note(log, "capability_reduction", ["no_maneuver"]) is None


def test_a_missing_log_file_is_a_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="reference log"):
        reference_note(tmp_path / "absent.jsonl", "normal", ["no_maneuver"])


def test_only_the_requested_fleet_size_is_counted(tmp_path):
    log = _write(tmp_path / "e.jsonl", [_rec("no_maneuver", "normal", 1, n_satellites=2), _rec("no_maneuver", "normal", 0, n_satellites=15)])

    assert reference_note(log, "normal", ["no_maneuver"], n_agents=15) == "no_maneuver 0/1 episodes with a collision"
    assert reference_note(log, "normal", ["no_maneuver"], n_agents=2) == "no_maneuver 1/1 episodes with a collision"


def test_malformed_records_are_reported_with_file_and_line(tmp_path):
    import pytest

    from orbit_guard.visualization.reference import ReferenceLogError

    cases = {
        "{not json\n": "e.jsonl:1",
        "[1, 2]\n": "e.jsonl:1",
        json.dumps({**_rec("no_maneuver", "normal", 0), "collision_count": None}) + "\n": "collision_count",
    }
    for text, expected in cases.items():
        (tmp_path / "e.jsonl").write_text(text)
        with pytest.raises(ReferenceLogError, match=expected):
            reference_note(tmp_path / "e.jsonl", "normal", ["no_maneuver"])
