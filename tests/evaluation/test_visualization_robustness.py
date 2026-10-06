"""Failure paths of the replay app: corrupt traces, concurrent payload builds, CLI errors, reference-log guards."""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dash")

from orbit_guard.visualization import app as app_module
from orbit_guard.visualization.app import create_app, main
from orbit_guard.visualization.callbacks import PayloadProvider
from orbit_guard.visualization.episodes import discover_episodes
from orbit_guard.visualization.reference import ReferenceLogError, reference_note
from orbit_guard.visualization.replay import load_trace, save_trace
from orbit_guard.visualization.replay_meta import ReplayMetaError
from orbit_guard.visualization.selection import Selection
from tests.evaluation.replay_fixtures import shielded_trace


@pytest.fixture()
def trace_path(tmp_path) -> Path:
    path = tmp_path / "ep.npz"
    save_trace(shielded_trace(), path)
    return path


def _rewrite(path: Path, *, drop: str | None = None, header_patch: dict | None = None, replace: dict | None = None) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if drop:
        del arrays[drop]
    if header_patch is not None:
        arrays["header"] = np.array(json.dumps({**json.loads(str(arrays["header"])), **header_patch}))
    arrays.update(replace or {})
    np.savez_compressed(path, **arrays)


def test_a_trace_missing_an_array_names_the_file(trace_path):
    _rewrite(trace_path, drop="risk")

    with pytest.raises(ReplayMetaError, match=r"ep\.npz.*risk"):
        load_trace(trace_path)


def test_a_trace_with_inconsistent_array_shapes_names_the_file(trace_path):
    _rewrite(trace_path, replace={"risk": np.zeros((2, 1))})

    with pytest.raises(ReplayMetaError, match=r"ep\.npz"):
        load_trace(trace_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0, "x"])
def test_a_non_positive_or_non_finite_control_dt_is_refused(trace_path, bad):
    _rewrite(trace_path, header_patch={"control_dt_s": bad})

    with pytest.raises(ReplayMetaError, match=r"control_dt_s"):
        load_trace(trace_path)


def test_the_payload_cache_survives_concurrent_requests_with_a_tiny_cache(tmp_path, monkeypatch):
    from orbit_guard.visualization import callbacks

    save_trace(shielded_trace(), tmp_path / "a.npz")
    index = discover_episodes(tmp_path)
    monkeypatch.setattr(callbacks, "_CACHE_SIZE", 1)
    provider = PayloadProvider(index)
    entry = index.entries[0].meta
    selection = Selection(entry.scenario_id, entry.policy, entry.shield)
    errors: list[BaseException] = []

    def hammer():
        try:
            for _ in range(25):
                assert provider(selection) is not None
        except BaseException as error:  # noqa: BLE001 - collected and asserted below
            errors.append(error)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert errors == []


def test_a_trace_damaged_after_indexing_raises_a_named_error_from_the_provider(tmp_path):
    save_trace(shielded_trace(), tmp_path / "ok.npz")
    index = discover_episodes(tmp_path)
    _rewrite(tmp_path / "ok.npz", drop="risk")
    meta = index.entries[0].meta

    with pytest.raises(ReplayMetaError, match=r"ok\.npz"):
        PayloadProvider(index)(Selection(meta.scenario_id, meta.policy, meta.shield))


def test_a_click_on_a_damaged_episode_answers_200_with_an_alert_and_changes_nothing_else(tmp_path):
    base = shielded_trace()
    save_trace(_with_policy(base, "untrained"), tmp_path / "a.npz")
    save_trace(_with_policy(base, "trained"), tmp_path / "b.npz")
    index = discover_episodes(tmp_path)
    app = create_app(index)  # builds the initial payload while both files are healthy
    victim = next(e for e in index.entries if e.meta.policy == "trained")  # the initial selection (untrained) is cached
    _rewrite(victim.path, drop="risk")
    meta = index.entries[0].meta

    response = _click(app, index, {"scenario_id": meta.scenario_id, "policy": "untrained", "shield": True}, "trained")

    assert response.status_code == 200, response.get_data(as_text=True)
    out = response.get_json()["response"]
    assert "Could not load this episode" in out["og-alert"]["children"] and "b.npz" in out["og-alert"]["children"]
    assert "og-selection" not in out and "og-payload" not in out


def _with_policy(trace, policy):
    from orbit_guard.visualization.replay_meta import build_replay_meta

    base = trace.meta
    extra = {"weights_sha256": "a" * 64} if policy == "trained" else {"init_seed": 0}
    meta = build_replay_meta(
        condition=base["condition"], seed=base["seed"], policy=policy, shield=True, model=base["model"], n_agents=base["n_agents"],
        delta_v_budget_mps=base["delta_v_budget_mps"], max_delta_v_per_action_mps=base["max_delta_v_per_action_mps"], **extra,
    )
    return dataclasses.replace(trace, meta=meta)


def _click(app, index, state, policy):
    from tests.evaluation.test_visualization_app import _expand_outputs, _wildcard_ids

    key = next(k for k in app.callback_map if "og-selection.data" in k)
    inputs = []
    for item in app.callback_map[key]["inputs"]:
        kind = json.loads(item["id"])["type"]
        inputs.append([{"id": {"type": kind, "id": i}, "property": "n_clicks", "value": 0} for i in _wildcard_ids(index, kind)])
    body = {"output": key, "outputs": _expand_outputs(index, key), "inputs": inputs,
            "state": [{"id": "og-selection", "property": "data", "value": state}],
            "changedPropIds": [json.dumps({"id": policy, "type": "og-policy"}, separators=(",", ":")) + ".n_clicks"]}
    return app.server.test_client().post("/_dash-update-component", json=body)


def test_main_turns_a_missing_directory_into_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--episodes", str(tmp_path / "nope")])

    assert exit_info.value.code == 2
    assert "not a directory" in capsys.readouterr().err


def test_main_turns_a_bad_trace_into_a_usage_error(tmp_path, capsys):
    (tmp_path / "bad.npz").write_bytes(b"not a zip")

    with pytest.raises(SystemExit) as exit_info:
        main(["--episodes", str(tmp_path)])

    assert exit_info.value.code == 2
    assert "bad.npz" in capsys.readouterr().err


def test_main_checks_the_reference_log_before_serving(tmp_path, capsys, monkeypatch):
    save_trace(shielded_trace(), tmp_path / "a.npz")
    monkeypatch.setattr(app_module.Dash, "run", lambda *a, **k: pytest.fail("must not serve with a bad reference log"))

    with pytest.raises(SystemExit) as exit_info:
        main(["--episodes", str(tmp_path), "--reference-log", str(tmp_path / "missing.jsonl")])

    assert exit_info.value.code == 2
    assert "missing.jsonl" in capsys.readouterr().err


def test_a_non_string_model_name_in_the_reference_log_is_ignored(tmp_path):
    log = tmp_path / "eval.jsonl"
    rows = [{"model_name": ["x"], "shift_type": "normal", "status": "ok", "collision_count": 0},
            {"model_name": "m", "shift_type": "normal", "status": "ok", "collision_count": 1}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert reference_note(log, "normal", ("m",)) == "m 1/1 episodes with a collision"


def test_a_reference_log_that_is_not_utf8_is_a_reference_log_error(tmp_path):
    log = tmp_path / "eval.jsonl"
    log.write_bytes(b"\xff\xfe\x00bad\n")

    with pytest.raises(ReferenceLogError, match=r"eval\.jsonl"):
        reference_note(log, "normal", ("m",))


def test_save_trace_leaves_no_partial_file_when_writing_fails(tmp_path, monkeypatch):
    target = tmp_path / "ep.npz"
    monkeypatch.setattr(np, "savez_compressed", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError):
        save_trace(shielded_trace(), target)

    assert list(tmp_path.iterdir()) == []


def test_meta_children_tolerates_a_missing_payload(tmp_path):
    from orbit_guard.visualization.layout import meta_children

    save_trace(shielded_trace(), tmp_path / "a.npz")
    index = discover_episodes(tmp_path)
    meta = index.entries[0].meta

    children = meta_children(index, Selection(meta.scenario_id, meta.policy, meta.shield), None)

    assert len(children) == 3
