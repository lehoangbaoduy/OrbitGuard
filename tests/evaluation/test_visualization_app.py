"""The replay app wiring (PRD §41, FR16): discovers recorded episodes, serves payloads, no trainer needed."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dash")

from dash import dcc

from orbit_guard.visualization.app import create_app
from orbit_guard.visualization.callbacks import (
    PayloadProvider,
    apply_click,
    parse_selection,
)
from orbit_guard.visualization.episodes import discover_episodes, index_from_trace
from orbit_guard.visualization.layout import control_states
from orbit_guard.visualization.payload import decode_array
from orbit_guard.visualization.replay import EpisodeTrace, save_trace
from orbit_guard.visualization.replay_meta import build_replay_meta
from orbit_guard.visualization.selection import Selection, initial_selection
from tests.evaluation.replay_fixtures import collision_trace, shielded_trace


def _variant(trace: EpisodeTrace, *, policy: str, shield: bool, model: str | None = None) -> EpisodeTrace:
    base = trace.meta
    extra = {"weights_sha256": "a" * 64} if policy == "trained" else {"init_seed": 0}
    meta = build_replay_meta(
        condition=base["condition"], seed=base["seed"], policy=policy, shield=shield, model=model or base["model"],
        n_agents=base["n_agents"], delta_v_budget_mps=base["delta_v_budget_mps"],
        max_delta_v_per_action_mps=base["max_delta_v_per_action_mps"], **extra,
    )
    summary = {**trace.summary, "total_delta_v_mps": trace.summary["total_delta_v_mps"] * (2.0 if policy == "untrained" else 1.0)}
    return dataclasses.replace(trace, meta=meta, summary=summary)


@pytest.fixture(scope="module")
def episodes_dir(tmp_path_factory) -> Path:
    """normal:0 with untrained+trained on the shield (two episodes), plus a trained shield-off one; names scrambled."""
    root = tmp_path_factory.mktemp("episodes")
    base = shielded_trace()
    save_trace(_variant(base, policy="untrained", shield=True), root / "aaa.npz")
    save_trace(_variant(base, policy="trained", shield=True), root / "bbb.npz")
    save_trace(_variant(collision_trace(), policy="trained", shield=False), root / "ccc.npz")
    return root


@pytest.fixture(scope="module")
def index(episodes_dir):
    return discover_episodes(episodes_dir)


NORMAL = "normal:0"


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, (str, int, float)):
            yield from _walk(child)


def test_app_layout_contains_the_shell_and_no_graph(index):
    app = create_app(index)

    assert not [c for c in _walk(app.layout) if isinstance(c, dcc.Graph)]
    ids = {c.id for c in _walk(app.layout) if isinstance(getattr(c, "id", None), str)}
    assert {"og-payload", "og-selection", "og-strip", "og-scene"} <= ids


def test_initial_layout_embeds_the_payload_of_the_first_episode(index):
    app = create_app(index)

    store = next(c for c in _walk(app.layout) if getattr(c, "id", None) == "og-payload")
    assert store.data["meta"]["scenario_id"] == NORMAL and store.data["meta"]["policy"] == "untrained"
    assert store.data["meta"]["shield"] is True


def test_provider_builds_payloads_from_the_recorded_trace(index):
    provider = PayloadProvider(index)

    payload = provider(Selection(NORMAL, "trained", True))

    positions = decode_array(payload["arrays"]["positions_km"])
    assert positions.shape[1] == shielded_trace().num_agents
    assert payload["summary"]["delta_v_mps"] == pytest.approx(shielded_trace().summary["total_delta_v_mps"])


def test_provider_strip_pairs_the_policies_at_the_same_shield(index):
    strip = PayloadProvider(index)(Selection(NORMAL, "trained", True))["strip"]

    assert strip["delta_v_mps"]["before"] == pytest.approx(2.0 * strip["delta_v_mps"]["after"])
    assert strip["delta_v_mps"]["note"] is not None


def test_provider_leaves_the_counterpart_missing_when_it_was_not_recorded(index):
    strip = PayloadProvider(index)(Selection("simultaneous_conjunctions:3", "trained", False))["strip"]

    assert strip["delta_v_mps"]["before"] is None and strip["delta_v_mps"]["after"] is not None


def test_provider_returns_none_for_an_unrecorded_selection(index):
    assert PayloadProvider(index)(Selection(NORMAL, "trained", False)) is None
    assert PayloadProvider(index)(Selection(None, None, True)) is None


def test_provider_caches_by_selection(index):
    provider = PayloadProvider(index)

    assert provider(Selection(NORMAL, "trained", True)) is provider(Selection(NORMAL, "trained", True))


def test_reference_note_is_absent_without_a_log_and_read_from_the_log_when_given(index, tmp_path):
    assert PayloadProvider(index)(Selection(NORMAL, "trained", True))["reference_note"] is None
    log = tmp_path / "eval.jsonl"
    log.write_text(json.dumps({"model_name": "no_maneuver", "shift_type": "normal", "status": "ok", "collision_count": 1, "n_satellites": 2}) + "\n")

    payload = PayloadProvider(index, reference_log=log)(Selection(NORMAL, "trained", True))

    assert payload["reference_note"] == "no_maneuver 1/1 episodes with a collision"


def test_clicking_a_disabled_or_missing_combination_is_ignored(index):
    current = Selection(NORMAL, "trained", True)

    assert apply_click(index, current, "og-shield", "off") == current  # no shield-off trace for normal:0
    assert apply_click(index, current, "og-policy", "untrained") == Selection(NORMAL, "untrained", True)
    assert apply_click(index, current, "og-scenario", "simultaneous_conjunctions:3").scenario_id == "simultaneous_conjunctions:3"


@pytest.mark.parametrize(
    "state",
    [None, "x", {}, {"scenario_id": "../../etc/passwd", "policy": "trained", "shield": True}, {"scenario_id": NORMAL, "policy": "trained", "shield": True, "extra": 1}],
)
def test_client_supplied_selection_state_is_validated_not_trusted(index, state):
    selection = parse_selection(index, state)

    assert selection.scenario_id is None or index.find(selection.scenario_id, selection.policy, selection.shield) is not None


def test_a_valid_client_selection_round_trips(index):
    assert parse_selection(index, {"scenario_id": NORMAL, "policy": "trained", "shield": True}) == Selection(NORMAL, "trained", True)


def test_control_updates_follow_the_selection(index):
    updates = control_states(index, Selection(NORMAL, "trained", True))

    assert updates["og-shield"]["off"]["aria-disabled"] is True and updates["og-shield"]["on"]["aria-pressed"] is True
    assert "AFTER TRAINING" in str(updates["og-policy"]["trained"]["children"])


def _wildcard_ids(index, kind):
    return {"og-scenario": [s.scenario_id for s in index.scenarios()], "og-policy": list(index.policies()), "og-shield": ["on", "off"]}[kind]


def _expand_outputs(index, key):
    """The outputs the browser would send: wildcard (ALL) outputs expanded to the matching components."""
    outputs = []
    for part in key.strip(".").split("..."):
        ident, prop = part.rsplit(".", 1)
        try:
            parsed = json.loads(ident)
        except json.JSONDecodeError:
            outputs.append({"id": ident, "property": prop})
            continue
        outputs.append([{"id": {"type": parsed["type"], "id": i}, "property": prop} for i in _wildcard_ids(index, parsed["type"])])
    return outputs


def test_server_callback_round_trip_updates_selection_payload_and_controls(index):
    app = create_app(index)
    key = next(k for k in app.callback_map if "og-selection.data" in k)
    inputs = []
    for item in app.callback_map[key]["inputs"]:
        kind = json.loads(item["id"])["type"]
        inputs.append([{"id": {"type": kind, "id": i}, "property": "n_clicks", "value": 1 if i == "untrained" else 0} for i in _wildcard_ids(index, kind)])
    body = {
        "output": key,
        "outputs": _expand_outputs(index, key),
        "inputs": inputs,
        "state": [{"id": "og-selection", "property": "data", "value": {"scenario_id": NORMAL, "policy": "trained", "shield": True}}],
        "changedPropIds": ['{"id":"untrained","type":"og-policy"}.n_clicks'],
    }

    response = app.server.test_client().post("/_dash-update-component", json=body)

    assert response.status_code == 200, response.get_data(as_text=True)
    out = response.get_json()["response"]
    assert out["og-selection"]["data"]["policy"] == "untrained"
    assert out["og-payload"]["data"]["meta"]["policy"] == "untrained"
    assert out['{"id":"untrained","type":"og-policy"}']["aria-pressed"] is True
    assert out['{"id":"trained","type":"og-policy"}']["aria-pressed"] is False
    assert out['{"id":"off","type":"og-shield"}']["title"] == "no trace recorded"


def _post(app, index, *, state, changed, scenario_ids=None):
    key = next(k for k in app.callback_map if "og-selection.data" in k)
    inputs = []
    for item in app.callback_map[key]["inputs"]:
        kind = json.loads(item["id"])["type"]
        inputs.append([{"id": {"type": kind, "id": i}, "property": "n_clicks", "value": 0} for i in _wildcard_ids(index, kind)])
    body = {"output": key, "outputs": _expand_outputs(index, key), "inputs": inputs,
            "state": [{"id": "og-selection", "property": "data", "value": state}], "changedPropIds": [changed]}
    return app.server.test_client().post("/_dash-update-component", json=body)


def test_clicking_the_active_control_sends_nothing_back(index):  # EC-13: the stage must not see a "new episode"
    app = create_app(index)

    response = _post(app, index, state={"scenario_id": NORMAL, "policy": "trained", "shield": True}, changed='{"id":"trained","type":"og-policy"}.n_clicks')

    assert response.status_code == 204  # PreventUpdate: no outputs at all, so the stage never sees a "new episode"


def test_clicking_a_control_with_no_trace_changes_nothing(index):
    app = create_app(index)

    response = _post(app, index, state={"scenario_id": NORMAL, "policy": "trained", "shield": True}, changed='{"id":"off","type":"og-shield"}.n_clicks')

    assert response.status_code == 204


@pytest.mark.parametrize("changed", ['{"foo":1}.n_clicks', '{"id":"bogus","type":"og-shield"}.n_clicks', "not-json.n_clicks"])
def test_malformed_triggers_do_not_crash_the_callback(index, changed):
    response = _post(create_app(index), index, state={"scenario_id": NORMAL, "policy": "trained", "shield": True}, changed=changed)

    assert response.status_code in (200, 204)


def test_a_stale_output_id_from_an_old_page_does_not_crash_the_callback(index):
    app = create_app(index)
    key = next(k for k in app.callback_map if "og-selection.data" in k)
    outputs = _expand_outputs(index, key)
    outputs[5].append({"id": {"type": "og-scenario", "id": "old:1"}, "property": "className"})  # a scenario this server no longer has
    inputs = []
    for item in app.callback_map[key]["inputs"]:
        kind = json.loads(item["id"])["type"]
        inputs.append([{"id": {"type": kind, "id": i}, "property": "n_clicks", "value": 0} for i in _wildcard_ids(index, kind)])
    body = {"output": key, "outputs": outputs, "inputs": inputs,
            "state": [{"id": "og-selection", "property": "data", "value": {"scenario_id": NORMAL, "policy": "trained", "shield": True}}],
            "changedPropIds": ['{"id":"untrained","type":"og-policy"}.n_clicks']}

    response = app.server.test_client().post("/_dash-update-component", json=body)

    assert response.status_code == 200, response.get_data(as_text=True)


def test_the_page_declares_its_language(index):
    assert 'lang="en"' in create_app(index).index_string


def test_single_trace_mode_builds_an_app(tmp_path):
    path = tmp_path / "one.npz"
    save_trace(shielded_trace(), path)

    app = create_app(index_from_trace(path))

    assert initial_selection(index_from_trace(path)).scenario_id == "normal:0"
    assert app.layout is not None


def test_app_with_no_episodes_shows_the_empty_state(tmp_path):
    app = create_app(discover_episodes(tmp_path))

    assert "No episode loaded" in str([c.children for c in _walk(app.layout) if isinstance(getattr(c, "children", None), str)])


def test_replay_app_runs_standalone_without_ray_or_torch(episodes_dir):
    code = (
        "import sys;"
        "from orbit_guard.visualization.episodes import discover_episodes;"
        "from orbit_guard.visualization.app import create_app;"
        f"app = create_app(discover_episodes(r'{episodes_dir}'));"
        "assert 'ray' not in sys.modules and 'torch' not in sys.modules and 'gymnasium' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
