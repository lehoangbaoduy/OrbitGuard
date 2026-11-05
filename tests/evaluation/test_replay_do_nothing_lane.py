"""The do-nothing lane of the replay app (2026-11-04): the `no_maneuver` baseline shown beside BEFORE/AFTER training.

The baseline is the evaluated `no_maneuver` model, so it is recorded with the shield off only. The lane must stay
reachable while the shield is ON (a click switches to the shield setting it has), it gets its own name, and the
summary strip gains a third bar row so its totals sit on the same scale as BEFORE and AFTER, whatever the shield.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from orbit_guard.evaluation.replay_recording import (
    BASELINE_POLICIES,
    SCENARIOS,
    EpisodePlan,
    build_plan,
    record_plan_item,
)
from orbit_guard.visualization.episodes import (
    EpisodeEntry,
    EpisodeIndex,
    discover_episodes,
    read_episode_summary,
)
from orbit_guard.visualization.payload import build_strip
from orbit_guard.visualization.replay import load_trace
from orbit_guard.visualization.replay_meta import (
    build_replay_meta,
    validate_replay_meta,
)
from orbit_guard.visualization.selection import Selection, resolve_selection
from orbit_guard.visualization.summary import SummaryTotals
from tests.evaluation.replay_fixtures import ENV_KWARGS


def _entry(condition: str, seed: int, policy: str, shield: bool) -> EpisodeEntry:
    extra = {"weights_sha256": "a" * 64} if policy == "trained" else {"init_seed": 0} if policy == "untrained" else {}
    meta = build_replay_meta(condition=condition, seed=seed, policy=policy, shield=shield, model="m", n_agents=15,
                             delta_v_budget_mps=2.0, max_delta_v_per_action_mps=0.1, **extra)
    return EpisodeEntry(Path(f"{condition}-{seed}-{policy}-{shield}.npz"), validate_replay_meta(meta))


def _index(*combos) -> EpisodeIndex:
    return EpisodeIndex(tuple(_entry(*combo) for combo in combos))


def _with_nothing(condition="normal", seed=5):
    trained = [(condition, seed, policy, shield) for policy in ("untrained", "trained") for shield in (True, False)]
    return _index(*trained, (condition, seed, "no_maneuver", False))


def _totals(dv, man, shield, coll, dev):
    return SummaryTotals(delta_v_mps=dv, maneuvers=man, shield=shield, collisions=coll, mean_deviation=dev)


# --- selection --------------------------------------------------------------------------------------------------


def test_choosing_do_nothing_with_the_shield_on_switches_to_its_shield_off_trace():
    current = Selection("normal:5", "trained", True)

    assert resolve_selection(_with_nothing(), current, policy="no_maneuver") == Selection("normal:5", "no_maneuver", False)


def test_a_policy_with_no_trace_at_either_shield_setting_is_still_ignored():
    index = _index(("normal", 5, "untrained", True))
    current = Selection("normal:5", "untrained", True)

    assert resolve_selection(index, current, policy="trained") == current


def test_a_policy_recorded_at_the_current_shield_keeps_the_shield():
    current = Selection("normal:5", "untrained", False)

    assert resolve_selection(_with_nothing(), current, policy="trained") == Selection("normal:5", "trained", False)


# --- layout -----------------------------------------------------------------------------------------------------


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None and not isinstance(children, (str, int, float)):
        yield from _walk(children)


def _text(node) -> str:
    return " ".join(str(c.children) for c in _walk(node) if isinstance(getattr(c, "children", None), (str, int, float)))


def _policy_tabs(index, selection):
    pytest.importorskip("dash")
    from orbit_guard.visualization.layout import build_layout

    layout = build_layout(index, selection, None)
    return {c.id["id"]: c for c in _walk(layout) if isinstance(getattr(c, "id", None), dict) and c.id.get("type") == "og-policy"}


def test_the_do_nothing_tab_comes_after_before_and_after_training_and_is_named_for_what_it_does():
    tabs = _policy_tabs(_with_nothing(), Selection("normal:5", "untrained", False))

    assert list(tabs) == ["untrained", "trained", "no_maneuver"]
    assert "DO NOTHING" in _text(tabs["no_maneuver"]) and "no_maneuver policy · shield OFF" in _text(tabs["no_maneuver"])


def test_the_do_nothing_tab_stays_enabled_with_the_shield_on_and_says_it_opens_shield_off():
    tab = _policy_tabs(_with_nothing(), Selection("normal:5", "trained", True))["no_maneuver"]

    assert tab.__dict__.get("aria-disabled") is False
    assert "no trace" not in _text(tab)
    assert "shield OFF" in _text(tab)  # the trace a click opens, not the shield currently selected


def test_without_a_do_nothing_trace_there_is_no_do_nothing_tab():
    index = _index(*[("normal", 5, p, s) for p in ("untrained", "trained") for s in (True, False)])

    assert list(_policy_tabs(index, Selection("normal:5", "untrained", True))) == ["untrained", "trained"]


# --- summary strip ----------------------------------------------------------------------------------------------


def test_the_strip_carries_a_do_nothing_value_only_when_one_is_given():
    before, after, nothing = _totals(30.0, 400, 3000, 0, 0.7), _totals(3.0, 40, 12, 0, 0.2), _totals(0.0, 0, None, 2, 0.9)

    with_lane = build_strip("trained", {"untrained": before, "trained": after, "no_maneuver": nothing})
    without = build_strip("trained", {"untrained": before, "trained": after})

    assert with_lane["collisions"]["nothing"] == 2 and with_lane["delta_v_mps"]["nothing"] == 0.0
    assert with_lane["shield"]["nothing"] is None
    assert "nothing" not in without["collisions"]


def _strip_for(active: str, totals: dict):
    pytest.importorskip("dash")
    from orbit_guard.visualization.layout_strip import summary_strip
    from orbit_guard.visualization.payload import build_payload
    from orbit_guard.visualization.source import ReplaySource
    from tests.evaluation.replay_fixtures import collision_trace

    payload = build_payload(ReplaySource(collision_trace()), strip=build_strip(active, totals))
    payload = {**payload, "meta": {**payload["meta"], "policy": active}}
    return summary_strip(payload)


def test_the_do_nothing_row_shares_the_bar_scale_and_is_the_active_row_on_its_own_tab():
    totals = {"untrained": _totals(30.0, 400, None, 1, 0.7), "trained": _totals(3.0, 40, None, 0, 0.2),
              "no_maneuver": _totals(0.0, 0, None, 2, 0.9)}

    strip = _strip_for("no_maneuver", totals)
    bars = {c.id: c for c in _walk(strip) if isinstance(getattr(c, "id", None), str) and c.id.startswith("og-strip-bar-collisions")}
    rows = [c for c in _walk(strip) if "og-bar-nothing" in str(getattr(c, "className", ""))]

    assert bars["og-strip-bar-collisions-nothing"].style["transform"] == "scaleX(1)"
    assert bars["og-strip-bar-collisions-before"].style["transform"] == "scaleX(0.5)"
    assert rows and all("is-active" in r.className for r in rows)
    assert "NOTHING" in _text(strip)


def test_the_do_nothing_shield_cell_says_shield_off_not_no_trace():
    totals = {"untrained": _totals(30.0, 400, 3000, 1, 0.7), "trained": _totals(3.0, 40, 12, 0, 0.2),
              "no_maneuver": _totals(0.0, 0, None, 2, 0.9)}

    cell = next(c for c in _walk(_strip_for("trained", totals)) if getattr(c, "id", None) == "og-strip-cell-shield")
    nothing_row = next(c for c in _walk(cell) if "og-bar-nothing" in str(getattr(c, "className", "")))

    assert "shield off" in _text(nothing_row) and "no trace" not in _text(nothing_row)


def test_the_payload_provider_fills_the_do_nothing_row_whichever_shield_is_selected(tmp_path):
    pytest.importorskip("dash")
    from orbit_guard.visualization.callbacks import PayloadProvider
    from orbit_guard.visualization.replay import save_trace
    from tests.evaluation.replay_fixtures import collision_trace, shielded_trace

    nothing = collision_trace()
    scenario = nothing.meta["scenario_id"]
    condition, seed = nothing.meta["condition"], nothing.meta["seed"]
    shielded = dataclasses.replace(shielded_trace(), meta=build_replay_meta(
        condition=condition, seed=seed, policy="trained", shield=True, model="adaptive_mappo", n_agents=2,
        delta_v_budget_mps=nothing.meta["delta_v_budget_mps"], max_delta_v_per_action_mps=0.1, weights_sha256="0" * 64))
    save_trace(nothing, tmp_path / "a.npz")
    save_trace(shielded, tmp_path / "b.npz")
    provider = PayloadProvider(discover_episodes(tmp_path))

    on = provider(Selection(scenario, "trained", True))
    off = provider(Selection(scenario, "no_maneuver", False))

    expected = read_episode_summary(tmp_path / "a.npz").collisions
    assert expected > 0  # the fixture is the colliding do-nothing episode
    assert on["strip"]["collisions"]["nothing"] == off["strip"]["collisions"]["nothing"] == expected


# --- recording --------------------------------------------------------------------------------------------------


def test_the_default_plan_is_unchanged_and_the_baseline_is_opt_in(tmp_path):
    assert BASELINE_POLICIES == ("no_maneuver",)
    assert len(build_plan(tmp_path)) == 12
    plan = build_plan(tmp_path, baselines=("no_maneuver",))
    baseline = [item for item in plan if item.policy == "no_maneuver"]

    assert len(plan) == 15
    assert [(item.condition, item.seed) for item in baseline] == list(SCENARIOS)
    assert all(not item.shield and item.model == "no_maneuver" and item.weights_path is None and item.training_seed is None
               for item in baseline)


def test_an_unknown_baseline_is_refused(tmp_path):
    with pytest.raises(ValueError, match="rule_based"):
        build_plan(tmp_path, baselines=("rule_based",))


def _record(tmp_path, item):
    return record_plan_item(item, num_agents=2, out_dir=tmp_path / "out", init_seed=0, env_kwargs=ENV_KWARGS)


def test_a_recorded_do_nothing_episode_never_burns_and_names_the_evaluated_baseline(tmp_path):
    path = _record(tmp_path, EpisodePlan("normal", 3, "no_maneuver", False, "no_maneuver", None))

    (entry,) = discover_episodes(path.parent).entries
    trace = load_trace(path)
    assert entry.meta.policy == "no_maneuver" and entry.meta.model == "no_maneuver" and entry.meta.shield is False
    assert entry.meta.weights_sha256 is None and entry.meta.init_seed is None
    assert not np.any(trace.applied_dv_mps) and not trace.intervened.any()
    assert not (path.parent / "weights").exists()  # no actor, so nothing to persist beside the traces


def test_the_do_nothing_episode_starts_from_the_same_state_as_the_untrained_one(tmp_path):
    nothing = load_trace(_record(tmp_path, EpisodePlan("normal", 3, "no_maneuver", False, "no_maneuver", None)))
    untrained = load_trace(_record(tmp_path, EpisodePlan("normal", 3, "untrained", False, "untrained_policy", None)))

    np.testing.assert_array_equal(nothing.positions_m[0], untrained.positions_m[0])


def test_a_baseline_with_the_shield_on_is_refused(tmp_path):
    with pytest.raises(ValueError, match="shield"):
        _record(tmp_path, EpisodePlan("normal", 3, "no_maneuver", True, "no_maneuver", None))


def test_the_recording_script_adds_the_do_nothing_lane_only_when_asked(tmp_path, monkeypatch):
    from tests.evaluation.test_replay_recording_overlay import _all_weights, _script

    script, plans = _script(), []
    _all_weights(tmp_path / "runs", trained_with=None)
    monkeypatch.setattr(script, "record_all", lambda plan, out, **kwargs: plans.append(plan) or [])
    base = ["--runs-dir", str(tmp_path / "runs"), "--unfrozen"]

    script.main([*base, "--out", str(tmp_path / "plain")])
    script.main([*base, "--out", str(tmp_path / "lane"), "--baseline", "no_maneuver"])

    assert [len(plan) for plan in plans] == [12, 15]
    assert [item.policy for item in plans[1]].count("no_maneuver") == 3
    assert "no_maneuver" not in {item.policy for item in plans[0]}


def test_a_scenario_without_a_do_nothing_trace_says_no_trace_in_every_nothing_cell():
    totals = {"untrained": _totals(30.0, 400, 3000, 1, 0.7), "trained": _totals(3.0, 40, 12, 0, 0.2), "no_maneuver": None}

    strip = _strip_for("trained", totals)
    rows = [c for c in _walk(strip) if "og-bar-nothing" in str(getattr(c, "className", ""))]

    assert len(rows) == 5 and all("no trace recorded" in _text(row) for row in rows)


def test_a_before_or_after_tab_recorded_only_at_the_other_shield_opens_that_setting():
    index = _index(("normal", 5, "untrained", True), ("normal", 5, "trained", True), ("normal", 5, "trained", False))
    current = Selection("normal:5", "trained", False)

    tab = _policy_tabs(index, current)["untrained"]

    assert resolve_selection(index, current, policy="untrained") == Selection("normal:5", "untrained", True)
    assert tab.__dict__.get("aria-disabled") is False and "untrained policy · shield ON" in _text(tab)


def test_a_baseline_whose_registered_model_runs_the_shield_is_refused(tmp_path):
    import yaml

    from orbit_guard.config import DEFAULT_CONFIG_PATH

    config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    config["models"]["no_maneuver"]["shield"] = True
    path = tmp_path / "shielded_baseline.yaml"
    path.write_text(yaml.safe_dump(config))
    item = EpisodePlan("normal", 3, "no_maneuver", False, "no_maneuver", None)

    with pytest.raises(ValueError, match="shield"):
        record_plan_item(item, num_agents=2, out_dir=tmp_path / "out", env_kwargs=ENV_KWARGS, config_path=path)
