"""The replay app shell (PR2): header, tabs, shield control, summary strip, empty states (EC-02, EC-03, EC-06,
EC-26, EC-33). Built from real recorded fixtures; the layout tree is inspected without a browser."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("dash")

from dash import dcc

from orbit_guard.visualization.episodes import EpisodeEntry, EpisodeIndex
from orbit_guard.visualization.layout import build_layout, summary_strip
from orbit_guard.visualization.payload import build_payload, build_strip
from orbit_guard.visualization.replay_meta import (
    build_replay_meta,
    validate_replay_meta,
)
from orbit_guard.visualization.selection import Selection
from orbit_guard.visualization.source import ReplaySource
from orbit_guard.visualization.summary import SummaryTotals
from tests.evaluation.replay_fixtures import depletion_trace, shielded_trace


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


def _entry(condition, seed, policy, shield):
    extra = {"weights_sha256": "a" * 64} if policy == "trained" else {"init_seed": 0}
    meta = build_replay_meta(condition=condition, seed=seed, policy=policy, shield=shield, model="m", n_agents=15,
                             delta_v_budget_mps=2.0, max_delta_v_per_action_mps=0.1, **extra)
    return EpisodeEntry(Path("x.npz"), validate_replay_meta(meta))


def _index(combos):
    return EpisodeIndex(tuple(_entry(*c) for c in combos))


FULL = [(c, s, p, sh) for c, s in (("normal", 5), ("capability_reduction", 10)) for p in ("untrained", "trained") for sh in (True, False)]
SELECTION = Selection("normal:5", "untrained", True)


def _layout(index=None, selection=SELECTION, payload=None, **kw):
    return build_layout(index or _index(FULL), selection, payload, **kw)


def _buttons(layout, kind):
    return [c for c in _walk(layout) if isinstance(getattr(c, "id", None), dict) and c.id.get("type") == kind]


def test_layout_has_no_dcc_graph():  # EC-33: dash's bundled plotly.js (4.1.1) would conflict with the vendored 2.35.2
    assert not [c for c in _walk(_layout()) if isinstance(c, dcc.Graph)]


def test_one_scenario_button_per_recorded_scenario_with_label_and_seed():
    buttons = _buttons(_layout(), "og-scenario")
    assert [b.id["id"] for b in buttons] == ["normal:5", "capability_reduction:10"]
    assert "Normal" in _text(buttons[0]) and "seed 5" in _text(buttons[0])  # the seed comes from trace.meta
    assert buttons[0].__dict__.get("aria-pressed") in (True, "true")


def test_policy_tabs_are_named_and_described_from_the_selection():
    tabs = _buttons(_layout(), "og-policy")
    assert [b.id["id"] for b in tabs] == ["untrained", "trained"]
    assert "BEFORE TRAINING" in _text(tabs[0]) and "untrained policy · shield ON" in _text(tabs[0])
    assert "AFTER TRAINING" in _text(tabs[1]) and "trained policy · shield ON" in _text(tabs[1])


def test_shield_control_has_on_and_off_with_the_selected_one_pressed():
    buttons = _buttons(_layout(selection=Selection("normal:5", "trained", False)), "og-shield")
    assert [b.id["id"] for b in buttons] == ["on", "off"]
    assert [b.__dict__.get("aria-pressed") for b in buttons] == [False, True]


def _aria_disabled(button):
    return button.__dict__.get("aria-disabled")


def test_missing_combinations_are_marked_disabled_but_stay_focusable_with_a_visible_reason():  # EC-03
    index = _index([("normal", 5, "untrained", True), ("normal", 5, "trained", True)])  # no shield-off episodes
    layout = _layout(index=index)
    off = next(b for b in _buttons(layout, "og-shield") if b.id["id"] == "off")
    on = next(b for b in _buttons(layout, "og-shield") if b.id["id"] == "on")

    assert _aria_disabled(off) is True and off.title == "no trace recorded" and "no trace" in _text(off)
    assert not getattr(off, "disabled", False)  # a natively disabled button leaves the tab order and hides its reason from keyboards and touch
    assert _aria_disabled(on) is False and "no trace" not in _text(on)


def test_a_policy_tab_without_a_trace_says_so_in_visible_text():
    index = _index([("normal", 5, "untrained", True)])  # no AFTER trace at all

    tabs = _buttons(_layout(index=index), "og-policy")

    assert [t.id["id"] for t in tabs] == ["untrained", "trained"]  # BEFORE and AFTER are always offered
    assert _aria_disabled(tabs[1]) is True and "no trace recorded" in _text(tabs[1])
    assert _aria_disabled(tabs[0]) is False and "no trace recorded" not in _text(tabs[0])


def test_shield_off_indicator_appears_only_when_the_shield_is_off():
    assert "SHIELD OFF — nothing overrides the policy" not in _text(_layout())
    assert "SHIELD OFF — nothing overrides the policy" in _text(_layout(selection=Selection("normal:5", "trained", False)))


def test_shield_off_bar_explains_what_still_shows():
    text = _text(_layout(selection=Selection("normal:5", "trained", False)))

    assert "applied as-is" in text and "✕ collision" in text


def test_empty_index_shows_the_no_episode_state_and_disables_controls():  # EC-02
    layout = _layout(index=_index([]), selection=Selection(None, None, True))
    assert "No episode loaded" in _text(layout)
    controls = _buttons(layout, "og-shield") + _buttons(layout, "og-policy") + _buttons(layout, "og-scenario")
    assert len(_buttons(layout, "og-policy")) == 2 and len(_buttons(layout, "og-shield")) == 2
    assert all(_aria_disabled(c) is True for c in controls)


def test_a_loading_state_exists_and_panes_have_stable_regions():  # EC-01
    layout = _layout()
    assert "LOADING TRACE" in _text(layout)
    ids = {c.id for c in _walk(layout) if isinstance(getattr(c, "id", None), str)}
    for region in ("og-payload", "og-stage", "og-scene", "og-strip", "og-playback", "og-timeline", "og-rail-right"):
        assert region in ids, region


def test_layout_uses_classes_not_inline_styles():
    offenders = []
    for component in _walk(_layout(payload=_payload(shielded_trace()))):
        style = getattr(component, "style", None)
        if style and any(key not in {"transform"} and not str(key).startswith("--") for key in style):
            offenders.append((type(component).__name__, style))
    assert offenders == []


# ----------------------------------------------------------------------- summary strip
def _payload(trace, **kwargs):
    return build_payload(ReplaySource(trace), **kwargs)


def _totals(dv, man, shield, coll, dev):
    return SummaryTotals(delta_v_mps=dv, maneuvers=man, shield=shield, collisions=coll, mean_deviation=dev)


def _strip(payload):
    return summary_strip(payload)


def test_strip_has_five_cells_in_the_designed_order():
    text = _text(_strip(_payload(shielded_trace())))
    positions = [text.index(label) for label in ("TOTAL Δv", "MANEUVERS", "SHIELD", "COLLISIONS", "MISSION DEVIATION")]
    assert positions == sorted(positions)


def test_strip_totals_come_from_the_recorded_summary_and_shield_off_is_a_dash():
    payload = _payload(depletion_trace())  # shield off
    cells = {c.id: c for c in _walk(_strip(payload)) if isinstance(getattr(c, "id", None), str) and c.id.startswith("og-strip-val-")}
    assert cells["og-strip-val-shield"].children == "—"
    assert cells["og-strip-val-delta_v_mps"].children == f"{payload['summary']['delta_v_mps']:.2f}"
    assert cells["og-strip-val-collisions"].children == str(payload["summary"]["collisions"])


def test_strip_bars_share_one_scale_and_missing_counterparts_have_no_bar():
    payload = _payload(shielded_trace(), strip=build_strip(
        "untrained", {"untrained": _totals(30.0, 400, 3000, 0, 0.7), "trained": _totals(3.0, 40, 12, 0, 0.2)}))
    bars = {c.id: c for c in _walk(_strip(payload)) if isinstance(getattr(c, "id", None), str) and c.id.startswith("og-strip-bar-delta_v_mps")}
    assert bars["og-strip-bar-delta_v_mps-before"].style["transform"] == "scaleX(1)"
    assert bars["og-strip-bar-delta_v_mps-after"].style["transform"] == "scaleX(0.1)"
    lone = _payload(shielded_trace(), strip=build_strip("trained", {"untrained": None, "trained": _totals(3.0, 40, 12, 0, 0.2)}))
    text = _text(_strip(lone))
    assert "no trace recorded" in text and "less than" not in text


def test_strip_notes_are_data_generated():
    payload = _payload(shielded_trace(), strip=build_strip(
        "untrained", {"untrained": _totals(30.0, 400, 3000, 0, 0.7), "trained": _totals(3.0, 40, 12, 0, 0.2)}))
    assert "10× after" in _text(_strip(payload))


def test_reference_note_appears_in_the_collisions_cell_only_when_the_payload_has_one():
    note = "adaptive_mappo 1/10 · no_maneuver 5/10 episodes with a collision"
    shown = _text(_strip(_payload(shielded_trace(), reference_note=note)))
    hidden = _payload(shielded_trace())
    cell = next(c for c in _walk(_strip(_payload(shielded_trace(), reference_note=note))) if getattr(c, "id", None) == "og-strip-cell-collisions")

    assert note in shown
    assert note in _text(cell) and [c for c in _walk(cell) if getattr(c, "id", None) == "og-strip-ref"]
    assert "episodes with a collision" not in _text(_strip(hidden))
    assert not [c for c in _walk(_strip(hidden)) if getattr(c, "id", None) == "og-strip-ref"]


def test_strip_is_empty_state_safe_without_a_payload():
    assert _text(_strip(None)).count("—") >= 5


def test_building_the_layout_never_mutates_the_payload():
    payload = _payload(shielded_trace())
    before = copy.deepcopy(payload)

    _layout(payload=payload)
    summary_strip(payload)

    assert payload == before


def test_status_regions_and_the_scene_have_accessible_roles():
    layout = _layout(selection=Selection("normal:5", "trained", False))
    by_id = {c.id: c for c in _walk(layout) if isinstance(getattr(c, "id", None), str)}

    assert by_id["og-shield-off-slot"].__dict__.get("role") == "status"  # a persistent live region, so text changes are announced
    scene = by_id["og-scene"].__dict__
    assert scene.get("role") == "group" and scene.get("aria-label") and scene.get("tabIndex") == 0  # keyboard-operable, named
    assert by_id["og-live"].__dict__.get("aria-live") == "polite" and by_id["og-alert"].__dict__.get("role") == "alert"
    assert by_id["og-banner"].__dict__.get("aria-hidden") == "true"  # the tracked panel carries the same information


def test_transport_and_speed_controls_have_names_that_contain_their_visible_text_and_shortcut_hints():
    by_id = {c.id: c for c in _walk(_layout()) if isinstance(getattr(c, "id", None), str)}

    assert by_id["og-next-event"].__dict__.get("aria-keyshortcuts") == "]"
    assert by_id["og-prev-event"].__dict__.get("aria-keyshortcuts") == "["
    speeds = [c for c in _walk(_layout()) if getattr(c, "className", "") and "og-speed-btn" in c.className]
    assert speeds and all(c.children in c.__dict__["aria-label"] for c in speeds)
    view = by_id["og-view-reset"]
    assert "aria-label" not in view.__dict__  # visible word is the name


def test_the_loading_detail_is_left_to_the_stage_so_it_can_never_go_stale():
    detail = next(c for c in _walk(_layout(payload=_payload(shielded_trace()))) if getattr(c, "id", None) == "og-loading-detail")

    assert not detail.children


def test_phone_strip_labels_stay_readable_to_assistive_tech():
    cell = _strip(_payload(shielded_trace()))
    short = [c for c in _walk(cell) if getattr(c, "className", "") == "og-cell-label-short"]

    assert short and all(c.__dict__.get("aria-hidden") == "true" for c in short)


def test_legend_lists_every_status_with_glyph_and_label_and_states_the_visual_scale():
    legend = next(c for c in _walk(_layout()) if getattr(c, "id", None) == "og-legend")
    text = _text(legend)

    for glyph, label in (("○", "INSIDE d_safe"), ("◆", "HIGH RISK"), ("✕", "COLLIDED"), ("■", "MANEUVER"), ("◇", "SHIELD INTERVENTION")):
        assert glyph in text and label in text, (glyph, label)
    assert "not drawn to scale" in text and "not data" in text


def test_legend_chips_are_tagged_with_their_vocabulary_name_so_the_stage_can_dim_the_shield_entry():
    from orbit_guard.visualization.vocabulary import EVENT_VOCAB

    legend = next(c for c in _walk(_layout()) if getattr(c, "id", None) == "og-legend")
    tags = {c.__dict__.get("data-legend") for c in _walk(legend) if getattr(c, "className", "") == "og-legend-chip"}

    assert EVENT_VOCAB[2].name in tags and EVENT_VOCAB[1].name in tags
    assert None not in tags


def test_the_stage_has_a_starfield_for_the_camera_parallax():
    ids = {c.id for c in _walk(_layout()) if isinstance(getattr(c, "id", None), str)}

    assert {"og-stars-far", "og-stars-near"} <= ids


# ----------------------------------------------------------------------- playback bar (PR4)
def _by_id(layout):
    return {c.id: c for c in _walk(layout) if isinstance(getattr(c, "id", None), str)}


def test_playback_bar_has_the_five_transport_buttons_with_accessible_names():
    ids = _by_id(_layout(payload=_payload(shielded_trace())))

    names = {key: ids[key].__dict__.get("aria-label") for key in ("og-prev-event", "og-step-back", "og-play", "og-step-fwd", "og-next-event")}
    assert names == {"og-prev-event": "Previous event", "og-step-back": "Step back", "og-play": "Play", "og-step-fwd": "Step forward", "og-next-event": "Next event"}
    assert ids["og-play"].__dict__.get("aria-pressed") is False


def test_the_scrubber_is_a_keyboard_slider_over_the_recorded_steps():
    payload = _payload(shielded_trace())
    scrub = _by_id(_layout(payload=payload))["og-scrub"]

    assert scrub.__dict__.get("role") == "slider" and scrub.__dict__.get("tabIndex") == 0
    assert scrub.__dict__.get("aria-valuemin") == 0 and scrub.__dict__.get("aria-valuemax") == payload["dims"]["steps"]


def test_speed_buttons_offer_1_2_4_8_with_one_pressed_and_loop_is_a_toggle():
    layout = _layout(payload=_payload(shielded_trace()))
    speeds = _buttons(layout, "og-speed")

    assert [b.id["id"] for b in speeds] == ["1", "2", "4", "8"]
    assert [b.__dict__.get("aria-pressed") for b in speeds] == [True, False, False, False]
    assert _by_id(layout)["og-loop"].__dict__.get("aria-pressed") is False


def test_the_scrubber_has_a_canvas_for_the_density_bars_and_a_hover_preview():
    ids = _by_id(_layout(payload=_payload(shielded_trace())))

    assert {"og-scrub-density", "og-scrub-fill", "og-scrub-thumb", "og-scrub-hover"} <= set(ids)
