"""Static shell of the replay app (redesign PR2): header, tab bar, shield control, summary strip, stage regions.

Server-rendered from the episode index, the selection and (optionally) the payload of the selected episode. There is
deliberately no `dcc.Graph`: every chart is drawn by the vendored plotly.js in the JS stage (decision D1), and the
regions below (`og-scene`, `og-playback`, `og-timeline`, `og-rail-right`) are the mount points those modules fill.
Missing combinations are disabled with the reason "no trace recorded" (real-data rule 4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from dash import dcc, html

from orbit_guard.visualization.episodes import EpisodeIndex
from orbit_guard.visualization.layout_playback import playback_bar
from orbit_guard.visualization.layout_strip import DASH, NO_TRACE, summary_strip
from orbit_guard.visualization.selection import Selection
from orbit_guard.visualization.vocabulary import EVENT_VOCAB, SAT_VOCAB

__all__ = ["build_layout", "control_states", "meta_children", "policy_label", "shield_off_bar", "summary_strip"]

SHIELD_OFF_TEXT = "SHIELD OFF — nothing overrides the policy"
SHIELD_OFF_NOTE = "Every proposed action is applied as-is. Remaining events: ■ maneuver · ◆ high risk · ○ inside d_safe · ✕ collision."
_POLICY_NAMES = {"untrained": "BEFORE TRAINING", "trained": "AFTER TRAINING"}
_SHIELD_GLYPHS = {"on": "◇", "off": "⊘"}
_PANES = (("og-fuel", "Fleet fuel"), ("og-risk", "Conjunction & risk"))


def policy_label(policy: str) -> str:
    return _POLICY_NAMES.get(policy, policy.replace("_", " ").upper())


NO_TRACE_SHORT = "no trace"


def _state(*, pressed: bool, disabled: bool, class_name: str) -> dict[str, Any]:
    """The props of a segmented-control button that change with the selection (also the callback outputs).

    An unavailable option is `aria-disabled`, not natively disabled: it stays in the tab order and keeps its tooltip,
    so keyboard and screen-reader users can reach it and learn why it does nothing.
    """
    return {
        "className": class_name + (" is-active" if pressed else "") + (" is-disabled" if disabled else ""),
        "aria-disabled": disabled,
        "title": NO_TRACE if disabled else None,
        "aria-pressed": pressed,
    }


def _segment(kind: str, key: str, text: Any, state: Mapping[str, Any]) -> html.Button:
    return html.Button(text, id={"type": kind, "id": key}, n_clicks=0, **state)


def _recorded(index: EpisodeIndex, selection: Selection, *, policy: str | None = None, shield: bool | None = None) -> bool:
    if selection.scenario_id is None:
        return False
    found = index.find(selection.scenario_id, selection.policy if policy is None else policy, selection.shield if shield is None else shield)
    return found is not None


def _policy_text(policy: str, shield: bool, recorded: bool) -> list[html.Span]:
    spans = [html.Span(policy_label(policy), className="og-tab-name"), html.Span(f"{policy} policy · shield {'ON' if shield else 'OFF'}", className="og-tab-sub")]
    return spans if recorded else [*spans, html.Span(NO_TRACE, className="og-tab-reason")]


def _shield_text(key: str, recorded: bool) -> list[Any]:
    spans: list[Any] = [html.Span(_SHIELD_GLYPHS[key], className=f"og-glyph og-glyph-{key}"), key.upper()]
    return spans if recorded else [*spans, html.Span(NO_TRACE_SHORT, className="og-seg-reason")]


def control_states(index: EpisodeIndex, selection: Selection) -> dict[str, dict[str, dict[str, Any]]]:
    """Per control kind and id, the props that depend on the selection; layout and callbacks share this one source."""
    scenarios = {
        s.scenario_id: _state(pressed=s.scenario_id == selection.scenario_id, disabled=False, class_name="og-seg")
        for s in index.scenarios()
    }
    policies = {
        policy: {
            **_state(pressed=policy == selection.policy, disabled=not (ok := _recorded(index, selection, policy=policy)), class_name="og-tab"),
            "children": _policy_text(policy, selection.shield, ok),
        }
        for policy in index.policies()
    }
    shields = {
        key: {
            **_state(pressed=selection.shield == value, disabled=not (ok := _recorded(index, selection, shield=value)), class_name="og-seg"),
            "children": _shield_text(key, ok),
        }
        for key, value in (("on", True), ("off", False))
    }
    return {"og-scenario": scenarios, "og-policy": policies, "og-shield": shields}


def _scenario_group(index: EpisodeIndex, states: Mapping[str, Mapping[str, Any]]) -> html.Div:
    buttons = [
        _segment("og-scenario", s.scenario_id, [html.Span(s.label, className="og-seg-label"), html.Span(f"seed {s.seed}", className="og-seg-sub")], states[s.scenario_id])
        for s in index.scenarios()
    ]
    return html.Div(
        [html.Span("SCENARIO", className="og-kicker"), html.Div(buttons, className="og-segmented", role="group", **{"aria-label": "Scenario"})],
        className="og-scenarios",
    )


def meta_children(index: EpisodeIndex, selection: Selection, payload: Mapping[str, Any] | None) -> list[html.Span]:
    """Model, condition, seed, size and the recorded outcome of the selected episode ("—" when none is loaded)."""
    entry = None if selection.scenario_id is None else index.find(selection.scenario_id, selection.policy, selection.shield)
    if entry is None:
        return [html.Span(DASH)]
    meta = entry.meta
    size = f"{meta.n_agents} sats" + ("" if payload is None else f" · {payload['dims']['steps']} × {payload['dims']['control_dt_s']:g} s")
    children = [html.Span(meta.model), html.Span(meta.condition), html.Span(f"seed {meta.seed} · {size}")]
    if payload is not None:
        collided = payload["summary"]["collisions"] > 0
        glyph, text = ("✕", "COLLISION") if collided else ("●", "NO COLLISION")
        children.append(html.Span(f"{glyph} {text}", className="og-outcome" + (" is-collision" if collided else "")))
    return children


def _header(index: EpisodeIndex, selection: Selection, payload: Mapping[str, Any] | None, states: Mapping[str, Any]) -> html.Header:
    return html.Header(
        [
            html.Div([html.H1("ORBITGUARD", className="og-wordmark"), html.Span("EPISODE REPLAY", className="og-kicker")], className="og-brand"),
            _scenario_group(index, states["og-scenario"]),
            html.Div(meta_children(index, selection, payload), id="og-meta-line", className="og-meta-line"),
        ],
        className="og-header",
    )


def _policy_tabs(states: Mapping[str, Mapping[str, Any]]) -> html.Div:
    tabs = [_segment("og-policy", policy, state["children"], {k: v for k, v in state.items() if k != "children"}) for policy, state in states.items()]
    return html.Div(tabs, className="og-tabs", role="group", **{"aria-label": "Policy"})


def _shield_control(states: Mapping[str, Mapping[str, Any]]) -> html.Div:
    buttons = [_segment("og-shield", key, states[key]["children"], {k: v for k, v in states[key].items() if k != "children"}) for key in _SHIELD_GLYPHS]
    return html.Div(
        [html.Span("SHIELD", className="og-kicker"), html.Div(buttons, className="og-segmented", role="group", **{"aria-label": "Shield"})],
        className="og-shield-control",
    )


def shield_off_bar(selection: Selection) -> html.Div | None:
    if selection.shield or selection.scenario_id is None:
        return None
    return html.Div(
        [html.Span(SHIELD_OFF_TEXT, className="og-shield-off-text"), html.Span(SHIELD_OFF_NOTE, id="og-shield-off-events", className="og-shield-off-events")],
        id="og-shield-off-bar",
        className="og-shield-off-bar",
    )


def _pane(pane_id: str, title: str) -> html.Section:
    return html.Section([html.H2(title, className="og-kicker og-pane-title"), html.Div(id=f"{pane_id}-body", className="og-pane-body")], id=pane_id, className="og-pane")


def _empty_state() -> html.Div:
    return html.Div(
        [html.Div("No episode loaded", className="og-empty-title"), html.Div("Record traces with scripts/record_replay_traces.py, then point the app at that directory.", className="og-empty-hint")],
        id="og-empty",
        className="og-empty",
        role="status",
    )


def _loading(selection: Selection) -> html.Div:
    """Overlay shown until the stage has drawn; the stage fills the detail line so it can never describe another episode."""
    return html.Div(
        [html.Span(className="og-loading-rule"), html.Span("LOADING TRACE", className="og-loading-title"), html.Span(id="og-loading-detail", className="og-loading-detail")],
        id="og-loading",
        className="og-loading",
        role="status",
        hidden=selection.scenario_id is None,
    )


SCALE_CAPTION = "Satellite models are enlarged for visibility, not drawn to scale, and the light direction is fixed: visual aids, not data."


def _legend() -> html.Div:
    """Status vocabulary chips: colour + glyph + label, straight from the single vocabulary (PRD §41)."""
    chips = [
        html.Span(
            [html.Span(style.glyph, className="og-legend-glyph", style={"--chip": style.color}), html.Span(style.label, className="og-legend-label"), html.Span(className="og-legend-off")],
            className="og-legend-chip",
            **{"data-legend": style.name},
        )
        for style in (*SAT_VOCAB[1:], *EVENT_VOCAB[1:3])
    ]
    return html.Div([html.Div(chips, className="og-legend-chips"), html.Div(SCALE_CAPTION, className="og-legend-caption")], id="og-legend", className="og-legend")


def _stage_controls() -> html.Div:
    """Zoom, orbit-ring toggle and camera reset; the browser stage wires them (assets/19)."""
    def button(button_id: str, text: str, label: str, **extra: Any) -> html.Button:
        # symbol buttons need a spoken name; words are their own name (label in name, WCAG 2.5.3)
        spoken = {"aria-label": label} if len(text) == 1 else {}
        return html.Button(text, id=button_id, className="og-stage-btn", type="button", title=label, **spoken, **extra)

    return html.Div(
        [button("og-zoom-out", "−", "Zoom out"), button("og-zoom-in", "+", "Zoom in"), button("og-orbits", "Orbits", "Show or hide the orbit rings", **{"aria-pressed": True}), button("og-view-reset", "View", "Reset the camera")],
        className="og-stage-controls",
    )


def _stage(selection: Selection) -> html.Section:
    return html.Section(
        [
            html.Div([html.Div(id="og-stars-far", className="og-stars og-stars-far"), html.Div(id="og-stars-near", className="og-stars og-stars-near")], className="og-starfield", **{"aria-hidden": "true"}),
            html.Div(id="og-scene", className="og-scene", role="group", tabIndex=0, **{"aria-label": "3D constellation view", "aria-roledescription": "3D view; arrow keys rotate, plus and minus zoom"}),
            *(html.Span(className=f"og-corner og-corner-{corner}", **{"aria-hidden": "true"}) for corner in ("tl", "tr", "bl", "br")),
            _loading(selection),
            html.Div(id="og-clock", className="og-clock"),
            html.Div(id="og-banner", className="og-banner", **{"aria-hidden": "true"}),
            _legend(),
            _stage_controls(),
            html.Div([_pane("og-tracked", "Tracked this step"), _pane("og-inspector", "Inspector")], id="og-dock", className="og-dock"),
        ],
        id="og-stage",
        className="og-stage",
    )


PHONE_TABS = (("risk", "Risk"), ("timeline", "Timeline"), ("inspect", "Inspect"))
DEFAULT_PHONE_TAB = "risk"


def _phone_tabs() -> html.Div:
    """Risk / Timeline / Inspect switcher; only shown below 760 px (CSS), where the panes become one tab at a time."""
    buttons = [
        html.Button(label, id=f"og-phone-tab-{key}", className="og-seg og-phone-tab" + (" is-active" if key == DEFAULT_PHONE_TAB else ""), type="button", **{"aria-pressed": key == DEFAULT_PHONE_TAB, "data-tab": key})
        for key, label in PHONE_TABS
    ]
    return html.Div(buttons, className="og-segmented og-phone-tabs", role="group", **{"aria-label": "Panels"})


def _main(selection: Selection, payload: Mapping[str, Any] | None) -> html.Main:
    return html.Main(
        [
            html.Div(
                [
                    _stage(selection),
                    playback_bar(payload),
                    _phone_tabs(),
                    html.Div(id="og-phone-inspect", className="og-phone-inspect"),
                    html.Section([html.H2("Timeline", className="og-kicker og-pane-title"), html.Div(id="og-timeline-body")], id="og-timeline", className="og-pane og-timeline"),
                ],
                className="og-col-main",
            ),
            html.Aside([_pane(pane_id, title) for pane_id, title in _PANES], id="og-rail-right", className="og-rail", **{"aria-label": "Fleet fuel and conjunction risk"}),
        ],
        className="og-main",
    )


def build_layout(index: EpisodeIndex, selection: Selection, payload: Mapping[str, Any] | None = None) -> html.Div:
    empty = selection.scenario_id is None
    states = control_states(index, selection)
    return html.Div(
        [
            html.A("Skip to playback controls", href="#og-playback", className="og-skip"),
            html.Div(id="og-live", className="og-sr-only", role="status", **{"aria-live": "polite"}),
            html.Div(id="og-alert", className="og-sr-only", role="alert"),
            dcc.Store(id="og-payload", data=payload),
            dcc.Store(id="og-selection", data=asdict(selection)),
            _header(index, selection, payload, states),
            html.Div(
                [
                    html.Div([_policy_tabs(states["og-policy"]), _shield_control(states["og-shield"])], className="og-tabbar-row"),
                    html.Div(shield_off_bar(selection), id="og-shield-off-slot", role="status"),
                ],
                id="og-tabbar",
                className="og-tabbar",
            ),
            summary_strip(payload),
            _empty_state() if empty else html.Div(id="og-empty-slot"),
            _main(selection, payload),
        ],
        id="og-app",
        className="og-app is-empty" if empty else "og-app",
        **{"data-phone-tab": DEFAULT_PHONE_TAB},
    )
