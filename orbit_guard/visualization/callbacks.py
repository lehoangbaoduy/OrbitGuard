"""Server side of the replay app: payload provider and the one selection callback.

The server only answers "which episode?" (a click on a scenario, policy or shield control) and hands the browser that
episode's payload; stepping, playback and every redraw happen client-side. Controls are updated in place through
wildcard outputs (not rebuilt), so keyboard focus survives a click.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dash import ALL, Dash, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from orbit_guard.visualization.episodes import EpisodeIndex, read_episode_summary
from orbit_guard.visualization.layout import (
    control_states,
    meta_children,
    shield_off_bar,
    summary_strip,
)
from orbit_guard.visualization.payload import NOTHING_POLICY, build_payload, build_strip
from orbit_guard.visualization.reference import ReferenceLogError, reference_note
from orbit_guard.visualization.replay import load_trace
from orbit_guard.visualization.replay_meta import ReplayMeta, ReplayMetaError
from orbit_guard.visualization.selection import (
    Selection,
    initial_selection,
    resolve_selection,
)
from orbit_guard.visualization.source import ReplaySource

_CACHE_SIZE = 6  # a payload is about 1 MB; keeps tab and shield toggling instant without growing unbounded
_CONTROL_PROPS = {
    "og-scenario": ("className", "aria-pressed"),
    "og-policy": ("className", "aria-disabled", "title", "aria-pressed", "children"),
    "og-shield": ("className", "aria-disabled", "title", "aria-pressed", "children"),
}
_FIXED_OUTPUTS = 6  # selection, payload, strip, shield-off slot, meta line, alert; the control outputs follow
_SHIELD_KEYS = {"on": True, "off": False}
_BASELINE_MODEL = "no_maneuver"


class PayloadProvider:
    """Browser payloads for recorded episodes, built from the trace on first use and cached (read-only)."""

    def __init__(self, index: EpisodeIndex, reference_log: str | Path | None = None):
        self._index = index
        self._reference_log = reference_log
        self._cache: OrderedDict[Selection, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()  # Flask serves requests on threads; the LRU's reorder/evict must not interleave

    def __call__(self, selection: Selection) -> dict[str, Any] | None:
        if selection.scenario_id is None or selection.policy is None:
            return None
        entry = self._index.find(selection.scenario_id, selection.policy, selection.shield)
        if entry is None:
            return None
        with self._lock:
            cached = self._cache.get(selection)
            if cached is not None:
                self._cache.move_to_end(selection)
        if cached is not None:
            return cached
        payload = self._build(selection, entry.path, entry.meta)  # outside the lock: building takes a second
        with self._lock:
            self._cache[selection] = payload
            self._cache.move_to_end(selection)
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return payload

    def _build(self, selection: Selection, path: Path, meta: ReplayMeta) -> dict[str, Any]:
        totals = {}
        for policy in ("untrained", "trained"):
            other = self._index.find(selection.scenario_id, policy, selection.shield)
            totals[policy] = None if other is None else read_episode_summary(other.path)
        if NOTHING_POLICY in self._index.policies():  # the do-nothing lane: shield-off only, shown at either setting
            nothing = self._index.find(selection.scenario_id, NOTHING_POLICY, False)
            totals[NOTHING_POLICY] = None if nothing is None else read_episode_summary(nothing.path)
        note = reference_note(self._reference_log, meta.condition, (meta.model, _BASELINE_MODEL), n_agents=meta.n_agents)
        source = ReplaySource(load_trace(path))
        return build_payload(source, strip=build_strip(selection.policy, totals), reference_note=note)


def parse_selection(index: EpisodeIndex, state: Any) -> Selection:
    """The browser's stored selection, trusted only if it names a recorded episode (else the initial selection)."""
    if isinstance(state, dict):
        scenario, policy, shield = state.get("scenario_id"), state.get("policy"), state.get("shield")
        if isinstance(scenario, str) and isinstance(policy, str) and isinstance(shield, bool) and index.find(scenario, policy, shield):
            return Selection(scenario, policy, shield)
    return initial_selection(index)


def apply_click(index: EpisodeIndex, current: Selection, kind: str, key: str) -> Selection:
    """The selection after one click; a control for a combination without a trace changes nothing."""
    if kind == "og-scenario":
        return resolve_selection(index, current, scenario_id=key)
    if kind == "og-policy":
        return resolve_selection(index, current, policy=key)
    if kind == "og-shield" and key in _SHIELD_KEYS:
        return resolve_selection(index, current, shield=_SHIELD_KEYS[key])
    return current


def register_callbacks(app: Dash, index: EpisodeIndex, provider: PayloadProvider) -> None:
    outputs = [
        Output("og-selection", "data"),
        Output("og-payload", "data"),
        Output("og-strip", "children"),
        Output("og-shield-off-slot", "children"),
        Output("og-meta-line", "children"),
        Output("og-alert", "children"),
        *(Output({"type": kind, "id": ALL}, prop) for kind, props in _CONTROL_PROPS.items() for prop in props),
    ]
    inputs = [Input({"type": kind, "id": ALL}, "n_clicks") for kind in _CONTROL_PROPS]

    @app.callback(*outputs, *inputs, State("og-selection", "data"), prevent_initial_call=True)
    def select_episode(*args):
        triggered = ctx.triggered_id
        current = parse_selection(index, args[-1])
        if not isinstance(triggered, dict):
            raise PreventUpdate
        selection = apply_click(index, current, str(triggered.get("type")), str(triggered.get("id")))
        if selection == current:  # nothing changed (active or unavailable control): the stage must not see a new episode
            raise PreventUpdate
        try:
            payload = provider(selection)
        except (ReplayMetaError, FileNotFoundError, ReferenceLogError) as error:  # a trace or reference log damaged or removed since startup
            return _failure(ctx.outputs_list, str(error))
        states = control_states(index, selection)
        control_lists = [[_control_value(states, spec) for spec in group] for group in ctx.outputs_list[_FIXED_OUTPUTS:]]
        return [asdict(selection), payload, summary_strip(payload).children, shield_off_bar(selection), meta_children(index, selection, payload), "", *control_lists]


def _failure(outputs_list: list[Any], message: str) -> list[Any]:
    """Leave everything as it is and say why in the alert region (the episode could not be loaded)."""
    alert = f"Could not load this episode: {message}"
    untouched = [[no_update] * len(group) if isinstance(group, list) else no_update for group in outputs_list]
    untouched[_FIXED_OUTPUTS - 1] = alert
    return untouched


def _control_value(states: dict[str, dict[str, dict[str, Any]]], spec: dict[str, Any]) -> Any:
    """One control output; a control this server does not know (a stale page after a restart) is left alone."""
    ident = spec["id"]
    state = states.get(ident["type"], {}).get(ident["id"])
    return no_update if state is None or spec["property"] not in state else state[spec["property"]]
