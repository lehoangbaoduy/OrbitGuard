"""The summary strip (design: five cells, BEFORE/AFTER bars on one shared scale), server-rendered from the payload.

Every number is a recorded episode total from the payload (`summary`, `strip`); the running value at the current step
is filled in by the JS stage into the empty `og-strip-run-*` spans. Missing data is shown as missing: the Shield cell
is "—" when the shield was off (never 0), and a policy without a recorded counterpart has no bar.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dash import html

from orbit_guard.visualization.payload import METRICS

DASH = "—"
NO_TRACE = "no trace recorded"
NOT_APPLICABLE = "shield off"

_LABELS = {
    "delta_v_mps": ("TOTAL Δv", "Δv"),
    "maneuvers": ("MANEUVERS", "MAN"),
    "shield": ("SHIELD", "SH"),
    "collisions": ("COLLISIONS", "COL"),
    "mean_deviation": ("MISSION DEVIATION", "DEV"),
}
_UNITS = {"delta_v_mps": "m/s", "mean_deviation": "mean"}
_FORMATS: dict[str, Callable[[float], str]] = {
    "delta_v_mps": lambda v: f"{v:.2f}",
    "maneuvers": lambda v: str(int(v)),
    "shield": lambda v: str(int(v)),
    "collisions": lambda v: str(int(v)),
    "mean_deviation": lambda v: f"{v:.3f}",
}
_POLICY_ROWS = (("before", "BEFORE"), ("after", "AFTER"))
_ACTIVE_ROW = {"untrained": "before", "trained": "after"}


def format_metric(metric: str, value: float | None) -> str:
    return DASH if value is None else _FORMATS[metric](value)


def _scale(value: float, largest: float) -> str:
    return f"scaleX({value / largest:.4g})" if largest > 0 else "scaleX(0)"


def _bar_row(metric: str, key: str, label: str, value: float | None, largest: float, *, active: bool, shield_off: bool) -> html.Div:
    classes = f"og-bar-row og-bar-{key}" + (" is-active" if active else "")
    if value is None:
        reason = NOT_APPLICABLE if shield_off and metric == "shield" else NO_TRACE
        return html.Div([html.Span(label, className="og-bar-label"), html.Span(reason, className="og-bar-missing")], className=classes)
    return html.Div(
        [
            html.Span(label, className="og-bar-label"),
            html.Span(
                html.Span(id=f"og-strip-bar-{metric}-{key}", className="og-bar-fill", style={"transform": _scale(value, largest)}),
                className="og-bar-track",
            ),
            html.Span(format_metric(metric, value), className="og-bar-value"),
        ],
        className=classes,
    )


def _bars(metric: str, entry: Mapping[str, Any], active_policy: str | None, shield_off: bool) -> html.Div:
    values = [entry.get("before"), entry.get("after")]
    largest = max((v for v in values if v is not None), default=0.0)
    active = _ACTIVE_ROW.get(active_policy or "")
    rows = [
        _bar_row(metric, key, label, value, largest, active=key == active, shield_off=shield_off)
        for (key, label), value in zip(_POLICY_ROWS, values, strict=True)
    ]
    return html.Div(rows, className="og-cell-bars")


def _cell(metric: str, payload: Mapping[str, Any] | None) -> html.Div:
    label, short = _LABELS[metric]
    summary = (payload or {}).get("summary") or {}
    strip = (payload or {}).get("strip")
    meta = (payload or {}).get("meta") or {}
    value = summary.get(metric)
    children = [
        html.Div([html.Span(label, className="og-cell-label-full"), html.Span(short, className="og-cell-label-short", **{"aria-hidden": "true"})], className="og-kicker"),
        html.Div(
            [
                html.Span(format_metric(metric, value), id=f"og-strip-val-{metric}", className="og-cell-val"),
                html.Span(_UNITS.get(metric, ""), className="og-cell-unit"),
            ],
            className="og-cell-main",
        ),
    ]
    if strip is not None:
        entry = strip.get(metric) or {}
        children.append(_bars(metric, entry, meta.get("policy"), meta.get("shield") is False))
        children.append(html.Div(entry.get("note") or "", id=f"og-strip-note-{metric}", className="og-cell-note"))
    children.append(html.Div(id=f"og-strip-run-{metric}", className="og-cell-run"))
    return html.Div(children, id=f"og-strip-cell-{metric}", className="og-cell")


def summary_strip(payload: Mapping[str, Any] | None) -> html.Div:
    return html.Div([_cell(metric, payload) for metric in METRICS], id="og-strip", className="og-strip", role="group", **{"aria-label": "Episode totals"})
