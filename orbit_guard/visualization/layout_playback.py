"""The playback bar markup (transport buttons, scrubber, speed, loop); the browser stage wires it (assets/14, 18).

Everything here is structure and accessible names. The scrubber's range is the recorded number of steps from the
payload; its density bars and conjunction ticks are drawn client-side from the payload's own series.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dash import html

SPEEDS = ("1", "2", "4", "8")
DEFAULT_SPEED = "1"

_TRANSPORT = (
    ("og-prev-event", "Previous event", "[", "og-ico-prev-event"),
    ("og-step-back", "Step back", "ArrowLeft", "og-ico-step-back"),
    ("og-play", "Play", "Space", "og-ico-play"),
    ("og-step-fwd", "Step forward", "ArrowRight", "og-ico-step-fwd"),
    ("og-next-event", "Next event", "]", "og-ico-next-event"),
)


_KEY_NAMES = {"ArrowLeft": "←", "ArrowRight": "→"}


def _transport_button(button_id: str, label: str, key: str, icon: str) -> html.Button:
    extra: dict[str, Any] = {"aria-pressed": False} if button_id == "og-play" else {}
    return html.Button(
        html.Span(className=f"og-ico {icon}", **{"aria-hidden": "true"}),
        id=button_id,
        className="og-transport-btn" + (" og-play" if button_id == "og-play" else ""),
        title=f"{label} ({_KEY_NAMES.get(key, key)})",
        type="button",
        **{"aria-label": label, "aria-keyshortcuts": key},
        **extra,
    )


def _scrubber(payload: Mapping[str, Any] | None) -> html.Div:
    steps = None if payload is None else payload["dims"]["steps"]
    opening = None if payload is None else payload["start_step"]
    return html.Div(
        [
            html.Canvas(id="og-scrub-density", className="og-scrub-density"),
            html.Div(id="og-scrub-fill", className="og-scrub-fill"),
            html.Div(id="og-scrub-thumb", className="og-scrub-thumb"),
            html.Div(id="og-scrub-hover", className="og-scrub-hover", role="tooltip", hidden=True),
        ],
        id="og-scrub",
        className="og-scrub",
        role="slider",
        tabIndex=0,
        **{"aria-label": "Playhead", "aria-valuemin": 0, "aria-valuemax": steps, "aria-valuenow": opening},
    )


def _speed_group() -> html.Div:
    buttons = [
        html.Button(
            f"{speed}×",
            id={"type": "og-speed", "id": speed},
            title=f"Playback speed {speed}×",
            className="og-seg og-speed-btn" + (" is-active" if speed == DEFAULT_SPEED else ""),
            type="button",
            **{"aria-pressed": speed == DEFAULT_SPEED, "data-speed": speed, "aria-label": f"{speed}× speed"},
        )
        for speed in SPEEDS
    ]
    return html.Div(buttons, className="og-segmented og-speed", role="group", **{"aria-label": "Playback speed"})


def playback_bar(payload: Mapping[str, Any] | None) -> html.Div:
    return html.Div(
        [
            html.Div([_transport_button(*spec) for spec in _TRANSPORT], className="og-transport"),
            _scrubber(payload),
            html.Div(
                [_speed_group(), html.Button("Loop", id="og-loop", className="og-seg og-loop", type="button", **{"aria-pressed": False})],
                className="og-playback-options",
            ),
        ],
        id="og-playback",
        className="og-playback",
        role="group",
        tabIndex=-1,
        **{"aria-label": "Playback"},
    )
