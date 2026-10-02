"""Dash replay app (PRD §41, FR16): standalone, loads a saved `EpisodeTrace`.

    python -m orbit_guard.visualization.app --trace episode.npz [--port 8050]

Views: constellation overview (3D) . time playback . conjunction/risk panel .
per-satellite inspector . maneuver/shield event timeline . key-metrics panel.
Nothing here imports Ray or torch; replay needs no training or inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html

from orbit_guard.visualization.figures import (
    EVENT_STYLE,
    GRID,
    INK,
    MUTED,
    PAIR_STYLE,
    PANEL,
    SAT_STYLE,
    TEXT,
    constellation_figure,
    event_timeline_figure,
    inspector_figure,
    metric_cards,
    pair_rows,
)
from orbit_guard.visualization.replay import EpisodeTrace, frame_at, load_trace

_TRAIL_STEPS = 8
_PLAY_INTERVAL_MS = 450
_FONT = "'IBM Plex Sans', Inter, system-ui, sans-serif"
_MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
_FONTS_URL = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap"


def render_step(trace: EpisodeTrace, step: int, satellite: int) -> dict:
    """All panel contents for one (step, selected satellite) -- pure, so it is testable without a browser."""
    step = max(0, min(int(step), trace.num_steps))
    frame = frame_at(trace, step)
    rows = pair_rows(frame, limit=8)
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th(h) for h in ("pair", "status", "d_min (m)", "d_safe (m)", "risk")])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(row["pair"]),
                            html.Td(row["status_label"], style={"color": PAIR_STYLE[row["status"]].color, "fontWeight": 600}),
                            html.Td(f"{row['d_min_m']:,.0f}"),
                            html.Td(f"{row['d_safe_m']:,.0f}"),
                            html.Td(f"{row['risk']:.3f}"),
                        ]
                    )
                    for row in rows
                ]
            ),
        ],
        className="pairs",
    )
    sat = frame.satellites[satellite]
    events = [
        html.Li(
            f"{EVENT_STYLE[s.event].glyph} S{s.index}: {EVENT_STYLE[s.event].label}"
            + (f" (proposed {s.proposed} → approved {s.approved})" if s.event == "shield_intervention" else ""),
            style={"color": EVENT_STYLE[s.event].color},
        )
        for s in frame.satellites
        if s.event != "none"
    ] or [html.Li("no maneuver or shield events on this step", style={"color": MUTED})]
    status = (
        f"t = {frame.time_s / 60.0:.0f} min · step {step}/{trace.num_steps} · "
        f"S{satellite}: {SAT_STYLE[sat.status].glyph} {SAT_STYLE[sat.status].label} · "
        f"Δv left {sat.remaining_dv_mps:.2f} m/s"
    )
    return {
        "constellation": constellation_figure(frame, trace, trail=_TRAIL_STEPS),
        "pairs": table,
        "inspector": inspector_figure(trace, satellite, step),
        "timeline": event_timeline_figure(trace, step),
        "status": status,
        "events": html.Ul(events, className="events"),
    }


def _legend() -> html.Div:
    def chip(style) -> html.Span:
        return html.Span(
            [html.Span(style.glyph, style={"color": style.color, "marginRight": "6px", "fontWeight": 700}), style.label],
            className="chip",
        )

    return html.Div(
        [chip(s) for s in (*SAT_STYLE.values(), *EVENT_STYLE.values())], className="legend", id="legend"
    )


_CSS = f"""
body {{ margin: 0; background: {INK}; color: {TEXT}; font-family: {_FONT}; }}
.shell {{ max-width: 1440px; margin: 0 auto; padding: 18px 20px 28px; }}
h1 {{ font-size: 20px; letter-spacing: .14em; text-transform: uppercase; margin: 0; font-weight: 600; }}
.sub {{ color: {MUTED}; font-family: {_MONO}; font-size: 12px; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(320px, 1fr); gap: 14px; margin-top: 14px; }}
.panel {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 6px; padding: 10px 12px; }}
.panel h2 {{ font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: {MUTED}; margin: 0 0 8px; font-weight: 600; }}
.cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
.card {{ border-left: 3px solid {GRID}; padding: 4px 8px; }}
.card .v {{ font-family: {_MONO}; font-size: 22px; font-weight: 600; }}
.card .l {{ font-size: 11px; color: {MUTED}; }}
table.pairs {{ width: 100%; border-collapse: collapse; font-family: {_MONO}; font-size: 12px; }}
table.pairs th {{ text-align: left; color: {MUTED}; font-weight: 400; padding: 2px 6px; border-bottom: 1px solid {GRID}; }}
table.pairs td {{ padding: 3px 6px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; margin-top: 8px; }}
.chip {{ white-space: nowrap; }}
.controls {{ display: flex; gap: 12px; align-items: center; margin-top: 10px; }}
button {{ background: {GRID}; color: {TEXT}; border: 1px solid {MUTED}; border-radius: 4px; padding: 6px 14px; cursor: pointer;
  font-family: {_MONO}; }}
button:hover, button:focus-visible {{ background: #34456a; outline: 2px solid #3dd6e6; }}
.events {{ list-style: none; padding: 0; margin: 0; font-family: {_MONO}; font-size: 12px; }}
.status {{ font-family: {_MONO}; font-size: 12px; color: {TEXT}; }}
@media (max-width: 960px) {{ .grid {{ grid-template-columns: 1fr; }} .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
"""


def create_app(trace: EpisodeTrace) -> Dash:
    app = Dash(__name__, title="OrbitGuard replay", external_stylesheets=[_FONTS_URL])
    app.index_string = app.index_string.replace("</head>", f"<style>{_CSS}</style></head>")
    meta = trace.meta
    outcome = "COLLISION" if trace.summary.get("collision_count", 0) else "no collision"
    marks = {0: "0", trace.num_steps: str(trace.num_steps)}

    app.layout = html.Div(
        className="shell",
        children=[
            html.Header(
                [
                    html.H1("OrbitGuard · episode replay"),
                    html.Div(
                        f"model {meta.get('model', '?')} · condition {meta.get('condition', 'normal')} · "
                        f"seed {meta.get('seed', '?')} · {trace.num_agents} satellites · {outcome}",
                        className="sub",
                    ),
                ]
            ),
            html.Main(
                className="grid",
                children=[
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H2("Constellation"),
                                    dcc.Graph(id="constellation", config={"displaylogo": False}),
                                    _legend(),
                                    html.Div(
                                        [
                                            html.Button("▶ play", id="play", n_clicks=0),
                                            html.Div(
                                                dcc.Slider(
                                                    id="step-slider", min=0, max=trace.num_steps, step=1, value=0,
                                                    marks=marks, updatemode="drag",
                                                ),
                                                style={"flex": 1},
                                            ),
                                        ],
                                        className="controls",
                                    ),
                                    html.Div(id="status", className="status"),
                                ],
                                className="panel",
                            ),
                            html.Div(
                                [html.H2("Maneuver & shield timeline"), dcc.Graph(id="event-timeline", config={"displaylogo": False})],
                                className="panel", style={"marginTop": "14px"},
                            ),
                        ]
                    ),
                    html.Aside(
                        [
                            html.Div(
                                [
                                    html.H2("Key metrics"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [html.Div(c["value"], className="v"), html.Div(c["label"], className="l")],
                                                className="card",
                                            )
                                            for c in metric_cards(trace)
                                        ],
                                        className="cards", id="metrics",
                                    ),
                                ],
                                className="panel",
                            ),
                            html.Div(
                                [html.H2("Conjunction & risk"), html.Div(id="pair-panel")],
                                className="panel", style={"marginTop": "14px"},
                            ),
                            html.Div(
                                [html.H2("This step"), html.Div(id="events")],
                                className="panel", style={"marginTop": "14px"},
                            ),
                            html.Div(
                                [
                                    html.H2("Satellite inspector"),
                                    dcc.Dropdown(
                                        id="inspector-satellite",
                                        options=[{"label": f"S{i}", "value": i} for i in range(trace.num_agents)],
                                        value=0, clearable=False,
                                        style={"color": "#111", "marginBottom": "8px"},
                                    ),
                                    dcc.Graph(id="inspector", config={"displaylogo": False}),
                                ],
                                className="panel", style={"marginTop": "14px"},
                            ),
                        ]
                    ),
                ],
            ),
            dcc.Interval(id="tick", interval=_PLAY_INTERVAL_MS, disabled=True),
        ],
    )

    @app.callback(
        Output("tick", "disabled"), Output("play", "children"),
        Input("play", "n_clicks"), State("tick", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_play(_clicks, disabled):
        return (not disabled), ("⏸ pause" if disabled else "▶ play")

    @app.callback(
        Output("step-slider", "value"),
        Input("tick", "n_intervals"), State("step-slider", "value"),
        prevent_initial_call=True,
    )
    def advance(_ticks, value):
        return 0 if value >= trace.num_steps else value + 1

    @app.callback(
        Output("constellation", "figure"), Output("pair-panel", "children"), Output("inspector", "figure"),
        Output("event-timeline", "figure"), Output("status", "children"), Output("events", "children"),
        Input("step-slider", "value"), Input("inspector-satellite", "value"),
    )
    def update(step, satellite):
        out = render_step(trace, step or 0, satellite or 0)
        return (out["constellation"], out["pairs"], out["inspector"], out["timeline"], out["status"], out["events"])

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a saved OrbitGuard episode in the browser.")
    parser.add_argument("--trace", required=True, help="an episode .npz written by visualization.replay.save_trace")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    create_app(load_trace(Path(args.trace))).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
