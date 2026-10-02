"""Dash replay app (PRD §41, FR16): standalone, replays recorded episodes discovered from their trace meta.

    python -m orbit_guard.visualization.app --episodes DIR [--reference-log eval.jsonl] [--port 8050]
    python -m orbit_guard.visualization.app --trace episode.npz

The server picks the episode and serves its payload; the browser stage (assets/*.js, vendored plotly.js) draws and
plays it. Nothing here imports Ray, torch or gymnasium; replay needs no training or inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dash import ClientsideFunction, Dash, Input, Output

from orbit_guard.visualization.callbacks import PayloadProvider, register_callbacks
from orbit_guard.visualization.episodes import (
    EpisodeIndex,
    discover_episodes,
    index_from_trace,
)
from orbit_guard.visualization.layout import build_layout
from orbit_guard.visualization.selection import initial_selection


def create_app(index: EpisodeIndex, reference_log: str | Path | None = None) -> Dash:
    app = Dash(__name__, title="OrbitGuard replay")
    app.index_string = app.index_string.replace("<html>", '<html lang="en">', 1)
    provider = PayloadProvider(index, reference_log)
    selection = initial_selection(index)
    app.layout = build_layout(index, selection, provider(selection))
    register_callbacks(app, index, provider)
    app.clientside_callback(  # the browser stage decodes the payload and draws it (assets/19_og_controller.js)
        ClientsideFunction(namespace="og", function_name="boot"), Output("og-stage", "data-og-state"), Input("og-payload", "data")
    )
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay recorded OrbitGuard episodes in the browser.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episodes", help="directory of episode .npz traces (see scripts/record_replay_traces.py)")
    source.add_argument("--trace", help="a single episode .npz written by visualization.replay.save_trace")
    parser.add_argument("--reference-log", help="evaluation JSONL log for the reference collision note (hidden when omitted)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    index = discover_episodes(args.episodes) if args.episodes else index_from_trace(args.trace)
    create_app(index, args.reference_log).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
