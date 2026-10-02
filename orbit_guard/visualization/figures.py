"""Pure Plotly figure/table builders for the replay app (PRD §41).

Visual rule (PRD §41): collision, high-risk conjunction, inside-`d_safe`, shield
intervention and applied manoeuvre must be told apart with colour AND a marker
shape AND a text label -- never colour alone. Every state below therefore has a
`Style(color, marker, label)`, and the legend shows all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from orbit_guard.visualization.replay import EpisodeTrace, Frame

EARTH_RADIUS_KM = 6378.137
INK = "#0b1220"
PANEL = "#111a2e"
GRID = "#243049"
TEXT = "#d7e0f0"
MUTED = "#7d8ba1"


@dataclass(frozen=True)
class Style:
    color: str
    marker: str
    label: str
    dash: str = "solid"
    width: float = 2.0
    glyph: str = "●"


SAT_STYLE = {
    "collided": Style("#ff3b6b", "x", "COLLIDED", glyph="✕"),
    "high_risk": Style("#ffb020", "diamond", "HIGH RISK", glyph="▲"),
    "inside_d_safe": Style("#3dd6e6", "circle-open", "INSIDE d_safe", glyph="◎"),
    "nominal": Style(MUTED, "circle", "nominal", glyph="●"),
}
PAIR_STYLE = {
    "collision": Style("#ff3b6b", "x", "COLLISION", dash="solid", width=5, glyph="✕"),
    "high_risk": Style("#ffb020", "diamond", "HIGH RISK", dash="dash", width=4, glyph="▲"),
    "inside_d_safe": Style("#3dd6e6", "circle-open", "INSIDE d_safe", dash="dot", width=3, glyph="◎"),
    "nominal": Style(GRID, "circle", "nominal", dash="solid", width=1, glyph="·"),
}
EVENT_STYLE = {
    "maneuver": Style("#4ade80", "square", "MANEUVER", glyph="■"),
    "shield_intervention": Style("#b794f6", "diamond-open", "SHIELD INTERVENTION", glyph="◇"),
    "collision": Style("#ff3b6b", "x", "COLLISION", glyph="✕"),
}
_PAIR_ORDER = ("collision", "high_risk", "inside_d_safe", "nominal")


def _km(position_m) -> np.ndarray:
    return np.asarray(position_m, dtype=float) / 1000.0


def _earth_surface() -> go.Surface:
    u, v = np.mgrid[0 : 2 * np.pi : 36j, 0 : np.pi : 18j]
    return go.Surface(
        x=EARTH_RADIUS_KM * np.cos(u) * np.sin(v), y=EARTH_RADIUS_KM * np.sin(u) * np.sin(v),
        z=EARTH_RADIUS_KM * np.cos(v), colorscale=[[0, "#0f2a4a"], [1, "#16406e"]], showscale=False,
        opacity=0.9, hoverinfo="skip", name="Earth",
    )


def constellation_figure(frame: Frame, trace: EpisodeTrace, trail: int = 8) -> go.Figure:
    """3D constellation at one frame: satellites (status), conjunction lines (pair status), event rings, trails."""
    figure = go.Figure(_earth_surface())

    for pair in sorted(frame.pairs, key=lambda p: _PAIR_ORDER.index(p.status)):
        if pair.status == "nominal":
            continue
        style = PAIR_STYLE[pair.status]
        a, b = (_km(frame.satellites[index].position_m) for index in (pair.i, pair.j))
        figure.add_trace(
            go.Scatter3d(
                x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
                line={"color": style.color, "dash": style.dash, "width": style.width * 2},
                name=f"S{pair.i}–S{pair.j} {style.label}",
                hovertemplate=(
                    f"S{pair.i}–S{pair.j}: {style.label}<br>predicted d_min {pair.d_min_m:,.0f} m"
                    f"<br>d_safe {pair.d_safe_m:,.0f} m<br>risk {pair.risk:.3f}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    t = frame.step
    for status, style in SAT_STYLE.items():
        members = [sat for sat in frame.satellites if sat.status == status]
        if not members:
            continue
        coords = np.array([_km(sat.position_m) for sat in members])
        figure.add_trace(
            go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers+text",
                marker={"symbol": style.marker, "color": style.color, "size": 7 if status == "nominal" else 10,
                        "line": {"color": style.color, "width": 2}},
                text=[f"S{sat.index}" if status == "nominal" else f"S{sat.index} · {style.label}" for sat in members],
                textposition="top center", textfont={"color": style.color, "size": 11},
                name=f"{style.glyph} {style.label}",
                hovertext=[
                    f"S{sat.index} {style.label}<br>Δv left {sat.remaining_dv_mps:.2f} m/s"
                    f"<br>mission dev {sat.mission_deviation:.3f}" for sat in members
                ],
                hoverinfo="text",
            )
        )

    for event, style in EVENT_STYLE.items():
        members = [sat for sat in frame.satellites if sat.event == event]
        if not members:
            continue
        coords = np.array([_km(sat.position_m) for sat in members])
        figure.add_trace(
            go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers",
                marker={"symbol": style.marker, "color": style.color, "size": 15, "opacity": 0.9,
                        "line": {"color": style.color, "width": 3}},
                name=f"{style.glyph} {style.label}", hoverinfo="skip",
            )
        )

    if trail > 0 and t > 0:
        start = max(0, t - trail)
        for sat in frame.satellites:
            if sat.status == "nominal" and sat.event == "none":
                continue
            path = _km(trace.positions_m[start : t + 1, sat.index])
            style = SAT_STYLE[sat.status]
            figure.add_trace(
                go.Scatter3d(
                    x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
                    line={"color": style.color, "width": 2}, opacity=0.5, hoverinfo="skip", showlegend=False,
                )
            )

    axis = {"showbackground": False, "showgrid": False, "zeroline": False, "showticklabels": False, "title": ""}
    figure.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font={"color": TEXT, "family": "Inter, system-ui, sans-serif"},
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "data", "bgcolor": INK},
        margin={"l": 0, "r": 0, "t": 8, "b": 0}, height=560,
        legend={"bgcolor": "rgba(17,26,46,0.85)", "bordercolor": GRID, "borderwidth": 1, "font": {"size": 12}},
        uirevision="constellation",
    )
    return figure


def event_timeline_figure(trace: EpisodeTrace, current_step: int) -> go.Figure:
    """One row per satellite; marker shape + colour + legend label per event type; vertical line at `current_step`."""
    figure = go.Figure()
    minutes = np.arange(trace.num_steps + 1) * trace.control_dt_s / 60.0

    def add(event: str, steps: np.ndarray, sats: np.ndarray) -> None:
        style = EVENT_STYLE[event]
        figure.add_trace(
            go.Scatter(
                x=minutes[steps + 1], y=sats, mode="markers", name=style.label,
                marker={"symbol": style.marker, "color": style.color, "size": 11, "line": {"color": style.color, "width": 2}},
                hovertemplate=f"{style.label}<br>S%{{y}} at %{{x:.0f}} min<extra></extra>",
            )
        )

    burned = (trace.applied_dv_mps > 0) & ~trace.intervened
    add("maneuver", *np.nonzero(burned))
    add("shield_intervention", *np.nonzero(trace.intervened))
    hit_steps, hit_pairs = np.nonzero(trace.collided_pair)
    hit_sats = trace.pair_index[hit_pairs].ravel()
    add("collision", np.repeat(hit_steps, 2), hit_sats)

    figure.add_vline(x=float(minutes[current_step]), line={"color": TEXT, "width": 1, "dash": "dot"})
    figure.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=INK, font={"color": TEXT, "family": "Inter, system-ui, sans-serif"},
        xaxis={"title": "episode time (min)", "gridcolor": GRID, "range": [0, float(minutes[-1]) + 1]},
        yaxis={"title": "satellite", "gridcolor": GRID, "dtick": 1, "range": [-0.7, trace.num_agents - 0.3]},
        margin={"l": 50, "r": 10, "t": 8, "b": 40}, height=250, legend={"orientation": "h", "y": 1.18},
        uirevision="timeline",
    )
    return figure


def inspector_figure(trace: EpisodeTrace, satellite: int, current_step: int) -> go.Figure:
    """Remaining Δv and mission deviation of one satellite over the episode."""
    minutes = np.arange(trace.num_steps + 1) * trace.control_dt_s / 60.0
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=minutes, y=trace.remaining_dv_mps[:, satellite], name="Δv remaining (m/s)",
                   line={"color": "#4ade80", "width": 2})
    )
    figure.add_trace(
        go.Scatter(x=minutes, y=trace.mission_deviation[:, satellite], name="mission deviation",
                   line={"color": "#ffb020", "width": 2, "dash": "dash"}, yaxis="y2")
    )
    figure.add_vline(x=float(minutes[current_step]), line={"color": TEXT, "width": 1, "dash": "dot"})
    figure.update_layout(
        title={"text": f"S{satellite} — Δv budget and mission deviation", "font": {"size": 13}},
        paper_bgcolor=PANEL, plot_bgcolor=INK, font={"color": TEXT, "family": "Inter, system-ui, sans-serif"},
        xaxis={"title": "min", "gridcolor": GRID},
        yaxis={"title": "m/s", "gridcolor": GRID},
        yaxis2={"title": "deviation", "overlaying": "y", "side": "right", "showgrid": False},
        margin={"l": 50, "r": 50, "t": 34, "b": 36}, height=250, legend={"orientation": "h", "y": -0.3},
        uirevision="inspector",
    )
    return figure


def pair_rows(frame: Frame, limit: int = 8) -> list[dict]:
    """Conjunction panel rows, worst status first then highest risk; the status is TEXT plus a glyph."""
    ordered = sorted(frame.pairs, key=lambda p: (_PAIR_ORDER.index(p.status), -p.risk, p.d_min_m))
    return [
        {
            "pair": f"S{p.i}–S{p.j}",
            "status": p.status,
            "status_label": f"{PAIR_STYLE[p.status].glyph} {PAIR_STYLE[p.status].label}",
            "d_min_m": float(p.d_min_m),
            "d_safe_m": float(p.d_safe_m),
            "risk": float(p.risk),
        }
        for p in ordered[:limit]
    ]


def metric_cards(trace: EpisodeTrace) -> list[dict]:
    summary = trace.summary
    return [
        {"label": "Collisions", "value": str(summary.get("collision_count", 0))},
        {"label": "Unsafe conjunctions", "value": str(summary.get("unsafe_conjunction_count", 0))},
        {"label": "Total Δv (m/s)", "value": f"{summary.get('total_delta_v_mps', 0.0):.2f}"},
        {"label": "Maneuvers", "value": str(summary.get("maneuver_count", 0))},
        {"label": "Shield interventions", "value": str(summary.get("shield_intervention_count", 0))},
        {"label": "Mean mission deviation", "value": f"{summary.get('mean_mission_deviation', 0.0):.3f}"},
    ]
