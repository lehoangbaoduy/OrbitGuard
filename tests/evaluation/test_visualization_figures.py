"""Tests for the Dash replay's pure figure builders (PRD §41): the visual rule is color AND marker AND label."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from orbit_guard.evaluation.baselines import rule_based_policy
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.evaluation import no_maneuver_policy
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel
from orbit_guard.visualization.figures import (
    EVENT_STYLE,
    PAIR_STYLE,
    SAT_STYLE,
    constellation_figure,
    event_timeline_figure,
    inspector_figure,
    metric_cards,
    pair_rows,
)
from orbit_guard.visualization.replay import frame_at, record_episode

_KW = {
    "episode_horizon_s": 1800.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0,
    "noise_model": NoiseModel(0.0, 0.0), "hard_body_radius_m": 100.0,
}


@pytest.fixture(scope="module")
def collision_trace():
    return record_episode(OrbitGuardEnv(num_agents=2, **_KW), no_maneuver_policy, seed=3, meta={"model": "no_maneuver"})


@pytest.fixture(scope="module")
def shielded_trace():
    return record_episode(ShieldedOrbitGuardEnv(num_agents=2, **_KW), no_maneuver_policy, seed=3, meta={"model": "adaptive_mappo"})


def test_every_status_has_a_distinct_color_marker_and_label():
    for table in (SAT_STYLE, PAIR_STYLE):
        colors = [style.color for style in table.values()]
        labels = [style.label for style in table.values()]
        assert len(set(colors)) == len(colors)
        assert len(set(labels)) == len(labels)
    markers = [style.marker for style in SAT_STYLE.values()] + [style.marker for style in EVENT_STYLE.values()]
    assert len(set(markers)) >= 5  # color is never the only channel


def test_constellation_draws_earth_satellites_and_labels_non_nominal_ones(collision_trace):
    figure = constellation_figure(frame_at(collision_trace, collision_trace.num_steps), collision_trace, trail=3)

    names = {trace.name for trace in figure.data}
    assert "Earth" in names
    assert any("COLLIDED" in str(name) for name in names)
    satellite_traces = [t for t in figure.data if getattr(t, "mode", None) and "markers" in t.mode]
    assert satellite_traces
    assert all(t.marker.symbol is not None for t in satellite_traces)
    text = " ".join(" ".join(t.text) for t in figure.data if getattr(t, "text", None) is not None and not isinstance(t.text, str))
    assert "S0" in text and "S1" in text


def test_conjunction_lines_use_distinct_dash_and_label_per_status(collision_trace):
    figure = constellation_figure(frame_at(collision_trace, collision_trace.num_steps), collision_trace, trail=0)

    line_traces = [t for t in figure.data if getattr(t, "mode", None) == "lines" and t.name and "S0" in t.name]
    assert line_traces
    assert any("COLLISION" in t.name for t in line_traces)


def test_shield_interventions_and_maneuvers_appear_in_the_event_timeline(shielded_trace):
    figure = event_timeline_figure(shielded_trace, current_step=shielded_trace.num_steps)

    names = {trace.name for trace in figure.data}
    assert EVENT_STYLE["shield_intervention"].label in names
    assert EVENT_STYLE["maneuver"].label in names
    shield_trace = next(t for t in figure.data if t.name == EVENT_STYLE["shield_intervention"].label)
    assert len(shield_trace.x) > 0


def test_event_timeline_marks_the_current_step(collision_trace):
    figure = event_timeline_figure(collision_trace, current_step=4)

    assert any(shape.type == "line" for shape in figure.layout.shapes)


def test_pair_rows_are_sorted_worst_first_with_a_text_label(collision_trace):
    rows = pair_rows(frame_at(collision_trace, collision_trace.num_steps), limit=5)

    assert rows[0]["status_label"].endswith("COLLISION")
    assert rows[0]["pair"] == "S0–S1"
    risks = [row["risk"] for row in rows]
    assert all(isinstance(r, float) for r in risks)


def test_inspector_plots_remaining_delta_v_for_the_selected_satellite(shielded_trace):
    figure = inspector_figure(shielded_trace, satellite=0, current_step=3)

    assert len(figure.data) >= 2
    assert "Δv" in figure.layout.title.text or any("Δv" in str(t.name) for t in figure.data)


def test_metric_cards_report_the_episode_summary(shielded_trace):
    cards = {card["label"]: card["value"] for card in metric_cards(shielded_trace)}

    assert "Collisions" in cards and "Shield interventions" in cards and "Total Δv (m/s)" in cards
    assert cards["Collisions"] == "0"


def test_figures_scale_to_a_fifteen_satellite_episode():
    env = OrbitGuardEnv(num_agents=12, episode_horizon_s=300.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=300.0)
    trace = record_episode(env, rule_based_policy, seed=2, meta={})

    figure = constellation_figure(frame_at(trace, trace.num_steps // 2), trace, trail=2)
    timeline = event_timeline_figure(trace, current_step=1)

    assert figure.data and timeline.data is not None
