"""Tests for the Dash replay app wiring (PRD §41, FR16): loads a saved trace, no trainer needed."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("dash")

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.evaluation import no_maneuver_policy
from orbit_guard.sensing.noise import NoiseModel
from orbit_guard.visualization.app import create_app, render_step
from orbit_guard.visualization.replay import record_episode, save_trace

_KW = {
    "episode_horizon_s": 1800.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0,
    "noise_model": NoiseModel(0.0, 0.0), "hard_body_radius_m": 100.0,
}


@pytest.fixture(scope="module")
def trace():
    return record_episode(OrbitGuardEnv(num_agents=2, **_KW), no_maneuver_policy, seed=3, meta={"model": "no_maneuver", "condition": "normal"})


def test_app_builds_with_the_required_views(trace):
    app = create_app(trace)

    layout = str(app.layout)
    for view in ("constellation", "step-slider", "pair-panel", "inspector", "event-timeline", "metrics"):
        assert view in layout, view


def test_render_step_returns_every_panel_for_a_step_and_satellite(trace):
    outputs = render_step(trace, step=trace.num_steps, satellite=0)

    assert set(outputs) == {"constellation", "pairs", "inspector", "timeline", "status", "events"}
    assert outputs["constellation"].data
    assert "COLLISION" in str(outputs["pairs"])
    assert "t =" in outputs["status"]


def test_render_step_clamps_out_of_range_steps(trace):
    assert render_step(trace, step=10_000, satellite=0)["status"] == render_step(trace, step=trace.num_steps, satellite=0)["status"]


def test_replay_app_runs_standalone_from_a_saved_file(tmp_path, trace):
    path = tmp_path / "episode.npz"
    save_trace(trace, path)
    code = (
        "import sys;"
        "from orbit_guard.visualization.replay import load_trace;"
        "from orbit_guard.visualization.app import create_app;"
        f"app = create_app(load_trace(r'{path}'));"
        "assert 'ray' not in sys.modules and 'torch' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
