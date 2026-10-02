"""Tests for orbit_guard.visualization.replay: produce -> save -> load -> replay, headlessly (P6-M5, FR16)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from orbit_guard.evaluation.baselines import rule_based_policy
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.evaluation import no_maneuver_policy
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel
from orbit_guard.visualization.replay import (
    EpisodeTrace,
    frame_at,
    load_trace,
    record_episode,
    save_trace,
)

_KW = {
    "episode_horizon_s": 1800.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0,
    "noise_model": NoiseModel(0.0, 0.0), "hard_body_radius_m": 100.0,
}
_COLLIDING_SEED = 3


@pytest.fixture(scope="module")
def collision_trace() -> EpisodeTrace:
    env = OrbitGuardEnv(num_agents=2, **_KW)
    return record_episode(env, no_maneuver_policy, seed=_COLLIDING_SEED, meta={"model": "no_maneuver"})


@pytest.fixture(scope="module")
def shielded_trace() -> EpisodeTrace:
    env = ShieldedOrbitGuardEnv(num_agents=2, **_KW)
    return record_episode(env, no_maneuver_policy, seed=_COLLIDING_SEED, meta={"model": "adaptive_mappo"})


def test_trace_has_consistent_shapes(collision_trace):
    trace = collision_trace
    steps, agents, pairs = trace.num_steps, trace.num_agents, trace.num_pairs

    assert agents == 2 and pairs == 1
    assert trace.positions_m.shape == (steps + 1, agents, 3)
    assert trace.alive.shape == (steps + 1, agents)
    assert trace.proposed.shape == trace.approved.shape == trace.applied_dv_mps.shape == (steps, agents)
    assert trace.d_min_m.shape == trace.d_safe_m.shape == trace.risk.shape == (steps + 1, pairs)
    assert trace.collided_pair.shape == (steps, pairs)
    assert trace.remaining_dv_mps.shape == trace.mission_deviation.shape == (steps + 1, agents)


def test_a_collision_is_visible_in_the_trace(collision_trace):
    trace = collision_trace

    assert trace.collided_pair.any()
    first_hit = int(np.argmax(trace.collided_pair[:, 0]))
    assert trace.alive[0].all()
    assert not trace.alive[first_hit + 1].any()  # both satellites terminated at the collision step
    assert trace.summary["collision_count"] == 1


def test_shield_interventions_are_visible_in_the_trace(shielded_trace):
    trace = shielded_trace

    assert trace.intervened.any()
    differs = trace.proposed != trace.approved
    np.testing.assert_array_equal(trace.intervened & trace.alive[:-1], differs & trace.alive[:-1])
    assert trace.applied_dv_mps.sum() > 0
    assert not trace.collided_pair.any()


def test_the_trace_round_trips_through_disk(tmp_path, shielded_trace):
    path = tmp_path / "episode.npz"

    save_trace(shielded_trace, path)
    loaded = load_trace(path)

    np.testing.assert_array_equal(loaded.positions_m, shielded_trace.positions_m)
    np.testing.assert_array_equal(loaded.proposed, shielded_trace.proposed)
    np.testing.assert_array_equal(loaded.intervened, shielded_trace.intervened)
    assert loaded.meta == shielded_trace.meta
    assert loaded.summary == shielded_trace.summary


def test_frame_at_reports_status_per_satellite_and_pair(collision_trace):
    trace = collision_trace
    last = trace.num_steps

    start = frame_at(trace, 0)
    end = frame_at(trace, last)

    assert start.time_s == 0.0
    assert len(start.satellites) == 2
    assert all(sat.status == "nominal" or sat.status == "inside_d_safe" or sat.status == "high_risk" for sat in start.satellites)
    assert any(sat.status == "collided" for sat in end.satellites)
    assert end.pairs[0].status in {"collision", "inside_d_safe", "high_risk", "nominal"}


def test_frame_status_vocabulary_separates_collision_risk_margin_and_events(shielded_trace):
    seen_pair, seen_sat = set(), set()
    for t in range(shielded_trace.num_steps + 1):
        frame = frame_at(shielded_trace, t)
        seen_pair |= {pair.status for pair in frame.pairs}
        seen_sat |= {sat.event for sat in frame.satellites}

    assert "shield_intervention" in seen_sat
    assert seen_pair <= {"nominal", "high_risk", "inside_d_safe", "collision"}


def test_frame_index_is_validated(collision_trace):
    with pytest.raises(IndexError):
        frame_at(collision_trace, collision_trace.num_steps + 5)


def test_replay_needs_neither_ray_nor_torch():
    code = (
        "import sys, orbit_guard.visualization.replay;"
        "assert 'ray' not in sys.modules and 'torch' not in sys.modules, 'replay pulled in a trainer dependency'"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_recording_works_with_a_heuristic_policy_on_a_larger_constellation():
    env = OrbitGuardEnv(num_agents=10, episode_horizon_s=300.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=300.0)

    trace = record_episode(env, rule_based_policy, seed=1, meta={"model": "rule_based"})

    assert trace.num_agents == 10
    assert trace.num_pairs == 45


_REPLAY_DATA_LAYER = (
    "orbit_guard.visualization.replay_meta",
    "orbit_guard.visualization.episodes",
    "orbit_guard.visualization.summary",
    "orbit_guard.visualization.source",
    "orbit_guard.visualization.payload",
    "orbit_guard.visualization.reference",
    "orbit_guard.visualization.derived.status",
    "orbit_guard.visualization.derived.fuel",
    "orbit_guard.visualization.derived.metrics",
    "orbit_guard.safety.maneuver_mapper",
)


@pytest.mark.parametrize("module", _REPLAY_DATA_LAYER)
def test_replay_data_layer_needs_neither_ray_nor_torch(module):
    """FR16: a saved episode replays with training and inference stopped, including the redesign's data layer."""
    code = (
        f"import sys, {module};"
        "bad = [name for name in ('ray', 'torch', 'gymnasium') if name in sys.modules];"
        "assert not bad, f'replay data layer pulled in {bad}'"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
