"""Contract tests for orbit_guard.safety.maneuver_mapper (PRD §23).

Six discrete maneuver intents map deterministically to an RTN-frame Δv,
converted to OG-ECI via the already-frozen Phase 1 frame transform
(physics.frames.rtn_vector_to_eci). This mapping is fixed in one module and
must never be reinterpreted elsewhere in the codebase (PRD §23).

| # | Intent            | RTN direction         | Magnitude               |
|---|-------------------|------------------------|--------------------------|
| 0 | NO_MANEUVER       | --                     | 0                        |
| 1 | RAISE_ORBIT       | +T (prograde)          | base_delta_v             |
| 2 | LOWER_ORBIT       | -T (retrograde)        | base_delta_v             |
| 3 | PHASE_FORWARD     | -T (retrograde, fine)  | phase_fraction*base_dv   |
| 4 | PHASE_BACKWARD    | +T (prograde, fine)    | phase_fraction*base_dv   |
| 5 | RADIAL_ADJUSTMENT | +R (radially outward)  | base_delta_v             |

Physical rationale (PRD §23 design-decision log): a prograde (+T) burn raises
semi-major axis, so RAISE_ORBIT/PHASE_BACKWARD (+T) must INCREASE specific
orbital energy; LOWER_ORBIT/PHASE_FORWARD (-T) must DECREASE it. This is
checked independently below via vis-viva energy, not by round-tripping
through the same RTN basis the implementation itself will use for that would
be tautological -- and via an actual multi-orbit drift simulation for
PHASE_FORWARD, both empirically pre-verified against Phase 1 physics before
this spec was frozen (see docs/OrbitGuard_progress.md Phase 3 section).

Capability checks (remaining Δv budget, maneuver_enabled) are NOT this
module's job -- it is a pure geometric mapping; the environment enforces
capability limits before/after calling it.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v

from orbit_guard.physics.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.frames import coe_to_cartesian, eci_vector_to_rtn
from orbit_guard.physics.integrator import propagate

_BASE_DELTA_V_MPS = 0.1
_PHASE_FRACTION = 0.5


def _leo_state(true_anomaly_rad: float = 0.0):
    return coe_to_cartesian(
        R_EARTH_EQUATORIAL_M + 463_000.0, 0.0002, np.radians(53.06),
        np.radians(10.0), np.radians(20.0), true_anomaly_rad,
    )


def _specific_energy(position_m: np.ndarray, velocity_mps: np.ndarray) -> float:
    r = np.linalg.norm(position_m)
    v = np.linalg.norm(velocity_mps)
    return v**2 / 2.0 - MU_EARTH_M3_S2 / r


# --- Basic bookkeeping (via the frame transform) ----------------------------

def test_no_maneuver_is_exactly_zero():
    r, v = _leo_state()
    delta_v = map_intent_to_delta_v(ManeuverIntent.NO_MANEUVER, r, v)
    assert np.array_equal(delta_v, np.zeros(3))


@pytest.mark.parametrize("intent,rtn_index,sign,magnitude", [
    (ManeuverIntent.RAISE_ORBIT, 1, +1, _BASE_DELTA_V_MPS),
    (ManeuverIntent.LOWER_ORBIT, 1, -1, _BASE_DELTA_V_MPS),
    (ManeuverIntent.PHASE_FORWARD, 1, -1, _PHASE_FRACTION * _BASE_DELTA_V_MPS),
    (ManeuverIntent.PHASE_BACKWARD, 1, +1, _PHASE_FRACTION * _BASE_DELTA_V_MPS),
    (ManeuverIntent.RADIAL_ADJUSTMENT, 0, +1, _BASE_DELTA_V_MPS),
])
def test_maneuver_matches_frozen_rtn_table(intent, rtn_index, sign, magnitude):
    r, v = _leo_state()
    delta_v_eci = map_intent_to_delta_v(
        intent, r, v, base_delta_v_mps=_BASE_DELTA_V_MPS, phase_fraction=_PHASE_FRACTION
    )
    delta_v_rtn = eci_vector_to_rtn(delta_v_eci, r, v)

    other_indices = [idx for idx in range(3) if idx != rtn_index]
    assert np.allclose(delta_v_rtn[other_indices], 0.0, atol=1e-9)
    assert delta_v_rtn[rtn_index] == pytest.approx(sign * magnitude)


def test_maneuver_magnitude_is_position_invariant_but_direction_is_not():
    r1, v1 = _leo_state(true_anomaly_rad=0.0)
    r2, v2 = _leo_state(true_anomaly_rad=np.radians(90.0))
    dv1 = map_intent_to_delta_v(ManeuverIntent.RAISE_ORBIT, r1, v1)
    dv2 = map_intent_to_delta_v(ManeuverIntent.RAISE_ORBIT, r2, v2)
    assert np.linalg.norm(dv1) == pytest.approx(np.linalg.norm(dv2))
    assert not np.allclose(dv1, dv2)


# --- Independent physics verification (NOT via eci_vector_to_rtn) ----------

@pytest.mark.parametrize("intent,expect_energy_increase", [
    (ManeuverIntent.RAISE_ORBIT, True),
    (ManeuverIntent.LOWER_ORBIT, False),
    (ManeuverIntent.PHASE_FORWARD, False),
    (ManeuverIntent.PHASE_BACKWARD, True),
])
def test_tangential_maneuvers_change_specific_energy_in_the_named_direction(intent, expect_energy_increase):
    r, v = _leo_state()
    e0 = _specific_energy(r, v)
    delta_v = map_intent_to_delta_v(intent, r, v)
    e1 = _specific_energy(r, v + delta_v)
    assert (e1 > e0) if expect_energy_increase else (e1 < e0)


def test_radial_adjustment_energy_change_is_much_smaller_than_tangential():
    r, v = _leo_state()
    e0 = _specific_energy(r, v)
    dv_radial = map_intent_to_delta_v(ManeuverIntent.RADIAL_ADJUSTMENT, r, v)
    dv_tangential = map_intent_to_delta_v(ManeuverIntent.RAISE_ORBIT, r, v)
    d_energy_radial = abs(_specific_energy(r, v + dv_radial) - e0)
    d_energy_tangential = abs(_specific_energy(r, v + dv_tangential) - e0)
    assert d_energy_radial < 0.01 * d_energy_tangential


def test_phase_forward_drifts_the_satellite_ahead_of_an_unmaneuvered_twin():
    # The literal, named-behavior demonstration of the frozen RTN sign
    # convention: PHASE_FORWARD's retrograde burn lowers the orbit slightly,
    # shortening the period, so the maneuvered satellite ends up AHEAD
    # (positive along-track offset) of an unmaneuvered twin after several
    # orbits -- empirically confirmed (+4216m after 5 orbits) before this
    # test was written; do not loosen this into a directionless check.
    r0, v0 = _leo_state()
    state0 = np.concatenate([r0, v0])
    delta_v = map_intent_to_delta_v(ManeuverIntent.PHASE_FORWARD, r0, v0)
    maneuvered_state0 = np.concatenate([r0, v0 + delta_v])

    a = R_EARTH_EQUATORIAL_M + 463_000.0
    period_s = 2 * np.pi * np.sqrt(a**3 / MU_EARTH_M3_S2)
    dt = 10.0
    n_steps = int(5 * period_s / dt)

    twin_traj = propagate(state0, dt, n_steps, state_derivative)
    maneuvered_traj = propagate(maneuvered_state0, dt, n_steps, state_derivative)

    r_twin_final, v_twin_final = twin_traj[-1, :3], twin_traj[-1, 3:]
    t_hat = v_twin_final / np.linalg.norm(v_twin_final)
    along_track_offset_m = float(np.dot(maneuvered_traj[-1, :3] - r_twin_final, t_hat))

    assert along_track_offset_m > 1000.0, (
        f"PHASE_FORWARD should drift the satellite ahead of an unmaneuvered twin "
        f"after 5 orbits; got along-track offset {along_track_offset_m:.1f}m"
    )


# --- Input validation & scope boundaries ------------------------------------

def test_rejects_invalid_intent():
    r, v = _leo_state()
    with pytest.raises(ValueError):
        map_intent_to_delta_v(6, r, v)
    with pytest.raises(ValueError):
        map_intent_to_delta_v(-1, r, v)


def test_rejects_negative_base_delta_v():
    r, v = _leo_state()
    with pytest.raises(ValueError):
        map_intent_to_delta_v(ManeuverIntent.RAISE_ORBIT, r, v, base_delta_v_mps=-0.1)


def test_is_deterministic():
    r, v = _leo_state()
    dv1 = map_intent_to_delta_v(ManeuverIntent.RAISE_ORBIT, r, v)
    dv2 = map_intent_to_delta_v(ManeuverIntent.RAISE_ORBIT, r, v)
    assert np.array_equal(dv1, dv2)


def test_mapper_signature_has_no_capability_parameters():
    params = set(inspect.signature(map_intent_to_delta_v).parameters)
    forbidden = {"remaining_delta_v_mps", "maneuver_enabled", "capability"}
    assert not (params & forbidden)


# --- Architectural import boundary -------------------------------------------

def test_safety_package_never_imports_rl_or_training_frameworks():
    # Checklist item #33 (docs/OrbitGuard_progress.md §8): physics/safety
    # modules never import RL -- extended here to the specific frameworks
    # Phase 3 introduces (gymnasium/pettingzoo/ray/torch).
    forbidden_modules = {"orbit_guard.rl", "gymnasium", "pettingzoo", "ray", "torch"}
    safety_dir = Path(__file__).resolve().parents[2] / "orbit_guard" / "safety"
    for py_file in safety_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module}
            else:
                continue
            for forbidden in forbidden_modules:
                hit = {name for name in names if name == forbidden or name.startswith(forbidden + ".")}
                assert not hit, f"{py_file.name} imports {hit}, violating the physics/safety-never-imports-RL rule"
