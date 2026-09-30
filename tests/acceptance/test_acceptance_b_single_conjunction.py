"""Acceptance Test B (PRD §38): a known injected conjunction is detected
and produces a meaningful risk signal.

Part A (scenarios/conjunction_generator, sensing/belief) and Part B
(conjunction/screening, tca, collision, risk) were implemented and reviewed
independently by two different agents working from a frozen interface
contract. Neither half's own test suite exercises the other's code -- this
is the first test that runs them together, and is Phase 2's actual exit
criterion for Test B, not just a rerun of either half's unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orbit_guard.conjunction.collision import detect_collision
from orbit_guard.conjunction.risk import risk_proxy
from orbit_guard.conjunction.screening import generate_candidate_pairs
from orbit_guard.conjunction.tca import compute_tca
from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.scenarios.conjunction_generator import (
    generate_conjunction_scenario,
    sample_episode_type,
)
from orbit_guard.scenarios.constellation import (
    fit_shell_distribution,
    load_tle_snapshot,
)
from orbit_guard.sensing.belief import (
    GroundTruthState,
    combined_relative_sigma,
    form_belief,
    observe,
)
from orbit_guard.sensing.noise import NoiseModel

_TLE_PATH = Path(__file__).resolve().parents[2] / "data" / "TLE_snapshot_53deg_shell.csv"
_PHYSICS_DT_S = 10.0
_EPISODE_HORIZON_S = 21600.0
_HARD_BODY_RADIUS_M = 5.0
_N_SATELLITES = 15
_NOISE_MODEL = NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1)


def _capability() -> SatelliteCapability:
    return SatelliteCapability(
        max_delta_v_per_action_mps=0.1,
        remaining_delta_v_mps=2.0,
        maneuver_enabled=True,
        mission_deviation_limit=1.0,
    )


def _find_single_conjunction_seed(search_range: int = 200) -> int:
    for seed in range(search_range):
        if sample_episode_type(np.random.default_rng(seed)) == "single_conjunction":
            return seed
    raise AssertionError(f"No seed in range({search_range}) produced a single_conjunction episode")


def _belief_position_covariance_sigma(state_at_t0: np.ndarray, rng: np.random.Generator):
    truth = GroundTruthState(state_vector=state_at_t0)
    return form_belief(observe(truth, rng, _NOISE_MODEL), _NOISE_MODEL)


def test_acceptance_b_injected_conjunction_is_detected_with_meaningful_risk():
    shell = fit_shell_distribution(load_tle_snapshot(_TLE_PATH))
    capability = _capability()
    seed = _find_single_conjunction_seed()

    scenario = generate_conjunction_scenario(
        _N_SATELLITES, seed, shell, capability,
        episode_horizon_s=_EPISODE_HORIZON_S, physics_dt_s=_PHYSICS_DT_S,
    )
    assert scenario.episode_type == "single_conjunction"
    assert len(scenario.injected_conjunctions) == 1
    injected = scenario.injected_conjunctions[0]

    n_steps = int(_EPISODE_HORIZON_S / _PHYSICS_DT_S)
    trajectories = [
        propagate(sat.state.as_vector(), _PHYSICS_DT_S, n_steps, state_derivative)
        for sat in scenario.satellites
    ]

    # 1. Screening (Part B) must surface the injected pair as a candidate.
    pairs = generate_candidate_pairs(_N_SATELLITES)
    injected_pair = tuple(sorted((injected.primary_index, injected.secondary_index)))
    assert injected_pair in pairs
    i, j = injected_pair

    # 2. Part B's TCA, computed independently of Part A's own self-check
    # during injection, must agree with the achieved geometry.
    t_tca_s, d_min_m = compute_tca(trajectories[i], trajectories[j], _PHYSICS_DT_S)
    assert t_tca_s == pytest.approx(injected.achieved_tca_s, abs=_PHYSICS_DT_S)
    assert d_min_m == pytest.approx(injected.achieved_d_min_m, rel=0.05, abs=5.0)

    # 3. Collision detection (Part B) runs cleanly against the injected pair
    # and agrees with Part B's own TCA on the minimum separation.
    collision_result = detect_collision(
        trajectories[i], trajectories[j], _PHYSICS_DT_S,
        hard_body_radius_i_m=_HARD_BODY_RADIUS_M, hard_body_radius_j_m=_HARD_BODY_RADIUS_M,
    )
    assert collision_result.min_separation_m == pytest.approx(d_min_m, rel=0.01)

    # 4. Risk proxy (Part B) fed sigma_rel_m from the real belief pipeline
    # (Part A) must show a MEANINGFULLY ELEVATED risk for the injected pair
    # versus a benign, naturally-separated pair -- the actual §38 requirement.
    rng = np.random.default_rng(seed)
    belief_i = _belief_position_covariance_sigma(trajectories[i][0], rng)
    belief_j = _belief_position_covariance_sigma(trajectories[j][0], rng)
    sigma_rel_m = combined_relative_sigma(belief_i, belief_j)
    injected_risk = risk_proxy(d_min_m, sigma_rel_m)

    benign_indices = [idx for idx in range(_N_SATELLITES) if idx not in injected_pair][:2]
    bi, bj = benign_indices
    _, benign_d_min_m = compute_tca(trajectories[bi], trajectories[bj], _PHYSICS_DT_S)
    belief_bi = _belief_position_covariance_sigma(trajectories[bi][0], rng)
    belief_bj = _belief_position_covariance_sigma(trajectories[bj][0], rng)
    benign_sigma_rel_m = combined_relative_sigma(belief_bi, belief_bj)
    benign_risk = risk_proxy(benign_d_min_m, benign_sigma_rel_m)

    assert injected_risk > benign_risk
    assert injected_risk > 0.5, (
        f"injected conjunction (d_min={d_min_m:.1f}m, sigma_rel={sigma_rel_m:.1f}m) "
        f"produced risk={injected_risk:.4f}, not a meaningful risk signal"
    )
    assert benign_risk < 0.1, (
        f"naturally-separated benign pair unexpectedly flagged risky: "
        f"d_min={benign_d_min_m:.1f}m, risk={benign_risk:.4f}"
    )
