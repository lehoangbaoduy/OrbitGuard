"""Tests for orbit_guard.rl.conjunction_assessment (Phase 3 Part A).

Exercises the belief-based prediction pipeline end-to-end: predict_belief ->
TCA -> risk proxy -> adaptive margin -> NeighborCandidate construction. Uses
a zero-noise NoiseModel so belief equals ground truth exactly, letting the
predicted d_min_m be checked directly against the injected scenario's own
achieved_d_min_m (already self-verified at injection time by
scenarios.conjunction_generator.inject_conjunction) rather than against a
separately-rederived expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.physics.frames import eci_vector_to_rtn
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.rl.conjunction_assessment import assess_all_pairs, build_neighbor_candidates
from orbit_guard.scenarios.constellation import ShellDistribution
from orbit_guard.scenarios.small_n_scenario import generate_small_n_conjunction_scenario
from orbit_guard.sensing.belief import GroundTruthState, form_belief, observe
from orbit_guard.sensing.noise import NoiseModel

_CAPABILITY = SatelliteCapability(
    max_delta_v_per_action_mps=0.1, remaining_delta_v_mps=2.0,
    maneuver_enabled=True, mission_deviation_limit=1.0,
)
_SHELL = ShellDistribution(
    altitude_km_mean=463.0, altitude_km_std=1.5, altitude_km_min=460.0, altitude_km_max=465.0,
    inclination_deg_mean=53.06, inclination_deg_std=0.03, inclination_deg_min=53.04, inclination_deg_max=53.17,
    eccentricity_mean=0.0002, eccentricity_std=0.00005, eccentricity_min=0.0001, eccentricity_max=0.0004,
)
_ZERO_NOISE = NoiseModel(position_sigma_m=0.0, velocity_sigma_mps=0.0)
_EPISODE_HORIZON_S = 3600.0
_PHYSICS_DT_S = 10.0


def _zero_noise_beliefs(ground_truth_vectors):
    rng = np.random.default_rng(0)
    return [form_belief(observe(GroundTruthState(gt), rng, _ZERO_NOISE), _ZERO_NOISE) for gt in ground_truth_vectors]


def _find_single_conjunction_scenario():
    for seed in range(50):
        scenario = generate_small_n_conjunction_scenario(
            2, seed=seed, shell=_SHELL, capability_template=_CAPABILITY,
            episode_horizon_s=_EPISODE_HORIZON_S, physics_dt_s=_PHYSICS_DT_S,
        )
        if scenario.episode_type == "single_conjunction":
            return scenario
    pytest.fail("no single_conjunction scenario found in 50 seeds")


def test_assess_all_pairs_predicts_the_injected_conjunctions_d_min():
    scenario = _find_single_conjunction_scenario()
    spec = scenario.injected_conjunctions[0]
    ground_truth = [sat.state.as_vector() for sat in scenario.satellites]
    beliefs = _zero_noise_beliefs(ground_truth)

    lookahead_s = spec.achieved_tca_s + _PHYSICS_DT_S
    assessments = assess_all_pairs(beliefs, lookahead_s, _PHYSICS_DT_S)

    assert len(assessments) == 1
    assessment = assessments[0]
    assert {assessment.i, assessment.j} == {0, 1}
    tolerance_m = max(10.0, 0.15 * spec.target_d_min_m)
    assert abs(assessment.d_min_m - spec.achieved_d_min_m) <= tolerance_m


def test_each_agent_gets_exactly_n_minus_one_candidates():
    scenario = _find_single_conjunction_scenario()
    ground_truth = [sat.state.as_vector() for sat in scenario.satellites]
    beliefs = _zero_noise_beliefs(ground_truth)
    assessments = assess_all_pairs(beliefs, lookahead_s=600.0, physics_dt_s=_PHYSICS_DT_S)

    for idx in range(len(beliefs)):
        candidates = build_neighbor_candidates(idx, beliefs, assessments)
        assert len(candidates) == len(beliefs) - 1


def test_pair_values_are_symmetric_between_the_two_agents():
    scenario = _find_single_conjunction_scenario()
    ground_truth = [sat.state.as_vector() for sat in scenario.satellites]
    beliefs = _zero_noise_beliefs(ground_truth)
    assessments = assess_all_pairs(beliefs, lookahead_s=600.0, physics_dt_s=_PHYSICS_DT_S)

    candidates_0 = build_neighbor_candidates(0, beliefs, assessments)
    candidates_1 = build_neighbor_candidates(1, beliefs, assessments)
    assert candidates_0[0].d_min_m == pytest.approx(candidates_1[0].d_min_m)
    assert candidates_0[0].risk_proxy == pytest.approx(candidates_1[0].risk_proxy)
    assert candidates_0[0].d_safe_m == pytest.approx(candidates_1[0].d_safe_m)
    assert candidates_0[0].uncertainty_summary_m == pytest.approx(candidates_1[0].uncertainty_summary_m)


def test_relative_geometry_matches_the_observing_agents_own_rtn_frame():
    scenario = _find_single_conjunction_scenario()
    ground_truth = [sat.state.as_vector() for sat in scenario.satellites]
    beliefs = _zero_noise_beliefs(ground_truth)
    assessments = assess_all_pairs(beliefs, lookahead_s=600.0, physics_dt_s=_PHYSICS_DT_S)

    candidates_0 = build_neighbor_candidates(0, beliefs, assessments)
    own_pos, own_vel = beliefs[0].state_vector[:3], beliefs[0].state_vector[3:6]
    other_pos, other_vel = beliefs[1].state_vector[:3], beliefs[1].state_vector[3:6]
    expected_rel_pos_rtn = eci_vector_to_rtn(other_pos - own_pos, own_pos, own_vel)
    expected_rel_vel_rtn = eci_vector_to_rtn(other_vel - own_vel, own_pos, own_vel)
    np.testing.assert_allclose(candidates_0[0].relative_position_m, expected_rel_pos_rtn)
    np.testing.assert_allclose(candidates_0[0].relative_velocity_mps, expected_rel_vel_rtn)
