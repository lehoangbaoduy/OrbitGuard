"""Contract tests for orbit_guard.rl.observation (PRD §24).

Fixed-size, per-agent, DECENTRALIZED observation for the actor network,
returned as a dict of plain numpy float64 arrays (this module owns no
gymnasium.spaces object and never imports gymnasium/pettingzoo -- the
ENVIRONMENT, Part A, wraps this function's output into the official
Gymnasium Dict space, and is also responsible for casting float64 -> the
space's float32 dtype at that wrapping boundary; that keeps rl/observation.py
importable, testable, and framework-free):

    {
      "own":            (10,)   [pos_norm(3), vel_norm(3), remaining_dv_norm(1),
                                  mission_deviation(1), max_dv_norm(1),
                                  maneuver_enabled(1)]
      "neighbors":      (k,10)  per slot: [rel_pos_norm(3), rel_vel_norm(3),
                                  risk_proxy(1), d_safe_norm(1),
                                  uncertainty_norm(1), neighbor_valid(1)]
      "global_summary": (4,)    [threatening_neighbor_count(1), max_risk(1),
                                  total_risk(1), min_separation_norm(1)]
    }

`threatening_neighbor_count` is a RAW COUNT (PRD §24's literal field name
and semantics), not a fraction -- a fraction over a variable-size candidate
pool would make "1 threat among 1 candidate" read identically to "1 threat
among 20", which is not what this field means. It is intentionally
unnormalized/unbounded, unlike every other field here; the network's input
normalization layer (Part C's concern, not this module's) may rescale it.

Neighbor ranking is by predicted THREAT (risk_proxy), never raw distance
and never by satellite index -- ranking by index would leak persistent
identity and violate §24's anonymity requirement. risk_proxy correctly
underflows to exactly 0.0 for well-separated pairs (empirically confirmed
during the Phase 2 PR review's Acceptance Test B), so most naturally
sampled candidates TIE at risk_proxy=0.0. The frozen tie-break is
ascending d_min_m (closer-but-currently-benign ranked above
farther-but-currently-benign), never input order or satellite index.

The field is named `neighbor_valid`, NOT `mask` -- deliberately, so it is
never confused with PRD §27's action masking, which this project's main
experiments do not use.

`NeighborCandidate.relative_position_m`/`relative_velocity_mps` are in the
OWNING agent's own RTN frame (PRD §23's already-frozen local frame, via
`physics.frames.eci_vector_to_rtn`), NOT raw ECI differences -- computed
upstream by Part A's environment, which has both satellites' ECI states.
Expressing neighbor geometry in the observer's own local frame (rather than
raw ECI, which would vary with the observer's own orbital phase/plane for
an otherwise-identical encounter) is what makes the shared, parameter-tied
policy actually generalize across constellation instances, per §24's
anonymity/generalization intent. This module performs no frame transform
itself and must not import `orbit_guard.physics.frames`.

`own_mission_deviation` arrives ALREADY normalized: it is the scalar
`D_mission = q_a*D_a + q_e*D_e + q_i*D_i + q_M*D_M` (PRD §25's own formula,
computed by Part A's environment from `physics/mission_deviation.py`'s four
raw components, using the SAME frozen q-weights as `rl.reward`'s
`compute_mission_term`). Numerically, `own_mission_deviation` must equal
`-compute_mission_term(d_a, d_e, d_i, d_m)` for the same satellite at the
same instant -- Part A's environment is the only place that computes both,
and must derive them from the identical four components so they agree by
construction. This module passes the scalar through unscaled and must not
re-derive or re-normalize it.

Normalization constants below are frozen contract, provisional pending
real review (Q11, docs/OrbitGuard_progress.md) -- any other Phase 3 module
needing the same scale MUST import it from here, never redefine it.
"""

from __future__ import annotations

import numpy as np
import pytest
from orbit_guard.rl.observation import (
    DELTA_V_BUDGET_SCALE_MPS,
    MAX_DELTA_V_PER_ACTION_SCALE_MPS,
    MINIMUM_SEPARATION_SCALE_M,
    POSITION_SCALE_M,
    RELATIVE_POSITION_SCALE_M,
    RELATIVE_VELOCITY_SCALE_MPS,
    THREATENING_RISK_THRESHOLD,
    VELOCITY_SCALE_MPS,
    NeighborCandidate,
    build_observation,
)

from orbit_guard.physics.constants import MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M

_K_NEIGHBORS = 4


def _own_kwargs(**overrides):
    base = {
        "own_position_m": np.array([POSITION_SCALE_M, 0.0, 0.0]),
        "own_velocity_mps": np.array([0.0, VELOCITY_SCALE_MPS, 0.0]),
        "own_remaining_delta_v_mps": 1.0,
        "own_mission_deviation": 0.2,
        "own_max_delta_v_per_action_mps": 0.1,
        "own_maneuver_enabled": True,
    }
    base.update(overrides)
    return base


def _candidate(
    rel_pos=(1000.0, 0.0, 0.0),
    rel_vel=(1.0, 0.0, 0.0),
    d_min=1000.0,
    risk=0.1,
    d_safe=1000.0,
    unc=100.0,
):
    return NeighborCandidate(
        relative_position_m=np.array(rel_pos),
        relative_velocity_mps=np.array(rel_vel),
        d_min_m=d_min,
        risk_proxy=risk,
        d_safe_m=d_safe,
        uncertainty_summary_m=unc,
    )


# --- Shape / key contract ----------------------------------------------------

def test_returns_exactly_the_three_expected_keys_with_correct_shapes():
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[_candidate()], k_neighbors=_K_NEIGHBORS)
    assert set(obs.keys()) == {"own", "neighbors", "global_summary"}
    assert obs["own"].shape == (10,)
    assert obs["neighbors"].shape == (_K_NEIGHBORS, 10)
    assert obs["global_summary"].shape == (4,)
    assert all(np.all(np.isfinite(v)) for v in obs.values())


# --- Normalization -----------------------------------------------------------

def test_normalization_constants_are_pinned_to_their_frozen_numeric_values():
    # A test that only checks self-consistency (obs[i] == 1.0 when the input
    # IS the constant) would pass even if every scale were silently redefined
    # to 1.0 -- this pins the actual frozen numbers so that can't happen.
    assert POSITION_SCALE_M == pytest.approx(R_EARTH_EQUATORIAL_M)
    assert VELOCITY_SCALE_MPS == pytest.approx(np.sqrt(MU_EARTH_M3_S2 / R_EARTH_EQUATORIAL_M))
    assert RELATIVE_POSITION_SCALE_M == pytest.approx(5000.0)  # PRD §22 d_max_m
    assert RELATIVE_VELOCITY_SCALE_MPS == pytest.approx(1000.0)
    assert DELTA_V_BUDGET_SCALE_MPS == pytest.approx(2.0)  # configs/default.yaml capability.remaining_delta_v_mps
    assert MAX_DELTA_V_PER_ACTION_SCALE_MPS == pytest.approx(0.1)  # configs/default.yaml maneuver.base_delta_v_mps
    assert MINIMUM_SEPARATION_SCALE_M == pytest.approx(5000.0)


def test_own_position_and_velocity_are_normalized_to_order_one():
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[], k_neighbors=_K_NEIGHBORS)
    assert obs["own"][0] == pytest.approx(1.0)
    assert obs["own"][4] == pytest.approx(1.0)


def test_own_remaining_delta_v_is_normalized_by_the_frozen_budget_scale():
    obs = build_observation(
        **_own_kwargs(own_remaining_delta_v_mps=DELTA_V_BUDGET_SCALE_MPS),
        neighbor_candidates=[],
        k_neighbors=_K_NEIGHBORS,
    )
    assert obs["own"][6] == pytest.approx(1.0)


def test_own_max_delta_v_per_action_is_normalized_by_the_frozen_scale():
    obs = build_observation(
        **_own_kwargs(own_max_delta_v_per_action_mps=MAX_DELTA_V_PER_ACTION_SCALE_MPS),
        neighbor_candidates=[],
        k_neighbors=_K_NEIGHBORS,
    )
    assert obs["own"][8] == pytest.approx(1.0)


def test_own_mission_deviation_passes_through_unscaled():
    obs = build_observation(
        **_own_kwargs(own_mission_deviation=0.37), neighbor_candidates=[], k_neighbors=_K_NEIGHBORS
    )
    assert obs["own"][7] == pytest.approx(0.37)


def test_own_maneuver_enabled_is_encoded_as_zero_or_one():
    obs_true = build_observation(
        **_own_kwargs(own_maneuver_enabled=True), neighbor_candidates=[], k_neighbors=_K_NEIGHBORS
    )
    obs_false = build_observation(
        **_own_kwargs(own_maneuver_enabled=False), neighbor_candidates=[], k_neighbors=_K_NEIGHBORS
    )
    assert obs_true["own"][9] == 1.0
    assert obs_false["own"][9] == 0.0


def test_neighbor_relative_position_and_velocity_are_normalized():
    candidate = _candidate(
        rel_pos=(RELATIVE_POSITION_SCALE_M, 0.0, 0.0), rel_vel=(RELATIVE_VELOCITY_SCALE_MPS, 0.0, 0.0)
    )
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[candidate], k_neighbors=_K_NEIGHBORS)
    assert obs["neighbors"][0, 0] == pytest.approx(1.0)
    assert obs["neighbors"][0, 3] == pytest.approx(1.0)


def test_neighbor_d_safe_is_normalized_by_relative_position_scale():
    candidate = _candidate(d_safe=RELATIVE_POSITION_SCALE_M)
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[candidate], k_neighbors=_K_NEIGHBORS)
    assert obs["neighbors"][0, 7] == pytest.approx(1.0)


def test_neighbor_risk_proxy_passes_through_unscaled_since_already_bounded_zero_one():
    candidate = _candidate(risk=0.83)
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[candidate], k_neighbors=_K_NEIGHBORS)
    assert obs["neighbors"][0, 6] == pytest.approx(0.83)


def test_neighbor_uncertainty_summary_is_normalized_by_the_relative_position_scale():
    candidate = _candidate(unc=RELATIVE_POSITION_SCALE_M)
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[candidate], k_neighbors=_K_NEIGHBORS)
    assert obs["neighbors"][0, 8] == pytest.approx(1.0)


# --- Padding & neighbor_valid -------------------------------------------------

def test_fewer_than_k_neighbors_are_zero_padded_with_neighbor_valid_false():
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[_candidate()], k_neighbors=_K_NEIGHBORS)
    assert obs["neighbors"][0, 9] == 1.0
    for slot in range(1, _K_NEIGHBORS):
        assert obs["neighbors"][slot, 9] == 0.0
        assert np.all(obs["neighbors"][slot, :9] == 0.0)


def test_zero_neighbor_candidates_produces_all_padded_slots():
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[], k_neighbors=_K_NEIGHBORS)
    assert np.all(obs["neighbors"][:, 9] == 0.0)
    assert np.all(np.isfinite(obs["global_summary"]))


def test_more_than_k_candidates_are_truncated_to_the_k_highest_risk():
    candidates = [_candidate(risk=r, d_min=1000.0) for r in [0.9, 0.1, 0.5, 0.7, 0.3, 0.2]]
    obs = build_observation(**_own_kwargs(), neighbor_candidates=candidates, k_neighbors=_K_NEIGHBORS)
    selected_risks = sorted(obs["neighbors"][:, 6].tolist(), reverse=True)
    assert selected_risks == [0.9, 0.7, 0.5, 0.3]


# --- Ranking: by threat, never by distance or identity ----------------------

def test_neighbor_ranking_is_by_risk_not_distance():
    nearer_but_safer = _candidate(rel_pos=(10.0, 0.0, 0.0), d_min=10.0, risk=0.1)
    farther_but_riskier = _candidate(rel_pos=(4000.0, 0.0, 0.0), d_min=4000.0, risk=0.9)
    obs = build_observation(
        **_own_kwargs(), neighbor_candidates=[nearer_but_safer, farther_but_riskier], k_neighbors=1
    )
    assert obs["neighbors"][0, 6] == pytest.approx(0.9)


def test_neighbor_ranking_ties_on_risk_break_by_ascending_d_min_not_input_order():
    farther_tie = _candidate(d_min=500.0, risk=0.0, d_safe=111.0)
    closer_tie = _candidate(d_min=50.0, risk=0.0, d_safe=222.0)
    obs = build_observation(
        **_own_kwargs(), neighbor_candidates=[farther_tie, closer_tie], k_neighbors=1
    )
    # closer_tie (smaller d_min) must win the single slot despite being listed second.
    assert obs["neighbors"][0, 7] == pytest.approx(222.0 / RELATIVE_POSITION_SCALE_M)


# --- Global summary ------------------------------------------------------------

def test_global_summary_max_and_total_risk_are_computed_over_all_candidates_not_just_top_k():
    risks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    candidates = [_candidate(risk=r) for r in risks]
    obs = build_observation(**_own_kwargs(), neighbor_candidates=candidates, k_neighbors=4)
    assert obs["global_summary"][1] == pytest.approx(0.6)
    assert obs["global_summary"][2] == pytest.approx(sum(risks))


def test_global_summary_threatening_neighbor_count_is_a_raw_count_using_the_frozen_threshold():
    assert THREATENING_RISK_THRESHOLD == 0.5
    candidates = [_candidate(risk=r) for r in [0.9, 0.6, 0.4, 0.1]]  # 2 of 4 exceed 0.5
    obs = build_observation(**_own_kwargs(), neighbor_candidates=candidates, k_neighbors=4)
    assert obs["global_summary"][0] == pytest.approx(2.0)


def test_global_summary_threatening_neighbor_count_is_not_normalized_by_pool_size():
    # One threat among one candidate and one threat among twenty candidates
    # both count as "1 threatening neighbor" -- this field is a raw count,
    # not a fraction, so it must NOT shrink as the candidate pool grows.
    one_threat_of_one = [_candidate(risk=0.9)]
    one_threat_of_twenty = [_candidate(risk=0.9)] + [_candidate(risk=0.0) for _ in range(19)]
    obs_small_pool = build_observation(**_own_kwargs(), neighbor_candidates=one_threat_of_one, k_neighbors=4)
    obs_large_pool = build_observation(**_own_kwargs(), neighbor_candidates=one_threat_of_twenty, k_neighbors=4)
    assert obs_small_pool["global_summary"][0] == pytest.approx(1.0)
    assert obs_large_pool["global_summary"][0] == pytest.approx(1.0)


def test_global_summary_minimum_predicted_separation_over_all_candidates():
    candidates = [_candidate(d_min=d) for d in [500.0, 50.0, 2000.0]]
    obs = build_observation(**_own_kwargs(), neighbor_candidates=candidates, k_neighbors=4)
    assert obs["global_summary"][3] == pytest.approx(50.0 / MINIMUM_SEPARATION_SCALE_M)


def test_empty_candidates_global_summary_indicates_no_known_danger():
    obs = build_observation(**_own_kwargs(), neighbor_candidates=[], k_neighbors=4)
    assert obs["global_summary"][0] == 0.0
    assert obs["global_summary"][1] == 0.0
    assert obs["global_summary"][2] == 0.0
    assert obs["global_summary"][3] == pytest.approx(1.0)


# --- Validation & determinism --------------------------------------------------

def test_rejects_k_neighbors_less_than_one():
    with pytest.raises(ValueError):
        build_observation(**_own_kwargs(), neighbor_candidates=[], k_neighbors=0)


def test_is_deterministic():
    candidates = [_candidate(risk=0.5), _candidate(risk=0.3)]
    obs1 = build_observation(**_own_kwargs(), neighbor_candidates=candidates, k_neighbors=4)
    obs2 = build_observation(**_own_kwargs(), neighbor_candidates=candidates, k_neighbors=4)
    for key in obs1:
        assert np.array_equal(obs1[key], obs2[key])
