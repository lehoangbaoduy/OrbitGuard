"""Per-control-step records for the PRD §40 logging schema (CORE-4c).

`OrbitGuardEnv.step_records()` returns one record per satellite that was active when the last step's
actions were chosen. Oracles are the env's own public surface: the observations the previous
reset/step returned, `snapshot()` before and after the step, and the shield's `last_shield_result`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from orbit_guard.rl.actor_policy import flatten_local_observation
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.rl.step_records import STEP_RECORD_KEYS, max_risk_pair
from orbit_guard.safety.maneuver_mapper import ManeuverIntent
from orbit_guard.sensing.noise import NoiseModel

_KW = {"episode_horizon_s": 180.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 1800.0}
_CONJUNCTION_SEED = 10  # N=2: the shield rejects the all-no-op proposal on every step
_MIXED_SEED = 0  # N=2: the shield rejects the first all-no-op proposal and accepts the later ones
_BENIGN_SEED = 4  # N=2: the shield accepts every all-no-op proposal
_KEYS = {
    "timestamp", "satellite_id", "observed_state", "belief_features", "selected_action", "mapped_delta_v",
    "shield_action", "shield_intervened", "shield_reason", "risk_before", "risk_after", "d_safe_before",
    "d_safe_after", "remaining_delta_v", "mission_deviation",
}
_ENV_CLASSES = [OrbitGuardEnv, ShieldedOrbitGuardEnv]
_IDS = ["no_shield", "shielded"]


def _no_op(env) -> dict:
    return {agent: int(ManeuverIntent.NO_MANEUVER) for agent in env.agents}


def _max_risk_pair(pairs, index):
    """(risk, d_safe) of the highest-risk pair containing `index`; the first such pair on ties."""
    involving = [pair for pair in pairs if index in pair[:2]]
    _, _, _, d_safe, risk = max(involving, key=lambda pair: pair[4])
    return risk, d_safe


def _leaves(value):
    if isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


@pytest.mark.parametrize("env_class", _ENV_CLASSES, ids=_IDS)
def test_there_are_no_step_records_before_the_first_step(env_class):
    env = env_class(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)

    assert env.step_records() == []


@pytest.mark.parametrize("env_class", _ENV_CLASSES, ids=_IDS)
def test_each_active_satellite_gets_one_record_with_exactly_the_prd_section_40_keys(env_class):
    env = env_class(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)

    env.step(_no_op(env))
    records = env.step_records()

    assert [record["satellite_id"] for record in records] == ["sat_0", "sat_1"]
    for record in records:
        assert set(record) == _KEYS
        assert tuple(record) == STEP_RECORD_KEYS


@pytest.mark.parametrize("env_class", _ENV_CLASSES, ids=_IDS)
def test_every_record_value_is_a_plain_json_serializable_python_type(env_class):
    env = env_class(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)
    env.step({"sat_0": int(ManeuverIntent.RAISE_ORBIT), "sat_1": int(ManeuverIntent.NO_MANEUVER)})

    records = env.step_records()

    assert json.loads(json.dumps(records, allow_nan=False)) == records
    for record in records:
        for key, value in record.items():
            assert type(value) in (int, float, bool, str, list), key
            assert all(type(leaf) in (int, float, str) for leaf in _leaves(value) if leaf is not value), key
        assert type(record["timestamp"]) is float
        assert type(record["selected_action"]) is int
        assert type(record["shield_action"]) is int
        assert type(record["shield_intervened"]) is bool
        assert len(record["observed_state"]) == 6
        assert len(record["mapped_delta_v"]) == 3
        assert all(type(x) is float for x in record["observed_state"] + record["belief_features"] + record["mapped_delta_v"])
        assert all(type(x) is str for x in record["shield_reason"])
        for key in ("risk_before", "risk_after", "d_safe_before", "d_safe_after", "remaining_delta_v", "mission_deviation"):
            assert type(record[key]) is float, key


@pytest.mark.parametrize("env_class", _ENV_CLASSES, ids=_IDS)
def test_records_carry_the_decision_time_state_and_the_post_step_outcome(env_class):
    env = env_class(num_agents=2, **_KW)
    observations, _ = env.reset(seed=_MIXED_SEED)
    for step in range(3):
        before = env.snapshot()
        decision_beliefs = [belief.state_vector.copy() for belief in env._pending_beliefs]
        actions = _no_op(env)

        observations_after, *_ = env.step(actions)
        after = env.snapshot()

        for record in env.step_records():
            agent = record["satellite_id"]
            index = env.possible_agents.index(agent)
            assert record["timestamp"] == step * _KW["control_dt_s"]
            assert record["observed_state"] == decision_beliefs[index].tolist()
            assert record["belief_features"] == flatten_local_observation(observations[agent]).tolist()
            assert record["selected_action"] == actions[agent]
            assert (record["risk_before"], record["d_safe_before"]) == _max_risk_pair(before["pairs"], index)
            assert (record["risk_after"], record["d_safe_after"]) == _max_risk_pair(after["pairs"], index)
            assert record["remaining_delta_v"] == after["remaining_dv_mps"][index]
            assert record["mission_deviation"] == after["mission_deviation"][index]
        observations = observations_after


@pytest.mark.parametrize("env_class", _ENV_CLASSES, ids=_IDS)
def test_mapped_delta_v_is_the_raw_request_in_eci(env_class):
    env = env_class(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)
    burn_limit_mps = env._capabilities[0].max_delta_v_per_action_mps

    env.step({"sat_0": int(ManeuverIntent.RAISE_ORBIT), "sat_1": int(ManeuverIntent.NO_MANEUVER)})
    burner, idle = env.step_records()

    assert np.linalg.norm(burner["mapped_delta_v"]) == pytest.approx(burn_limit_mps)
    assert idle["mapped_delta_v"] == [0.0, 0.0, 0.0]


def test_without_a_shield_the_approved_action_is_the_selected_one_and_there_is_no_reason():
    env = OrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)
    rng = np.random.default_rng(0)
    while env.agents:
        env.step({agent: int(rng.integers(len(ManeuverIntent))) for agent in env.agents})
        for record in env.step_records():
            assert record["shield_action"] == record["selected_action"]
            assert record["shield_intervened"] is False
            assert record["shield_reason"] == []


def test_the_shield_record_is_the_approved_action_of_the_raw_no_op_proposal():
    env = ShieldedOrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)

    env.step(_no_op(env))
    records = env.step_records()
    result = env.last_shield_result

    assert all(record["selected_action"] == 0 and record["mapped_delta_v"] == [0.0, 0.0, 0.0] for record in records)
    assert any(record["shield_action"] != 0 for record in records)
    for index, record in enumerate(records):
        assert record["shield_action"] == result.approved[index]
        assert record["shield_intervened"] == result.intervened[index]


@pytest.mark.parametrize("seed", [_CONJUNCTION_SEED, _MIXED_SEED, _BENIGN_SEED])
def test_shield_reason_is_non_empty_exactly_when_the_shield_rejected_a_proposal_involving_the_satellite(seed):
    env = ShieldedOrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=seed)
    rejected_steps = 0
    while env.agents:
        env.step(_no_op(env))
        result = env.last_shield_result
        rejected_steps += not result.accepted
        for index, record in enumerate(env.step_records()):
            involving = {v.kind.value for v in result.proposal_evaluation.violations if index in v.agents}
            expected = [] if result.accepted else sorted(involving)
            assert record["shield_reason"] == expected
            assert bool(record["shield_reason"]) == (not result.accepted and bool(involving))

    expected_rejections = {_CONJUNCTION_SEED: 3, _MIXED_SEED: 1, _BENIGN_SEED: 0}[seed]
    assert rejected_steps == expected_rejections  # precondition: the seed exercises the intended branch


def test_a_rejected_conjunction_reason_names_the_d_safe_violation():
    env = ShieldedOrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)

    env.step(_no_op(env))

    for record in env.step_records():
        assert "d_safe" in record["shield_reason"]
        assert record["shield_reason"] == sorted(set(record["shield_reason"]))


@pytest.mark.parametrize("env_class", _ENV_CLASSES, ids=_IDS)
def test_satellites_that_collided_earlier_get_no_record(env_class):
    # Seed 14 at N=4, zero noise, 60 m hard body: sat_0/sat_1 collide under all-no-op (test_partial_termination).
    env = env_class(
        num_agents=4, episode_horizon_s=900.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=1800.0,
        hard_body_radius_m=60.0, noise_model=NoiseModel(0.0, 0.0),
    )
    env.reset(seed=14)
    while len(env.agents) == 4:
        env.step(_no_op(env))
    assert len(env.step_records()) == 4, "the satellites colliding on this step still get this step's record"
    assert env.agents, "precondition: the episode continues after the collision"
    remaining = list(env.agents)

    env.step(_no_op(env))

    assert [record["satellite_id"] for record in env.step_records()] == remaining


@pytest.mark.parametrize("env_class", _ENV_CLASSES, ids=_IDS)
def test_step_records_are_cleared_by_reset_and_are_copies(env_class):
    env = env_class(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)
    env.step(_no_op(env))

    records = env.step_records()
    records[0]["observed_state"].append(1.0)
    assert len(env.step_records()[0]["observed_state"]) == 6

    env.reset(seed=_CONJUNCTION_SEED)
    assert env.step_records() == []


def test_max_risk_pair_rejects_a_satellite_in_no_assessed_pair():
    env = OrbitGuardEnv(num_agents=2, **_KW)
    env.reset(seed=_CONJUNCTION_SEED)

    with pytest.raises(ValueError, match="satellite 5"):
        max_risk_pair(env._pending_assessments, 5)
