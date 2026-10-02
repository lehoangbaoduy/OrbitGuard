"""Trace meta schema (redesign real-data rule 3, EC-27/EC-28): episodes identify themselves; nothing is inferred
from filenames, and a missing field fails loudly naming exactly what is absent."""

from __future__ import annotations

import pytest

from orbit_guard.visualization.replay_meta import (
    REPLAY_META_SCHEMA,
    ReplayMeta,
    ReplayMetaError,
    build_replay_meta,
    scenario_id,
    validate_replay_meta,
)

_BASE = {
    "condition": "normal",
    "seed": 7,
    "policy": "trained",
    "shield": True,
    "model": "adaptive_mappo",
    "n_agents": 15,
    "delta_v_budget_mps": 2.0,
    "max_delta_v_per_action_mps": 0.1,
    "weights_sha256": "a" * 64,
}


def _meta(**overrides):
    return build_replay_meta(**{**_BASE, **overrides})


def test_build_then_validate_round_trips():
    meta = validate_replay_meta(_meta())
    assert isinstance(meta, ReplayMeta)
    assert meta.scenario_id == "normal:7"
    assert (meta.policy, meta.shield, meta.n_agents) == ("trained", True, 15)
    assert meta.delta_v_budget_mps == 2.0 and meta.weights_sha256 == "a" * 64


def test_scenario_id_is_condition_and_seed():
    assert scenario_id("capability_reduction", 10) == "capability_reduction:10"


def test_built_meta_is_plain_json_types():
    import json

    assert json.loads(json.dumps(_meta()))["schema"] == REPLAY_META_SCHEMA


@pytest.mark.parametrize("field", ["condition", "seed", "policy", "shield", "model", "n_agents", "delta_v_budget_mps",
                                   "max_delta_v_per_action_mps", "scenario_id", "schema"])
def test_missing_required_field_fails_loudly_and_names_it(field):
    meta = _meta()
    del meta[field]
    with pytest.raises(ReplayMetaError, match=field):
        validate_replay_meta(meta)


def test_all_missing_fields_are_reported_together():
    with pytest.raises(ReplayMetaError) as error:
        validate_replay_meta({"model": "x", "seed": 1})
    for field in ("condition", "policy", "shield", "scenario_id"):
        assert field in str(error.value)


def test_an_old_trace_meta_without_the_schema_gets_a_rerecord_hint():
    with pytest.raises(ReplayMetaError, match="record_replay_traces"):
        validate_replay_meta({"model": "rule_based + shield (demo)", "condition": "normal", "seed": 111080})


def test_unknown_policy_is_rejected():
    with pytest.raises(ReplayMetaError, match="policy"):
        validate_replay_meta({**_meta(), "policy": "magic"})


@pytest.mark.parametrize("shield", [1, "on", None])
def test_shield_must_be_a_real_bool(shield):
    with pytest.raises(ReplayMetaError, match="shield"):
        validate_replay_meta({**_meta(), "shield": shield})


def test_scenario_id_must_match_condition_and_seed():
    with pytest.raises(ReplayMetaError, match="scenario_id"):
        validate_replay_meta({**_meta(), "scenario_id": "normal:8"})


def test_trained_policy_requires_a_weights_hash():
    meta = _meta()
    del meta["weights_sha256"]
    with pytest.raises(ReplayMetaError, match="weights_sha256"):
        validate_replay_meta(meta)


def test_untrained_policy_requires_the_init_seed():
    with pytest.raises(ReplayMetaError, match="init_seed"):
        validate_replay_meta(_meta(policy="untrained", weights_sha256=None))
    assert validate_replay_meta(_meta(policy="untrained", weights_sha256=None, init_seed=0)).init_seed == 0


def test_baseline_policies_need_neither_weights_nor_init_seed():
    assert validate_replay_meta(_meta(policy="no_maneuver", weights_sha256=None)).policy == "no_maneuver"


@pytest.mark.parametrize("field,value", [("n_agents", 0), ("delta_v_budget_mps", 0.0), ("delta_v_budget_mps", -1.0),
                                         ("max_delta_v_per_action_mps", 0.0), ("seed", "7")])
def test_numeric_fields_are_range_and_type_checked(field, value):
    with pytest.raises(ReplayMetaError, match=field):
        validate_replay_meta({**_meta(), field: value})


def test_validation_does_not_mutate_its_input():
    meta = _meta()
    snapshot = dict(meta)
    validate_replay_meta(meta)
    assert meta == snapshot
