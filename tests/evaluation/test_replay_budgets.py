"""Replay capability flags under per-episode Δv budgets (replay fix b).

With budgets drawn per episode from capability.remaining_delta_v_range_mps, a starting tank below the nominal
budget no longer means "limited": the trace meta records each satellite's drawn budget and which satellites the
capability_reduction degradation actually cut, and the replay flags exactly those. Traces recorded before these
fields existed still load and keep the old rule.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from orbit_guard.evaluation.shifts import condition_env_kwargs
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.evaluation import no_maneuver_policy
from orbit_guard.visualization.derived.capability import capability_flags
from orbit_guard.visualization.payload import build_payload
from orbit_guard.visualization.replay import load_trace, record_episode, save_trace
from orbit_guard.visualization.replay_meta import (
    DEGRADED_KEY,
    DRAWN_BUDGETS_KEY,
    build_replay_meta,
)
from orbit_guard.visualization.source import ReplaySource
from tests.evaluation.replay_fixtures import ENV_KWARGS, capability_trace

# The fixture kwargs minus their pinned fixed budget: the config's per-episode range applies, as in production.
_RANDOMIZED = {key: value for key, value in ENV_KWARGS.items() if key != "remaining_delta_v_range_mps"}
_N = 6
_SEED = 4  # normal: two satellites draw below the 2.0 m/s nominal budget
_REDUCTION_SEED = 1  # capability_reduction: two UNDEGRADED satellites draw below nominal


def _record(condition: str, seed: int = _SEED):
    env = OrbitGuardEnv(num_agents=_N, **_RANDOMIZED, **condition_env_kwargs(condition))
    meta = build_replay_meta(
        condition=condition, seed=seed, policy="no_maneuver", shield=False, model="no_maneuver", n_agents=_N,
        delta_v_budget_mps=env.delta_v_budget_mps, max_delta_v_per_action_mps=env.max_delta_v_per_action_mps,
    )
    return env, record_episode(env, no_maneuver_policy, seed=seed, meta=meta)


def test_the_recorded_meta_carries_each_satellites_drawn_budget_and_degraded_flag():
    env, trace = _record("capability_reduction", _REDUCTION_SEED)
    provenance = env.budget_provenance()
    assert trace.meta[DRAWN_BUDGETS_KEY] == list(provenance.drawn_mps)
    assert trace.meta[DEGRADED_KEY] == list(provenance.degraded)
    json.dumps(trace.meta)  # plain JSON, so save_trace can write it


def test_with_randomized_budgets_and_condition_normal_no_satellite_is_flagged_limited():
    _env, trace = _record("normal")
    nominal = trace.meta["delta_v_budget_mps"]
    assert capability_flags(trace, nominal).limited.any(), "precondition: the old rule over-flags this draw"

    capability = ReplaySource(trace).capability
    assert not capability.limited.any() and not capability.offline.any()


def test_under_capability_reduction_exactly_the_degraded_satellites_are_flagged():
    env, trace = _record("capability_reduction", _REDUCTION_SEED)
    degraded = np.array(env.budget_provenance().degraded)
    enabled = np.asarray(trace.maneuver_enabled, dtype=bool)
    untouched = enabled & ~degraded
    assert (trace.remaining_dv_mps[0][untouched] < trace.meta["delta_v_budget_mps"]).any(), (
        "precondition: an undegraded satellite drew a budget below nominal"
    )

    capability = ReplaySource(trace).capability
    assert capability.limited.tolist() == (enabled & degraded).tolist()
    assert int(capability.limited.sum()) == 2 and int(capability.offline.sum()) == 2


def test_fuel_bars_keep_each_satellites_own_starting_tank():
    env, trace = _record("capability_reduction", _REDUCTION_SEED)
    fuel = ReplaySource(trace).fuel
    assert fuel.budget_mps.tolist() == pytest.approx(list(env.initial_delta_v_budgets_mps().values()))


def test_the_budget_meta_survives_save_and_load(tmp_path):
    _env, trace = _record("capability_reduction", _REDUCTION_SEED)
    save_trace(trace, tmp_path / "episode.npz")
    loaded = ReplaySource(load_trace(tmp_path / "episode.npz"))
    assert loaded.capability.limited.tolist() == ReplaySource(trace).capability.limited.tolist()


def test_an_old_format_trace_without_the_new_fields_still_loads_and_keeps_the_old_rule(tmp_path):
    trace = capability_trace()
    legacy_meta = {k: v for k, v in trace.meta.items() if k not in (DRAWN_BUDGETS_KEY, DEGRADED_KEY, "config_digest")}
    save_trace(replace(trace, meta=legacy_meta), tmp_path / "legacy.npz")
    source = ReplaySource(load_trace(tmp_path / "legacy.npz"))

    assert source.meta.degraded is None and source.meta.drawn_delta_v_budgets_mps is None
    assert source.capability.offline.tolist() == [False, False, False, True]
    assert source.capability.limited.tolist() == [True, False, False, False]
    assert build_payload(source)["capability"] == {"offline": [3], "limited": [0]}
