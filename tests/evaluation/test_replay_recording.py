"""The re-runnable recorder for the before/after replay episodes (PR1, advisor point 2): output on disk, full meta,
weights hashes, and a deterministic untrained policy built the same way the trainer builds its actor."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from orbit_guard.evaluation.replay_recording import (
    SCENARIOS,
    EpisodePlan,
    build_plan,
    record_plan_item,
    untrained_actor_weights,
)
from orbit_guard.rl.actor_policy import save_actor_weights
from orbit_guard.visualization.episodes import discover_episodes
from tests.evaluation.replay_fixtures import ENV_KWARGS
from tests.evaluation.sidecar_fixtures import sidecar_record

_DIMS = {"obs_dim": 54, "num_actions": 6}


def test_untrained_weights_are_deterministic_in_the_init_seed():
    a, b, c = (untrained_actor_weights(**_DIMS, init_seed=seed) for seed in (0, 0, 1))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a.layers, b.layers, strict=True))
    assert not np.array_equal(a.layers[0][0], c.layers[0][0])


def test_untrained_weights_have_the_trainers_actor_shape():
    layers = untrained_actor_weights(**_DIMS, init_seed=0).layers
    assert [w.shape for w, _ in layers] == [(128, 54), (128, 128), (6, 128)]


def test_building_untrained_weights_does_not_disturb_global_torch_rng():
    import torch

    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    untrained_actor_weights(**_DIMS, init_seed=7)
    assert torch.equal(torch.rand(3), expected)


def test_plan_is_three_scenarios_by_two_policies_by_two_shield_settings(tmp_path):
    plan = build_plan(tmp_path)
    assert len(plan) == 12 == len(SCENARIOS) * 4
    assert len({(p.condition, p.seed, p.policy, p.shield) for p in plan}) == 12


def test_trained_episodes_use_the_matching_shield_variant_weights(tmp_path):
    plan = build_plan(tmp_path, trained_seed=2)
    trained = {p.shield: p for p in plan if p.policy == "trained"}
    assert trained[True].model == "adaptive_mappo" and trained[True].weights_path == tmp_path / "n15_adaptive_s2" / "actor_weights.npz"
    assert trained[False].model == "adaptive_mappo_no_shield"
    assert trained[False].weights_path == tmp_path / "n15_noshield_s2" / "actor_weights.npz"


def test_untrained_episodes_need_no_weights_file(tmp_path):
    assert all(p.weights_path is None for p in build_plan(tmp_path) if p.policy == "untrained")


def _trained_checkpoint(run, *, shield: bool):
    """A stand-in trained actor with the sidecar a 2-agent default-config run would write (the recorder refuses
    a trained checkpoint without one)."""
    run.mkdir()
    path = run / "actor_weights.npz"
    save_actor_weights(untrained_actor_weights(**_DIMS, init_seed=5), path)
    (run / "run_config.json").write_text(json.dumps(sidecar_record(None, shield=shield, num_agents=2)))
    return path


@pytest.fixture
def weights_file(tmp_path):
    return _trained_checkpoint(tmp_path / "shielded_run", shield=True)


@pytest.fixture
def unshielded_weights_file(tmp_path):
    return _trained_checkpoint(tmp_path / "unshielded_run", shield=False)


def _record(tmp_path, item, **overrides):
    return record_plan_item(item, num_agents=2, out_dir=tmp_path / "out", init_seed=0, env_kwargs=ENV_KWARGS, **overrides)


def test_recorded_trained_episode_has_full_meta_and_the_weights_hash(tmp_path, weights_file):
    item = EpisodePlan("normal", 0, "trained", True, "adaptive_mappo", weights_file)
    path = _record(tmp_path, item)
    (entry,) = discover_episodes(path.parent).entries
    assert entry.meta.policy == "trained" and entry.meta.shield is True and entry.meta.n_agents == 2
    assert entry.meta.weights_sha256 == hashlib.sha256(weights_file.read_bytes()).hexdigest()
    assert entry.meta.delta_v_budget_mps == pytest.approx(2.0)
    assert entry.meta.max_delta_v_per_action_mps == pytest.approx(0.1)


def test_recorded_untrained_episode_records_its_init_seed_and_persists_the_weights(tmp_path):
    item = EpisodePlan("normal", 0, "untrained", False, "untrained_policy", None)
    path = _record(tmp_path, item)
    (entry,) = discover_episodes(path.parent).entries
    assert entry.meta.policy == "untrained" and entry.meta.init_seed == 0 and entry.meta.shield is False
    assert (path.parent / "weights" / "untrained_actor_init0.npz").is_file()  # beside the traces, not among them


def test_shield_off_episode_has_no_shield_interventions(tmp_path, unshielded_weights_file):
    item = EpisodePlan("normal", 0, "trained", False, "adaptive_mappo_no_shield", unshielded_weights_file)
    path = _record(tmp_path, item)
    from orbit_guard.visualization.replay import load_trace

    assert not load_trace(path).intervened.any()


def test_file_names_are_informational_only(tmp_path, weights_file):
    path = _record(tmp_path, EpisodePlan("normal", 0, "trained", True, "adaptive_mappo", weights_file))
    assert path.suffix == ".npz" and "normal" in path.name and "shield-on" in path.name
