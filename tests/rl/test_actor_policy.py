"""Tests for orbit_guard.rl.actor_policy: Ray-free, picklable decentralized execution."""

from __future__ import annotations

import pickle

import numpy as np
import torch
from gymnasium.spaces.utils import flatten

from orbit_guard.rl.actor_policy import (
    NumpyActorPolicy,
    flatten_local_observation,
    load_actor_weights,
    save_actor_weights,
    weights_from_actor,
)
from orbit_guard.rl.ctde_module import _mlp
from orbit_guard.rl.environment import OrbitGuardEnv

_FAST = {"episode_horizon_s": 120.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 60.0}


def _actor(local_dim: int) -> torch.nn.Sequential:
    torch.manual_seed(0)
    return _mlp(local_dim, (128, 128), 6)


def test_flattening_matches_gymnasiums_sorted_key_order_used_in_training():
    env = OrbitGuardEnv(num_agents=2, **_FAST)
    observations, _ = env.reset(seed=1)
    agent = env.agents[0]

    ours = flatten_local_observation(observations[agent])
    reference = flatten(env.observation_space(agent), observations[agent])

    np.testing.assert_allclose(ours, reference, atol=1e-7)
    assert ours.dtype == np.float32


def test_numpy_actor_reproduces_the_torch_actor_logits():
    actor = _actor(54)
    weights = weights_from_actor(actor)
    policy = NumpyActorPolicy(weights)
    batch = np.random.default_rng(3).normal(size=(16, 54)).astype(np.float32)

    with torch.no_grad():
        expected = actor(torch.as_tensor(batch)).numpy()

    np.testing.assert_allclose(policy.logits(batch), expected, atol=1e-4)


def test_greedy_action_is_the_argmax_of_the_actor():
    actor = _actor(54)
    policy = NumpyActorPolicy(weights_from_actor(actor))
    env = OrbitGuardEnv(num_agents=2, **_FAST)
    observations, _ = env.reset(seed=2)
    agent = env.agents[0]

    action = policy(agent, observations[agent])

    with torch.no_grad():
        expected = int(torch.argmax(actor(torch.as_tensor(flatten_local_observation(observations[agent]))[None]), dim=-1))
    assert action == expected


def test_weights_round_trip_through_disk(tmp_path):
    weights = weights_from_actor(_actor(54))
    path = tmp_path / "actor.npz"

    save_actor_weights(weights, path)
    loaded = load_actor_weights(path)

    for (w1, b1), (w2, b2) in zip(weights.layers, loaded.layers, strict=True):
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(b1, b2)


def test_policy_is_picklable_for_multiprocess_evaluation():
    policy = NumpyActorPolicy(weights_from_actor(_actor(54)))

    clone = pickle.loads(pickle.dumps(policy))

    batch = np.ones((2, 54), dtype=np.float32)
    np.testing.assert_array_equal(clone.logits(batch), policy.logits(batch))
