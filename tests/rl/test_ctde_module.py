"""Tests for the centralized-critic RLModule (PRD §28's CTDE hard rule).

The rule: the critic may see global state during training, but no critic-only
feature may reach decentralized execution. These tests prove it structurally:
the actor's output is independent of the global slice in every forward mode
(and has exactly zero gradient w.r.t. it), while the critic's output depends
on the global slice.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from orbit_guard.rl.ctde_module import CentralizedCriticTorchRLModule

_LOCAL_DIM = 6
_GLOBAL_DIM = 12
_NUM_ACTIONS = 4


def _build_module(inference_only: bool = False) -> CentralizedCriticTorchRLModule:
    return CentralizedCriticTorchRLModule(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (_LOCAL_DIM + _GLOBAL_DIM,), np.float32),
        action_space=gym.spaces.Discrete(_NUM_ACTIONS),
        model_config={
            "local_obs_dim": _LOCAL_DIM,
            "actor_hidden_layers": (16,),
            "critic_hidden_layers": (16,),
        },
        inference_only=inference_only,
    )


def _batch(local: torch.Tensor, global_state: torch.Tensor) -> dict:
    return {Columns.OBS: torch.cat([local, global_state], dim=-1)}


@pytest.fixture
def observations() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(5, _LOCAL_DIM, generator=generator), torch.randn(5, _GLOBAL_DIM, generator=generator)


@pytest.mark.parametrize("forward_mode", ["forward_inference", "forward_exploration", "forward_train"])
def test_actor_logits_are_independent_of_the_global_state_slice(forward_mode, observations):
    module = _build_module()
    local, global_state = observations
    forward = getattr(module, forward_mode)

    logits_a = forward(_batch(local, global_state))[Columns.ACTION_DIST_INPUTS]
    logits_b = forward(_batch(local, global_state * -7.0 + 3.0))[Columns.ACTION_DIST_INPUTS]

    assert torch.equal(logits_a, logits_b)


def test_actor_has_zero_gradient_with_respect_to_the_global_state_slice(observations):
    module = _build_module()
    local, global_state = observations
    obs = torch.cat([local, global_state], dim=-1).requires_grad_(True)

    module.forward_train({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS].sum().backward()

    assert torch.all(obs.grad[..., _LOCAL_DIM:] == 0)
    assert torch.any(obs.grad[..., :_LOCAL_DIM] != 0)


def test_critic_value_depends_on_the_global_state_slice(observations):
    module = _build_module()
    local, global_state = observations

    values_a = module.compute_values(_batch(local, global_state))
    values_b = module.compute_values(_batch(local, global_state + 1.0))

    assert values_a.shape == (5,)
    assert not torch.allclose(values_a, values_b)


def test_critic_is_shared_across_agents_by_sharing_one_module(observations):
    module = _build_module()
    local, global_state = observations
    batch = _batch(local, global_state)

    # Same global state seen from two different agents' local views -> same value.
    other_local = local + 5.0
    assert torch.allclose(module.compute_values(batch), module.compute_values(_batch(other_local, global_state)))


def test_inference_only_module_does_not_build_the_critic():
    module = _build_module(inference_only=True)

    assert not hasattr(module, "critic")
    assert hasattr(module, "actor")


def test_module_rejects_an_observation_without_a_global_state_slice():
    with pytest.raises(ValueError):
        CentralizedCriticTorchRLModule(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (_LOCAL_DIM,), np.float32),
            action_space=gym.spaces.Discrete(_NUM_ACTIONS),
            model_config={"local_obs_dim": _LOCAL_DIM},
        )
