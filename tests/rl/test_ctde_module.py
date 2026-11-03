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

from orbit_guard.rl.actor_policy import NumpyActorPolicy, weights_from_actor
from orbit_guard.rl.ctde_module import CentralizedCriticTorchRLModule
from orbit_guard.rl.rllib_env import CTDEParallelPettingZooEnv
from tests.rl.stub_env import EPISODE_HORIZON_STEPS, StubOrbitGuardEnv

_LOCAL_DIM = 6
_GLOBAL_DIM = 12
_NUM_ACTIONS = 4


def _build_module(inference_only: bool = False, critic_input: str | None = None) -> CentralizedCriticTorchRLModule:
    model_config = {
        "local_obs_dim": _LOCAL_DIM,
        "actor_hidden_layers": (16,),
        "critic_hidden_layers": (16,),
    }
    if critic_input is not None:
        model_config["critic_input"] = critic_input
    return CentralizedCriticTorchRLModule(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (_LOCAL_DIM + _GLOBAL_DIM,), np.float32),
        action_space=gym.spaces.Discrete(_NUM_ACTIONS),
        model_config=model_config,
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


def test_the_default_critic_reads_only_the_global_state_slice():
    assert _build_module().critic[0].in_features == _GLOBAL_DIM
    assert _build_module(critic_input="global").critic[0].in_features == _GLOBAL_DIM


def test_the_agent_global_critic_reads_the_local_observation_and_the_global_state(observations):
    module = _build_module(critic_input="agent_global")
    local, global_state = observations

    assert module.critic[0].in_features == _LOCAL_DIM + _GLOBAL_DIM
    # Agent-specific global state (MAPPO): the same global state seen from two agents gives two values.
    assert not torch.allclose(
        module.compute_values(_batch(local, global_state)), module.compute_values(_batch(local + 5.0, global_state))
    )
    assert not torch.allclose(
        module.compute_values(_batch(local, global_state)), module.compute_values(_batch(local, global_state + 1.0))
    )


@pytest.mark.parametrize("forward_mode", ["forward_inference", "forward_exploration", "forward_train"])
def test_the_agent_global_actor_still_ignores_the_global_state_slice(forward_mode, observations):
    module = _build_module(critic_input="agent_global")
    local, global_state = observations
    forward = getattr(module, forward_mode)

    logits_a = forward(_batch(local, global_state))[Columns.ACTION_DIST_INPUTS]
    logits_b = forward(_batch(local, global_state * -7.0 + 3.0))[Columns.ACTION_DIST_INPUTS]
    obs = torch.cat([local, global_state], dim=-1).requires_grad_(True)
    module.forward_train({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS].sum().backward()

    assert torch.equal(logits_a, logits_b)
    assert torch.all(obs.grad[..., _LOCAL_DIM:] == 0)


def test_the_agent_global_inference_only_module_does_not_build_the_critic():
    assert not hasattr(_build_module(inference_only=True, critic_input="agent_global"), "critic")


def test_module_rejects_an_unknown_critic_input():
    with pytest.raises(ValueError, match="critic_input"):
        _build_module(critic_input="everything")


# --- Real OrbitGuard dimensions: the global state ends with an alive mask and the time remaining (critic-only) ---

_REAL_N = 2
_REAL_LOCAL = 54
_REAL_GLOBAL = _REAL_N * _REAL_LOCAL + _REAL_N + 1  # [slots | alive mask | time_remaining]
_REAL_EXTRA = _REAL_N + 1  # the entries appended after the slots


def _real_module(critic_input: str, inference_only: bool = False) -> CentralizedCriticTorchRLModule:
    return CentralizedCriticTorchRLModule(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (_REAL_LOCAL + _REAL_GLOBAL,), np.float32),
        action_space=gym.spaces.Discrete(6),
        model_config={
            "local_obs_dim": _REAL_LOCAL, "actor_hidden_layers": (32, 32),
            "critic_hidden_layers": (32,), "critic_input": critic_input,
        },
        inference_only=inference_only,
    )


def _real_obs(rows: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(rows, _REAL_LOCAL + _REAL_GLOBAL, generator=generator)


@pytest.mark.parametrize("critic_input", ["global", "agent_global"])
def test_the_critic_builds_with_the_alive_mask_and_time_remaining_dimension(critic_input):
    module = _real_module(critic_input)
    expected = _REAL_GLOBAL if critic_input == "global" else _REAL_LOCAL + _REAL_GLOBAL
    assert module.critic[0].in_features == expected
    assert module.compute_values({Columns.OBS: _real_obs()}).shape == (3,)


@pytest.mark.parametrize("critic_input", ["global", "agent_global"])
@pytest.mark.parametrize("forward_mode", ["forward_inference", "forward_exploration", "forward_train"])
def test_actor_logits_ignore_the_alive_mask_and_time_remaining(critic_input, forward_mode):
    module = _real_module(critic_input)
    obs = _real_obs()
    perturbed = obs.clone()
    perturbed[..., -_REAL_EXTRA:] = torch.tensor([0.0, 1.0, 0.0])  # one agent dead, horizon reached

    forward = getattr(module, forward_mode)
    assert torch.equal(
        forward({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS],
        forward({Columns.OBS: perturbed})[Columns.ACTION_DIST_INPUTS],
    )


@pytest.mark.parametrize("critic_input", ["global", "agent_global"])
def test_the_exported_numpy_actor_matches_the_module_actor_on_the_local_slice(critic_input):
    module = _real_module(critic_input)
    obs = _real_obs()
    policy = NumpyActorPolicy(weights_from_actor(module.actor))

    expected = module.forward_inference({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS].detach().numpy()
    np.testing.assert_allclose(policy.logits(obs[..., :_REAL_LOCAL].numpy()), expected, atol=1e-5)
    assert policy.weights.layers[0][0].shape[1] == _REAL_LOCAL


@pytest.mark.parametrize("critic_input", ["global", "agent_global"])
def test_the_critic_value_at_the_horizon_differs_from_an_early_step_through_the_time_feature(critic_input):
    """Two states identical except time_remaining (first decision 1.0 vs last 1/H) get different values."""
    stub = CTDEParallelPettingZooEnv(StubOrbitGuardEnv(num_agents=_REAL_N))
    observations, _ = stub.reset(seed=0)
    early = torch.as_tensor(observations["sat_0"])[None]
    while True:
        last = observations
        observations, _, _, truncations, _ = stub.step({agent: 0 for agent in stub.par_env.agents})
        if truncations["__all__"]:
            break
    final_decision = torch.as_tensor(last["sat_0"])[None]
    assert float(early[0, -1]) == 1.0
    assert float(final_decision[0, -1]) == pytest.approx(1.0 / EPISODE_HORIZON_STEPS)

    module = _real_module(critic_input)
    same_but_time = early.clone()
    same_but_time[0, -1] = final_decision[0, -1]
    assert not torch.allclose(module.compute_values({Columns.OBS: early}), module.compute_values({Columns.OBS: same_but_time}))

