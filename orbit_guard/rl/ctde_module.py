"""MAPPO/CTDE RLModule for RLlib's new API stack (PRD §28).

Ray 2.58 ships no centralized-critic example for the RLModule/Catalog stack
(only the legacy Policy-API one), so the pattern here is built from the
documented primitives: a `TorchRLModule` implementing `ValueFunctionAPI`.

Each agent's observation vector arrives as ``[local | global_state]``, where
``local`` is the agent's own flattened decentralized observation (PRD §24) and
``global_state`` is the environment's `state()` -- every agent's local
observation (a collided agent's zeroed), a per-agent alive mask and the
normalized time remaining (layout in `rl.global_state`), appended by
`rl.rllib_env`. The time feature makes Ray 2.58's forced terminal at the
horizon a Markov finite-horizon terminal for the critic. The module splits it:

  * ACTOR  -- an MLP over ``obs[..., :local_dim]`` ONLY.
  * CRITIC -- an MLP over ``obs[..., local_dim:]`` (the global state) with
    ``critic_input="global"`` (the default), or over the whole
    ``[local | global_state]`` vector with ``critic_input="agent_global"`` --
    MAPPO's "agent-specific global state", the B arm of the pre-registered
    critic A/B (prereg R4, CORE-4b). Either way only the critic changes.

The hard architectural rule of PRD §28 -- no critic-only feature may reach
decentralized execution -- is enforced structurally: the actor never reads the
global slice, in any forward mode, and `inference_only` modules (the
execution-time copy RLlib builds for env runners) do not even construct the
critic. One module instance is shared by every agent (parameter sharing), so
the actor and the centralized critic are both shared across satellites.
"""

from __future__ import annotations

from typing import Any

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

POLICY_ID = "shared_policy"
DEFAULT_ACTOR_HIDDEN_LAYERS = (128, 128)
DEFAULT_CRITIC_HIDDEN_LAYERS = (256, 256)
CRITIC_INPUT_GLOBAL = "global"
CRITIC_INPUT_AGENT_GLOBAL = "agent_global"
CRITIC_INPUTS = (CRITIC_INPUT_GLOBAL, CRITIC_INPUT_AGENT_GLOBAL)
DEFAULT_CRITIC_INPUT = CRITIC_INPUT_GLOBAL


def validate_critic_input(critic_input: str) -> str:
    """`critic_input` itself, or ValueError when it is not one of CRITIC_INPUTS."""
    if critic_input not in CRITIC_INPUTS:
        raise ValueError(f"critic_input must be one of {list(CRITIC_INPUTS)}, got {critic_input!r}")
    return critic_input


def _mlp(input_dim: int, hidden_layers: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.Tanh()])
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class CentralizedCriticTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """Parameter-shared actor over local obs + centralized critic over global state.

    `model_config` keys: ``local_obs_dim`` (required), ``actor_hidden_layers``,
    ``critic_hidden_layers``, ``critic_input`` (``"global"`` or ``"agent_global"``).
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        self.local_obs_dim = int(self.model_config["local_obs_dim"])
        total_obs_dim = int(self.observation_space.shape[0])
        global_dim = total_obs_dim - self.local_obs_dim
        if self.local_obs_dim < 1 or global_dim < 1:
            raise ValueError(
                f"observation of size {total_obs_dim} must hold a local slice "
                f"({self.local_obs_dim}) and a non-empty global-state slice"
            )

        actor_layers = tuple(self.model_config.get("actor_hidden_layers", DEFAULT_ACTOR_HIDDEN_LAYERS))
        critic_layers = tuple(self.model_config.get("critic_hidden_layers", DEFAULT_CRITIC_HIDDEN_LAYERS))
        self.critic_input = validate_critic_input(self.model_config.get("critic_input", DEFAULT_CRITIC_INPUT))
        # The slice of the observation the critic reads: the global state, or [own local | global state].
        self._critic_start = self.local_obs_dim if self.critic_input == CRITIC_INPUT_GLOBAL else 0

        self.actor = _mlp(self.local_obs_dim, actor_layers, int(self.action_space.n))
        if not self.inference_only:
            self.critic = _mlp(total_obs_dim - self._critic_start, critic_layers, 1)

    def _actor_logits(self, batch: dict[str, Any]) -> TensorType:
        local_obs = batch[Columns.OBS][..., : self.local_obs_dim]
        return self.actor(local_obs)

    @override(TorchRLModule)
    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {Columns.ACTION_DIST_INPUTS: self._actor_logits(batch)}

    @override(TorchRLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {Columns.ACTION_DIST_INPUTS: self._actor_logits(batch)}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> TensorType:
        critic_obs = batch[Columns.OBS][..., self._critic_start :]
        return self.critic(critic_obs).squeeze(-1)
