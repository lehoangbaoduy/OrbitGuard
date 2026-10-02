"""Ray-free, picklable decentralized execution of a trained actor (PRD §28).

The shared actor is a plain tanh MLP over each satellite's own flattened
observation. Exporting its weights lets evaluation run any number of episodes
in worker processes with NumPy alone: no Ray, no torch, no critic -- so the
execution-time copy structurally cannot see a critic-only feature, and the
policy is trivially deterministic and picklable.

Flatten order matches Gymnasium's sorted-key Dict flatten used in training:
`global_summary`, then `neighbors` (row-major), then `own`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActorWeights:
    """`layers[i] = (weight (out, in), bias (out,))`; tanh between layers, none after the last."""

    layers: tuple[tuple[np.ndarray, np.ndarray], ...]


def flatten_local_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observation["global_summary"]).ravel(),
            np.asarray(observation["neighbors"]).ravel(),
            np.asarray(observation["own"]).ravel(),
        ]
    ).astype(np.float32)


def weights_from_actor(actor) -> ActorWeights:
    """Pull the Linear layers out of the torch `nn.Sequential` actor."""
    linears = [module for module in actor if module.__class__.__name__ == "Linear"]
    return ActorWeights(
        tuple(
            (module.weight.detach().cpu().numpy().astype(np.float32), module.bias.detach().cpu().numpy().astype(np.float32))
            for module in linears
        )
    )


def save_actor_weights(weights: ActorWeights, path: str | Path) -> None:
    arrays = {}
    for index, (weight, bias) in enumerate(weights.layers):
        arrays[f"w{index}"] = weight
        arrays[f"b{index}"] = bias
    np.savez(path, **arrays)


def load_actor_weights(path: str | Path) -> ActorWeights:
    with np.load(path) as data:
        count = len([key for key in data.files if key.startswith("w")])
        return ActorWeights(tuple((data[f"w{i}"], data[f"b{i}"]) for i in range(count)))


class NumpyActorPolicy:
    """Greedy (argmax) decentralized policy: one agent's own observation -> an action."""

    def __init__(self, weights: ActorWeights):
        self.weights = weights

    def logits(self, flat_observations: np.ndarray) -> np.ndarray:
        activation = np.asarray(flat_observations, dtype=np.float32)
        last = len(self.weights.layers) - 1
        for index, (weight, bias) in enumerate(self.weights.layers):
            activation = activation @ weight.T + bias
            if index < last:
                activation = np.tanh(activation)
        return activation

    def __call__(self, agent: str, observation: dict[str, np.ndarray]) -> int:
        return int(np.argmax(self.logits(flatten_local_observation(observation)[None])[0]))


def export_from_algo(algo, path: str | Path) -> ActorWeights:
    """Export a trained RLlib algorithm's shared actor to `path` (a `.npz`)."""
    from orbit_guard.rl.ctde_module import POLICY_ID

    weights = weights_from_actor(algo.get_module(POLICY_ID).actor)
    save_actor_weights(weights, path)
    return weights
