"""Completeness of a model -> training seed -> checkpoint registry (PRD §33; review2: a failed run went unnoticed).

Every learned model (`policy: mappo` in the config's `models:` block) must be registered with exactly the training
seeds {0, ..., rl.num_training_seeds - 1}. A queue run that failed leaves its run directory without actor weights,
so without this check the final test would silently report that model over fewer training seeds than the others,
and the one-way freeze would lock the gap in. `allow_partial` tolerates missing seeds only: a learned model with no
checkpoint at all, or an unexpected seed, is always refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orbit_guard.evaluation.baselines import MODEL_NAMES, _spec_from_config

ALLOW_PARTIAL_FLAG = "--allow-partial"


class IncompleteRegistryError(ValueError):
    """A learned model is registered with other training seeds than the config's rl.num_training_seeds."""


def learned_models(config: Mapping[str, Any]) -> tuple[str, ...]:
    """The models in MODEL_NAMES whose policy is learned (MAPPO), in MODEL_NAMES order."""
    return tuple(model for model in MODEL_NAMES if _spec_from_config(model, config)["policy"] == "mappo")


def expected_training_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    """(0, ..., n-1) for n = rl.num_training_seeds, which must be a positive integer."""
    count = config["rl"]["num_training_seeds"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"rl.num_training_seeds must be a positive integer, got {count!r}")
    return tuple(range(count))


def check_registry_complete(
    registry: Mapping[str, Mapping[int, Any]], config: Mapping[str, Any], *, allow_partial: bool = False
) -> None:
    """Raise `IncompleteRegistryError` naming every learned model whose registered seeds are not the expected set."""
    expected = set(expected_training_seeds(config))
    problems = []
    for model in learned_models(config):
        seeds = {int(seed) for seed in registry.get(model, {})}
        missing, extra = sorted(expected - seeds), sorted(seeds - expected)
        if not seeds:
            problems.append(f"{model} has no checkpoint (expected training seeds {sorted(expected)})")
            continue
        if extra:
            problems.append(f"{model} has unexpected training seeds {extra} (expected {sorted(expected)})")
        if missing and not allow_partial:
            problems.append(f"{model} is missing training seeds {missing}")
    if problems:
        raise IncompleteRegistryError(
            f"the registry does not hold every learned model's training seeds: {'; '.join(problems)}. A queue run "
            f"probably failed: retrain it, or pass {ALLOW_PARTIAL_FLAG} to evaluate with missing seeds knowingly "
            "(the freeze records it; it never excuses a model with no checkpoint or an unexpected seed)"
        )
