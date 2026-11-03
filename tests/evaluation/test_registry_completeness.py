"""A missing training seed is refused, not silently evaluated over fewer seeds (review2, medium).

Every learned model (`policy: mappo` in the config's `models:` block) must be registered with exactly the training
seeds {0..rl.num_training_seeds-1}. The `registry` and `freeze` commands refuse anything else unless
`--allow-partial` is passed, which only tolerates missing seeds (never a learned model with none, never an
unexpected seed) and is recorded in the freeze snapshot.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH, load_config
from orbit_guard.evaluation import experiment
from orbit_guard.evaluation.registry_checks import (
    IncompleteRegistryError,
    check_registry_complete,
    expected_training_seeds,
    learned_models,
)

_LEARNED = ("fixed_margin_mappo", "adaptive_mappo", "adaptive_mappo_no_shield")
_VARIANTS = ("adaptive", "fixed", "noshield")


@pytest.fixture
def config() -> dict:
    return load_config(DEFAULT_CONFIG_PATH)


def _registry(seeds=(0, 1, 2), models=_LEARNED) -> dict[str, dict[int, str]]:
    return {model: {seed: f"/runs/{model}_s{seed}/actor_weights.npz" for seed in seeds} for model in models}


def test_the_learned_models_and_seeds_come_from_the_config(config):
    assert set(learned_models(config)) == set(_LEARNED)
    assert expected_training_seeds(config) == (0, 1, 2)  # configs/default.yaml rl.num_training_seeds: 3


def test_a_complete_registry_passes(config):
    check_registry_complete(_registry(), config)


def test_a_missing_training_seed_is_refused_and_named(config):
    registry = _registry()
    del registry["adaptive_mappo_no_shield"][2]

    with pytest.raises(IncompleteRegistryError, match=r"adaptive_mappo_no_shield.*\[2\].*--allow-partial"):
        check_registry_complete(registry, config)


def test_allow_partial_tolerates_a_missing_seed(config):
    registry = _registry()
    del registry["adaptive_mappo_no_shield"][2]

    check_registry_complete(registry, config, allow_partial=True)


@pytest.mark.parametrize("allow_partial", [False, True])
def test_a_learned_model_with_no_checkpoint_is_always_refused(config, allow_partial):
    registry = _registry(models=("adaptive_mappo", "fixed_margin_mappo"))

    with pytest.raises(IncompleteRegistryError, match="adaptive_mappo_no_shield"):
        check_registry_complete(registry, config, allow_partial=allow_partial)


@pytest.mark.parametrize("allow_partial", [False, True])
def test_an_unexpected_training_seed_is_always_refused(config, allow_partial):
    registry = _registry()
    registry["adaptive_mappo"][3] = "/runs/extra/actor_weights.npz"

    with pytest.raises(IncompleteRegistryError, match=r"adaptive_mappo.*unexpected.*\[3\]"):
        check_registry_complete(registry, config, allow_partial=allow_partial)


def test_the_expected_seed_count_follows_the_config(config):
    two = copy.deepcopy(config)
    two["rl"]["num_training_seeds"] = 2

    check_registry_complete(_registry(seeds=(0, 1)), two)
    with pytest.raises(IncompleteRegistryError):
        check_registry_complete(_registry(), two)


@pytest.mark.parametrize("bad", [0, -1, "3", True, 2.0])
def test_a_malformed_seed_count_is_rejected(config, bad):
    config["rl"]["num_training_seeds"] = bad

    with pytest.raises(ValueError, match="num_training_seeds"):
        expected_training_seeds(config)


# --- the registry command ---------------------------------------------------------------------------------------


def _runs(tmp_path: Path, skip: tuple[str, ...] = ()) -> Path:
    runs = tmp_path / "runs"
    for variant in _VARIANTS:
        for seed in (0, 1, 2):
            run = runs / f"n15_{variant}_s{seed}"
            run.mkdir(parents=True)
            if run.name not in skip:  # a failed queue run leaves its directory without weights
                (run / "actor_weights.npz").write_bytes(b"w")
    return runs


def test_the_registry_command_refuses_a_failed_run_and_writes_nothing(tmp_path):
    out = tmp_path / "registry.json"

    with pytest.raises(IncompleteRegistryError, match=r"adaptive_mappo_no_shield.*\[2\]"):
        experiment.main(["registry", "--runs-dir", str(_runs(tmp_path, ("n15_noshield_s2",))), "--out", str(out)])

    assert not out.exists()


def test_the_registry_command_writes_a_partial_registry_only_when_asked(tmp_path):
    out = tmp_path / "registry.json"

    experiment.main([
        "registry", "--runs-dir", str(_runs(tmp_path, ("n15_noshield_s2",))), "--out", str(out), "--allow-partial",
    ])

    assert json.loads(out.read_text())["adaptive_mappo_no_shield"].keys() == {"0", "1"}


def test_the_registry_command_writes_a_complete_registry(tmp_path):
    out = tmp_path / "registry.json"

    experiment.main(["registry", "--runs-dir", str(_runs(tmp_path)), "--out", str(out)])

    assert all(set(seeds) == {"0", "1", "2"} for seeds in json.loads(out.read_text()).values())
